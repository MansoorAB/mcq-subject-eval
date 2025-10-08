"""
MCQ Generation Module with GGUF Support
Uses quantized Gemma-3-4B-IT model via llama-cpp-python for memory efficiency
"""

import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from llama_cpp import Llama
from logger_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger("mcq_generator_gguf")


class MCQGeneratorGGUF:
    def __init__(self, model_path: str = None, model_filename: str = None, device: str = None):
        """
        Initialize MCQ generator with GGUF quantized model
        
        Args:
            model_path: Hugging Face model path (default from env)
            model_filename: Specific GGUF filename (default from env)
            device: Device to run on (default from env)
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', 'unsloth/gemma-3-4b-it-GGUF')
        self.model_filename = model_filename or os.getenv('MODEL_FILENAME', 'gemma-3-4b-it-Q4_0.gguf')
        self.device = device or os.getenv('DEVICE', 'cpu')
        
        # Hugging Face token for downloading
        self.hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        
        logger.info(f"Initializing GGUF MCQ Generator")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Model filename: {self.model_filename}")
        logger.info(f"Device: {self.device}")
        
        # Download model if needed
        self.model_file_path = self._download_model()
        
        # Load model
        self._load_model()
    
    def _download_model(self) -> str:
        """
        Download the GGUF model file if not present locally
        
        Returns:
            Path to the local model file
        """
        from huggingface_hub import hf_hub_download
        
        try:
            # Check if model exists in cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = os.path.join(cache_dir, f"models--{self.model_path.replace('/', '--')}")
            
            if os.path.exists(model_cache_path):
                # Look for the specific GGUF file in cache
                for root, dirs, files in os.walk(model_cache_path):
                    if self.model_filename in files:
                        local_path = os.path.join(root, self.model_filename)
                        logger.info(f"Found cached model: {local_path}")
                        return local_path
            
            # Download the model
            logger.info(f"Downloading model {self.model_filename} from {self.model_path}")
            
            if self.hf_token:
                logger.info("Using Hugging Face token for download")
                local_path = hf_hub_download(
                    repo_id=self.model_path,
                    filename=self.model_filename,
                    token=self.hf_token,
                    cache_dir=cache_dir
                )
            else:
                logger.warning("No HF token provided - download may fail for gated models")
                local_path = hf_hub_download(
                    repo_id=self.model_path,
                    filename=self.model_filename,
                    cache_dir=cache_dir
                )
            
            logger.info(f"Model downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise Exception(f"Model download failed: {str(e)}")
    
    def _load_model(self):
        """Load the GGUF model using llama-cpp-python"""
        try:
            logger.info(f"Loading GGUF model from: {self.model_file_path}")
            
            # Configure model parameters for CPU
            model_kwargs = {
                'model_path': self.model_file_path,
                'n_ctx': 4096,  # Context window
                'n_threads': os.cpu_count(),  # Use all CPU cores
                'verbose': False,
                'n_gpu_layers': 0,  # CPU only
            }
            
            # Add MPS support if available on macOS
            if self.device == "mps" and hasattr(self, '_check_mps_support'):
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        model_kwargs['n_gpu_layers'] = -1  # Use MPS
                        logger.info("Using MPS acceleration")
                except ImportError:
                    logger.info("PyTorch not available, using CPU")
            
            self.model = Llama(**model_kwargs)
            logger.info("GGUF model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")
    
    def create_mcq_prompt(self, text_chunk: str, num_questions: int = 1) -> str:
        """
        Create a prompt for MCQ generation
        
        Args:
            text_chunk: Text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""<start_of_turn>user
You are an expert educator creating multiple choice questions. Based on the following text, create {num_questions} high-quality multiple choice question(s).

Text:
{text_chunk}

For each question, provide:
1. A clear, well-formatted question
2. 4 answer options (A, B, C, D)
3. The correct answer marked with [CORRECT]
4. A brief explanation of why the correct answer is right

Format your response as JSON with this structure:
{{
    "questions": [
        {{
            "question": "Your question here?",
            "options": {{
                "A": "Option A",
                "B": "Option B", 
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A",
            "explanation": "Brief explanation of why this is correct"
        }}
    ]
}}

Make sure the questions test understanding, not just memorization. Focus on key concepts, relationships, and applications.<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def generate_mcq(self, text_chunk: str, num_questions: int = 1, max_tokens: int = None) -> Dict:
        """
        Generate MCQ from text chunk using GGUF model
        
        Args:
            text_chunk: Text to generate questions from
            num_questions: Number of questions to generate
            max_tokens: Maximum tokens to generate (default from env)
            
        Returns:
            Dictionary containing generated questions or error
        """
        try:
            max_tokens = max_tokens or int(os.getenv('MAX_NEW_TOKENS', 1024))
            
            # Create prompt
            prompt = self.create_mcq_prompt(text_chunk, num_questions)
            
            logger.info(f"Generating MCQ for chunk with {len(text_chunk)} characters...")
            
            # Generate response using GGUF model
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<end_of_turn>", "\n\n\n"],
                echo=False
            )
            
            # Extract text from response
            generated_text = response['choices'][0]['text']
            logger.info(f"Generated response: {len(generated_text)} characters")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    return {
                        'success': True,
                        'questions': parsed_response.get('questions', []),
                        'raw_response': generated_text
                    }
                else:
                    # If no JSON found, return the raw response
                    return {
                        'success': False,
                        'error': 'No valid JSON found in response',
                        'raw_response': generated_text
                    }
                    
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'JSON parsing error: {str(e)}',
                    'raw_response': generated_text
                }
                
        except Exception as e:
            logger.error(f"MCQ generation failed: {str(e)}")
            return {
                'success': False,
                'error': f'Generation error: {str(e)}',
                'raw_response': ''
            }
    
    def generate_multiple_mcqs(self, chunks: List[Dict], questions_per_chunk: int = 1) -> List[Dict]:
        """
        Generate MCQs for multiple text chunks
        
        Args:
            chunks: List of text chunks
            questions_per_chunk: Number of questions per chunk
            
        Returns:
            List of results for each chunk
        """
        results = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            logger.debug(f"Chunk text preview: {chunk['text'][:100]}...")
            
            result = self.generate_mcq(
                chunk['text'], 
                num_questions=questions_per_chunk
            )
            
            # Add chunk metadata to result
            result['chunk_id'] = chunk['id']
            result['chunk_text'] = chunk['text']
            result['chunk_tokens'] = chunk['token_count']
            
            results.append(result)
            
            if result['success']:
                logger.info(f"Successfully generated {len(result.get('questions', []))} questions for chunk {i+1}")
            else:
                logger.warning(f"Failed to generate questions for chunk {i+1}: {result.get('error', 'Unknown error')}")
            
            # Add a small delay to prevent overwhelming the system
            import time
            time.sleep(0.5)  # Shorter delay for GGUF model
        
        return results


if __name__ == "__main__":
    # Test the GGUF MCQ generator
    generator = MCQGeneratorGGUF()
    
    # Test with sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. There are three main types of machine learning: supervised 
    learning, unsupervised learning, and reinforcement learning. Supervised learning uses 
    labeled data to train models, while unsupervised learning finds patterns in unlabeled data.
    """
    
    result = generator.generate_mcq(sample_text, num_questions=1)
    print("Generated MCQ:")
    print(json.dumps(result, indent=2))
