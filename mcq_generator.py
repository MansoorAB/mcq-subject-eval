"""
MCQ Generation Module
Uses Gemma-3-4B-IT model to generate multiple choice questions from text chunks
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Optional
import os
from logger_config import get_logger

logger = get_logger("mcq_generator")


class MCQGenerator:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Initialize MCQ generator with Gemma model
        
        Args:
            model_path: Path to the model (if None, uses default cache location)
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.device = device
        self.model_path = model_path or "google/gemma-3-4b-it"
        
        logger.info(f"Loading Gemma-3-4B-IT model from: {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu" if device == "cpu" else "auto",
            trust_remote_code=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
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
    
    def generate_mcq(self, text_chunk: str, num_questions: int = 1, max_length: int = 1024) -> Dict:
        """
        Generate MCQ from text chunk
        
        Args:
            text_chunk: Text to generate questions from
            num_questions: Number of questions to generate
            max_length: Maximum length of generated response
            
        Returns:
            Dictionary containing generated questions or error
        """
        try:
            # Create prompt
            prompt = self.create_mcq_prompt(text_chunk, num_questions)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Leave room for response
                padding=True
            ).to(self.device)
            
            logger.info(f"Generating MCQ for chunk with {len(text_chunk)} characters...")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            logger.info(f"Generated response: {len(response)} characters")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    return {
                        'success': True,
                        'questions': parsed_response.get('questions', []),
                        'raw_response': response
                    }
                else:
                    # If no JSON found, return the raw response
                    return {
                        'success': False,
                        'error': 'No valid JSON found in response',
                        'raw_response': response
                    }
                    
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'JSON parsing error: {str(e)}',
                    'raw_response': response
                }
                
        except Exception as e:
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
            time.sleep(1)
        
        return results


if __name__ == "__main__":
    # Test the MCQ generator
    generator = MCQGenerator(device="cpu")
    
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
