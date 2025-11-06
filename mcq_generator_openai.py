"""
MCQ Generation Module using OpenAI API
Uses GPT-4o-mini via OpenAI API for fast inference
"""

import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from logger_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger("mcq_generator_openai")


class MCQGeneratorOpenAI:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize MCQ generator with OpenAI API
        
        Args:
            api_key: OpenAI API key (default from env)
            model_name: OpenAI model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file")
        
        logger.info(f"Initializing OpenAI MCQ Generator")
        logger.info(f"Model: {self.model_name}")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI client initialized successfully!")
    
    def create_mcq_prompt(self, text_chunk: str, num_questions: int = 1) -> str:
        """
        Create a prompt for MCQ generation
        
        Args:
            text_chunk: Text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert educator creating multiple choice questions. Based on the following text, create {num_questions} high-quality multiple choice question(s).

Text:
{text_chunk}

For each question, provide:
1. A clear, well-formatted question
2. 4 answer options (A, B, C, D)
3. The correct answer (just the letter: A, B, C, or D)
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

Make sure the questions test understanding, not just memorization. Focus on key concepts, relationships, and applications. Return ONLY valid JSON, no additional text."""
        return prompt
    
    def generate_mcq(self, text_chunk: str, num_questions: int = 1, max_tokens: int = 1024) -> Dict:
        """
        Generate MCQ from text chunk using OpenAI API
        
        Args:
            text_chunk: Text to generate questions from
            num_questions: Number of questions to generate
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing generated questions or error
        """
        try:
            # Create prompt
            prompt = self.create_mcq_prompt(text_chunk, num_questions)
            
            logger.info(f"Generating MCQ for chunk with {len(text_chunk)} characters using {self.model_name}...")
            
            # Call OpenAI API
            # Try with JSON response format first (supported by GPT-4o-mini)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert educator who creates high-quality multiple choice questions. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}  # Force JSON response
                )
            except Exception as e:
                # Fallback if JSON response format is not supported
                logger.warning(f"JSON response format not supported, falling back to standard format: {str(e)}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert educator who creates high-quality multiple choice questions. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
            
            # Extract response text
            generated_text = response.choices[0].message.content
            logger.info(f"Generated response: {len(generated_text)} characters")
            
            # Parse JSON response
            try:
                parsed_response = json.loads(generated_text)
                
                # Ensure questions is a list
                questions = parsed_response.get('questions', [])
                if not isinstance(questions, list):
                    # If questions is not a list, try to extract it
                    if isinstance(parsed_response, dict) and 'questions' in parsed_response:
                        questions = parsed_response['questions']
                    else:
                        questions = []
                
                return {
                    'success': True,
                    'questions': questions,
                    'raw_response': generated_text
                }
                    
            except json.JSONDecodeError as e:
                # Try to extract JSON from response if it's wrapped in text
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        parsed_response = json.loads(json_str)
                        questions = parsed_response.get('questions', [])
                        return {
                            'success': True,
                            'questions': questions if isinstance(questions, list) else [],
                            'raw_response': generated_text
                        }
                    except json.JSONDecodeError:
                        pass
                
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
        
        return results


if __name__ == "__main__":
    # Test the OpenAI MCQ generator
    generator = MCQGeneratorOpenAI()
    
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

