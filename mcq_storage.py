"""
MCQ Storage Module
Handles saving and loading of generated MCQs in JSON format
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from logger_config import get_logger

logger = get_logger("mcq_storage")


class MCQStorage:
    def __init__(self, storage_dir: str = "generated_mcq"):
        """
        Initialize MCQ storage system
        
        Args:
            storage_dir: Directory to store MCQ files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"MCQ storage initialized - Directory: {self.storage_dir.absolute()}")
    
    def _generate_filename(self, source_filename: str, timestamp: Optional[str] = None) -> str:
        """
        Generate a unique filename for MCQ storage
        
        Args:
            source_filename: Original PDF filename
            timestamp: Optional timestamp (if None, uses current time)
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean the source filename
        clean_name = Path(source_filename).stem.replace(" ", "_").replace(".", "_")
        
        return f"mcq_{clean_name}_{timestamp}.json"
    
    def save_mcqs(self, 
                  source_filename: str,
                  chunks: List[Dict],
                  mcq_results: List[Dict],
                  metadata: Optional[Dict] = None) -> str:
        """
        Save generated MCQs to JSON file
        
        Args:
            source_filename: Original PDF filename
            chunks: List of text chunks used for generation
            mcq_results: List of MCQ generation results
            metadata: Optional metadata about the generation process
            
        Returns:
            Path to the saved JSON file
        """
        try:
            # Generate filename
            filename = self._generate_filename(source_filename)
            filepath = self.storage_dir / filename
            
            # Prepare data structure
            data = {
                "metadata": {
                    "source_filename": source_filename,
                    "generation_timestamp": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "total_questions": sum(len(result.get('questions', [])) for result in mcq_results if result.get('success')),
                    "successful_generations": len([r for r in mcq_results if r.get('success')]),
                    "failed_generations": len([r for r in mcq_results if not r.get('success')]),
                    **(metadata or {})
                },
                "chunks": chunks,
                "mcq_results": mcq_results
            }
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"MCQs saved to: {filepath}")
            logger.info(f"Total questions saved: {data['metadata']['total_questions']}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save MCQs: {str(e)}")
            raise
    
    def _extract_questions_summary(self, mcq_results: List[Dict]) -> List[Dict]:
        """
        Extract question summaries (without full duplication) from results
        
        Args:
            mcq_results: List of MCQ generation results
            
        Returns:
            List of question summaries with references to full data
        """
        questions = []
        
        for i, result in enumerate(mcq_results):
            if result.get('success') and result.get('questions'):
                for j, question in enumerate(result['questions']):
                    question_summary = {
                        "question_id": f"chunk_{result.get('chunk_id', i)}_q_{j+1}",
                        "chunk_id": result.get('chunk_id', i),
                        "chunk_tokens": result.get('chunk_tokens', 0),
                        "question_preview": question.get('question', '')[:100] + "..." if len(question.get('question', '')) > 100 else question.get('question', ''),
                        "correct_answer": question.get('correct_answer', ''),
                        "generation_success": True,
                        "full_question_ref": f"mcq_results[{i}].questions[{j}]"
                    }
                    questions.append(question_summary)
            else:
                # Add failed generation entry
                failed_question = {
                    "question_id": f"chunk_{result.get('chunk_id', i)}_failed",
                    "chunk_id": result.get('chunk_id', i),
                    "chunk_tokens": result.get('chunk_tokens', 0),
                    "question_preview": "",
                    "correct_answer": "",
                    "generation_success": False,
                    "error": result.get('error', 'Unknown error'),
                    "full_question_ref": f"mcq_results[{i}]"
                }
                questions.append(failed_question)
        
        return questions
    
    def get_all_questions(self, mcq_data: Dict) -> List[Dict]:
        """
        Extract all questions from mcq_results with metadata
        
        Args:
            mcq_data: Loaded MCQ data
            
        Returns:
            List of all questions with chunk metadata
        """
        all_questions = []
        
        for i, result in enumerate(mcq_data.get('mcq_results', [])):
            if result.get('success') and result.get('questions'):
                for j, question in enumerate(result['questions']):
                    question_with_metadata = {
                        "question_id": f"chunk_{result.get('chunk_id', i)}_q_{j+1}",
                        "chunk_id": result.get('chunk_id', i),
                        "chunk_tokens": result.get('chunk_tokens', 0),
                        "question": question.get('question', ''),
                        "options": question.get('options', {}),
                        "correct_answer": question.get('correct_answer', ''),
                        "explanation": question.get('explanation', ''),
                        "generation_success": True
                    }
                    all_questions.append(question_with_metadata)
            elif not result.get('success'):
                # Add failed generation entry
                failed_question = {
                    "question_id": f"chunk_{result.get('chunk_id', i)}_failed",
                    "chunk_id": result.get('chunk_id', i),
                    "chunk_tokens": result.get('chunk_tokens', 0),
                    "question": "",
                    "options": {},
                    "correct_answer": "",
                    "explanation": "",
                    "generation_success": False,
                    "error": result.get('error', 'Unknown error')
                }
                all_questions.append(failed_question)
        
        return all_questions
    
    def get_full_question_data(self, mcq_data: Dict, question_ref: str) -> Dict:
        """
        Get full question data using the reference from questions summary
        
        Args:
            mcq_data: Loaded MCQ data
            question_ref: Reference like "mcq_results[0].questions[0]"
            
        Returns:
            Full question data or None if not found
        """
        try:
            # Parse reference like "mcq_results[0].questions[0]"
            if question_ref.startswith("mcq_results["):
                # Extract indices
                parts = question_ref.replace("mcq_results[", "").replace("]", "").split("].questions[")
                result_idx = int(parts[0])
                
                if len(parts) > 1:
                    question_idx = int(parts[1])
                    return mcq_data['mcq_results'][result_idx]['questions'][question_idx]
                else:
                    return mcq_data['mcq_results'][result_idx]
            
            return None
        except (IndexError, KeyError, ValueError) as e:
            logger.warning(f"Could not resolve question reference {question_ref}: {str(e)}")
            return None
    
    def load_mcqs(self, filename: str) -> Dict:
        """
        Load MCQs from JSON file
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            Loaded MCQ data
        """
        try:
            filepath = self.storage_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"MCQ file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"MCQs loaded from: {filepath}")
            logger.info(f"Loaded {data['metadata']['total_questions']} questions")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load MCQs: {str(e)}")
            raise
    
    def list_mcq_files(self) -> List[Dict]:
        """
        List all MCQ files in the storage directory
        
        Returns:
            List of file information
        """
        try:
            files = []
            for filepath in self.storage_dir.glob("*.json"):
                try:
                    # Load metadata only
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    files.append({
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "source_filename": data.get('metadata', {}).get('source_filename', 'Unknown'),
                        "generation_timestamp": data.get('metadata', {}).get('generation_timestamp', ''),
                        "total_questions": data.get('metadata', {}).get('total_questions', 0),
                        "file_size": filepath.stat().st_size
                    })
                except Exception as e:
                    logger.warning(f"Could not read metadata from {filepath}: {str(e)}")
                    files.append({
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "source_filename": "Unknown",
                        "generation_timestamp": "",
                        "total_questions": 0,
                        "file_size": filepath.stat().st_size
                    })
            
            # Sort by timestamp (newest first)
            files.sort(key=lambda x: x['generation_timestamp'], reverse=True)
            
            logger.info(f"Found {len(files)} MCQ files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list MCQ files: {str(e)}")
            return []
    
    def delete_mcq_file(self, filename: str) -> bool:
        """
        Delete an MCQ file
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = self.storage_dir / filename
            
            if not filepath.exists():
                logger.warning(f"File not found for deletion: {filepath}")
                return False
            
            filepath.unlink()
            logger.info(f"Deleted MCQ file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete MCQ file {filename}: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the storage system
    storage = MCQStorage()
    
    # Test data
    test_chunks = [
        {"id": 0, "text": "Sample text chunk 1", "token_count": 100},
        {"id": 1, "text": "Sample text chunk 2", "token_count": 150}
    ]
    
    test_results = [
        {
            "success": True,
            "chunk_id": 0,
            "chunk_tokens": 100,
            "questions": [
                {
                    "question": "What is machine learning?",
                    "options": {"A": "A type of AI", "B": "A programming language", "C": "A database", "D": "A web framework"},
                    "correct_answer": "A",
                    "explanation": "Machine learning is a subset of AI"
                }
            ]
        },
        {
            "success": False,
            "chunk_id": 1,
            "chunk_tokens": 150,
            "error": "Generation failed"
        }
    ]
    
    # Test save
    saved_file = storage.save_mcqs("test_document.pdf", test_chunks, test_results)
    print(f"Saved to: {saved_file}")
    
    # Test load
    loaded_data = storage.load_mcqs(os.path.basename(saved_file))
    print(f"Loaded {loaded_data['metadata']['total_questions']} questions")
    
    # Test getting all questions
    all_questions = storage.get_all_questions(loaded_data)
    if all_questions:
        print(f"All questions extracted: {len(all_questions)}")
        print(f"First question: {all_questions[0]['question'][:50]}...")
    
    # Test list
    files = storage.list_mcq_files()
    print(f"Found {len(files)} files")
