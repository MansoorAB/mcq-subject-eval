"""
FastAPI Web Application for MCQ Generation
Provides a web interface for uploading multiple PDFs and generating MCQs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import random
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import traceback
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from mcq_generator import MCQGenerator
from mcq_generator_gguf import MCQGeneratorGGUF
from mcq_generator_openai import MCQGeneratorOpenAI
from mcq_storage import MCQStorage
from logger_config import get_logger

# Load environment variables
load_dotenv()

app = FastAPI(title="MCQ Generator API", version="2.0.0")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
MAX_FILES = 3  # Maximum number of PDFs that can be uploaded at once

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
logger = get_logger("fastapi_app")
mcq_storage = MCQStorage()

# Choose generator type based on environment
USE_GGUF = os.getenv('USE_GGUF', 'true').lower() == 'true'

# Global MCQ generators (lazy loading) - separate instances for each model type
mcq_generator_openai = None
mcq_generator_gemma = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_mcq_generator(model_type: str = "openai"):
    """
    Initialize MCQ generator (lazy loading)
    
    Args:
        model_type: "openai" or "gemma"
    
    Returns:
        Initialized MCQ generator instance
    """
    global mcq_generator_openai, mcq_generator_gemma
    
    if model_type == "openai":
        if mcq_generator_openai is None:
            logger.info("Initializing OpenAI MCQ generator...")
            try:
                mcq_generator_openai = MCQGeneratorOpenAI()
                logger.info("OpenAI MCQ generator initialized!")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI generator: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize OpenAI generator: {str(e)}. Please check your OPENAI_API_KEY in .env file."
                )
        return mcq_generator_openai
    else:  # gemma
        if mcq_generator_gemma is None:
            logger.info("Initializing Gemma MCQ generator...")
            if USE_GGUF:
                logger.info("Using GGUF quantized model")
                mcq_generator_gemma = MCQGeneratorGGUF()
            else:
                logger.info("Using standard transformers model")
                mcq_generator_gemma = MCQGenerator(device=os.getenv('DEVICE', 'cpu'))
            logger.info("Gemma MCQ generator initialized!")
        return mcq_generator_gemma


def select_chunks_for_questions(chunks: List[Dict], num_questions: int, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Select chunks for question generation with 3-chunk difference rule
    
    Args:
        chunks: List of all chunks
        num_questions: Number of questions to generate
        chunk_size: Chunk size from config
        overlap: Overlap from config
        
    Returns:
        List of selected chunks for question generation
    """
    if len(chunks) <= num_questions:
        return chunks
    
    selected_chunks = []
    used_indices = set()
    
    # Calculate minimum gap between selected chunks (3 chunks difference)
    min_gap = 3
    
    while len(selected_chunks) < num_questions and len(used_indices) < len(chunks):
        # Get available indices (not used and not within min_gap of used indices)
        available_indices = []
        for i in range(len(chunks)):
            if i not in used_indices:
                # Check if this index is far enough from all used indices
                too_close = any(abs(i - used_idx) < min_gap for used_idx in used_indices)
                if not too_close:
                    available_indices.append(i)
        
        if not available_indices:
            # If no indices available with min_gap, reduce gap and try again
            min_gap = max(1, min_gap - 1)
            continue
        
        # Randomly select from available indices
        selected_idx = random.choice(available_indices)
        selected_chunks.append(chunks[selected_idx])
        used_indices.add(selected_idx)
    
    return selected_chunks


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_mcqs(
    files: List[UploadFile] = File(...),
    questions_per_pdf: int = Form(default=10),
    chunk_size: int = Form(default=1000),
    overlap: int = Form(default=100),
    model_type: str = Form(default="openai")
):
    """
    Generate MCQs from uploaded PDF files
    
    Args:
        files: List of uploaded PDF files (max 3)
        questions_per_pdf: Number of questions to generate per PDF
        chunk_size: Size of text chunks (from config)
        overlap: Overlap between chunks (from config)
        model_type: Model to use - "openai" or "gemma" (default: "openai")
    """
    try:
        # Validate model_type
        if model_type not in ["openai", "gemma"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {model_type}. Must be 'openai' or 'gemma'."
            )
        
        # Validate number of files
        if len(files) > MAX_FILES:
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum {MAX_FILES} PDF files allowed. You uploaded {len(files)} files."
            )
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Validate files
        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are allowed. File '{file.filename}' is not a PDF."
                )
        
        # Initialize MCQ generator based on model_type
        logger.info(f"Using model type: {model_type}")
        generator = initialize_mcq_generator(model_type=model_type)
        
        # Process each PDF
        all_results = []
        all_questions = []
        
        for file_idx, file in enumerate(files):
            logger.info(f"Processing file {file_idx + 1}/{len(files)}: {file.filename}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Process PDF
                processor = PDFProcessor(chunk_size=chunk_size, overlap=overlap)
                chunks = processor.process_pdf(temp_file_path)
                
                if not chunks:
                    logger.error(f"No text could be extracted from PDF: {file.filename}")
                    all_results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'No text could be extracted from PDF',
                        'chunks': [],
                        'questions': []
                    })
                    continue
                
                # Select chunks for question generation
                selected_chunks = select_chunks_for_questions(
                    chunks, questions_per_pdf, chunk_size, overlap
                )
                
                logger.info(f"Selected {len(selected_chunks)} chunks for question generation from {len(chunks)} total chunks")
                
                # Generate MCQs for selected chunks
                pdf_results = []
                pdf_questions = []
                
                for chunk_idx, chunk in enumerate(selected_chunks):
                    logger.info(f"Generating question {chunk_idx + 1}/{len(selected_chunks)} for {file.filename}")
                    
                    # Generate one question per chunk
                    result = generator.generate_mcq(chunk['text'], num_questions=1)
                    
                    # Add chunk metadata to result
                    result['chunk_id'] = chunk['id']
                    result['chunk_text'] = chunk['text']
                    result['chunk_tokens'] = chunk['token_count']
                    result['filename'] = file.filename
                    
                    pdf_results.append(result)
                    
                    # If generation was successful, add to questions list
                    if result['success'] and result.get('questions'):
                        for question in result['questions']:
                            question_with_metadata = {
                                'filename': file.filename,
                                'chunk_id': chunk['id'],
                                'chunk_tokens': chunk['token_count'],
                                'question': question.get('question', ''),
                                'options': question.get('options', {}),
                                'correct_answer': question.get('correct_answer', ''),
                                'explanation': question.get('explanation', ''),
                                'generation_success': True
                            }
                            pdf_questions.append(question_with_metadata)
                    else:
                        # Add failed generation entry
                        failed_question = {
                            'filename': file.filename,
                            'chunk_id': chunk['id'],
                            'chunk_tokens': chunk['token_count'],
                            'question': '',
                            'options': {},
                            'correct_answer': '',
                            'explanation': '',
                            'generation_success': False,
                            'error': result.get('error', 'Unknown error')
                        }
                        pdf_questions.append(failed_question)
                
                # Save MCQs to JSON file for this PDF
                try:
                    saved_file = mcq_storage.save_mcqs(
                        source_filename=file.filename,
                        chunks=chunks,
                        mcq_results=pdf_results,
                        metadata={
                            'chunk_size': chunk_size,
                            'overlap': overlap,
                            'questions_per_pdf': questions_per_pdf,
                            'selected_chunks': len(selected_chunks),
                            'total_chunks': len(chunks),
                            'file_index': file_idx + 1,
                            'total_files': len(files),
                            'model_type': model_type
                        }
                    )
                    logger.info(f"MCQs saved to: {saved_file}")
                except Exception as e:
                    logger.error(f"Failed to save MCQs for {file.filename}: {str(e)}")
                    saved_file = None
                
                # Add to overall results
                total_questions = sum(1 for q in pdf_questions if q['generation_success'])
                all_results.append({
                    'filename': file.filename,
                    'success': True,
                    'total_chunks': len(chunks),
                    'selected_chunks': len(selected_chunks),
                    'total_questions': total_questions,
                    'saved_file': saved_file,
                    'results': pdf_results
                })
                
                all_questions.extend(pdf_questions)
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        
        # Prepare response
        total_questions_generated = sum(1 for q in all_questions if q['generation_success'])
        
        response_data = {
            'success': True,
            'total_files': len(files),
            'total_questions': total_questions_generated,
            'questions_per_pdf': questions_per_pdf,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'model_type': model_type,
            'file_results': all_results,
            'all_questions': all_questions
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'Processing failed: {str(e)}')


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MCQ Generator API is running"}


@app.get("/model-info")
async def model_info(model_type: str = "openai"):
    """
    Get information about the loaded model
    
    Args:
        model_type: "openai" or "gemma" (default: "openai")
    """
    try:
        if model_type == "openai":
            try:
                generator = initialize_mcq_generator("openai")
                return {
                    'model_name': generator.model_name,
                    'model_type': 'OpenAI API',
                    'model_loaded': True,
                    'provider': 'OpenAI'
                }
            except Exception as e:
                return {
                    'model_name': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    'model_type': 'OpenAI API',
                    'model_loaded': False,
                    'error': str(e),
                    'provider': 'OpenAI'
                }
        else:  # gemma
            try:
                generator = initialize_mcq_generator("gemma")
                if USE_GGUF:
                    return {
                        'model_name': os.getenv('MODEL_PATH', 'unsloth/gemma-3-4b-it-GGUF'),
                        'model_filename': os.getenv('MODEL_FILENAME', 'gemma-3-4b-it-Q4_0.gguf'),
                        'model_type': 'GGUF Quantized',
                        'device': generator.device,
                        'model_loaded': True,
                        'provider': 'Local'
                    }
                else:
                    return {
                        'model_name': 'google/gemma-3-4b-it',
                        'model_type': 'Standard Transformers',
                        'device': generator.device,
                        'model_loaded': True,
                        'provider': 'Local'
                    }
            except Exception as e:
                return {
                    'model_name': os.getenv('MODEL_PATH', 'unsloth/gemma-3-4b-it-GGUF'),
                    'model_type': 'GGUF Quantized' if USE_GGUF else 'Standard Transformers',
                    'device': os.getenv('DEVICE', 'cpu'),
                    'model_loaded': False,
                    'error': str(e),
                    'provider': 'Local'
                }
    except Exception as e:
        return {
            'model_type': 'Unknown',
            'model_loaded': False,
            'error': str(e)
        }


@app.get("/mcq-files")
async def list_mcq_files():
    """List all saved MCQ files"""
    try:
        files = mcq_storage.list_mcq_files()
        return {
            'success': True,
            'files': files,
            'total_files': len(files)
        }
    except Exception as e:
        logger.error(f"Failed to list MCQ files: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Failed to list files: {str(e)}')


@app.get("/mcq-files/{filename}")
async def get_mcq_file(filename: str):
    """Get a specific MCQ file"""
    try:
        data = mcq_storage.load_mcqs(filename)
        return {
            'success': True,
            'data': data
        }
    except Exception as e:
        logger.error(f"Failed to load MCQ file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Failed to load file: {str(e)}')


@app.delete("/mcq-files/{filename}")
async def delete_mcq_file(filename: str):
    """Delete a specific MCQ file"""
    try:
        success = mcq_storage.delete_mcq_file(filename)
        if success:
            return {'success': True, 'message': f'File {filename} deleted successfully'}
        else:
            raise HTTPException(status_code=404, detail=f'File {filename} not found')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete MCQ file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Failed to delete file: {str(e)}')


# Setup templates
templates = Jinja2Templates(directory="templates")

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting MCQ Generator FastAPI Application...")
    logger.info("Model will be loaded on first use to save startup time.")
    uvicorn.run(app, host='0.0.0.0', port=8000)
