# MCQ Generator v2.0 - Multiple PDF to Multiple Choice Questions

A GenAI-powered FastAPI application that converts multiple PDF documents into multiple choice questions using Google's Gemma-3-4B-IT model running locally on CPU.

## Features

- 📄 **Multiple PDF Processing**: Upload up to 3 PDFs simultaneously for batch processing
- 🤖 **AI-Powered**: Uses Google's Gemma-3-4B-IT model for question generation
- 🖥️ **CPU Optimized**: Runs efficiently on CPU (no GPU required for development)
- 🌐 **Modern Web Interface**: Beautiful, responsive FastAPI-based web UI with tabbed interface
- ⚙️ **Smart Chunking**: Intelligent chunk selection with 3-chunk difference rule for diverse questions
- 🎯 **Configurable Questions**: Set number of questions per PDF (default: 10)
- 📊 **Advanced UI**: Tabbed interface with question viewer and filtering capabilities
- 🧪 **Testing**: Built-in test functionality for quick experimentation
- 📝 **Comprehensive Logging**: Timestamped log files with detailed operation tracking
- 💾 **JSON Storage**: Automatic saving of generated MCQs in structured JSON format
- 📁 **File Management**: List, view, and delete saved MCQ files via API
- 🚀 **FastAPI**: Modern, fast, and auto-documented API with interactive docs

## Prerequisites

- Python 3.12 
- 8GB+ RAM (16GB recommended for optimal performance)
- Hugging Face account and token (for downloading models)
- Gemma-3-4B-IT model (will be downloaded automatically)

## Installation

1. **Activate your conda environment**:
   You need to have Anaconda installed on your machine
   Go to the project directory and run...
   
   ```bash
   conda create -n your_env_name python=3.12 -y
   conda activate your_env_name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your Hugging Face token
   # Get your token from: https://huggingface.co/settings/tokens
   ```

4. **Configure your .env file**:
   ```bash
   # Required: Add your HF token (with quotes)
   HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
   
   # Optional: Customize model settings
   MODEL_PATH="unsloth/gemma-3-4b-it-GGUF"
   MODEL_FILENAME="gemma-3-4b-it-Q4_0.gguf"
   DEVICE="cpu"
   USE_GGUF=true
   ```

## Usage

### 1. Start the Application

**Option A: Using the startup script (Recommended)**
```bash
python run_fastapi.py
```

**Option B: Direct FastAPI command**
```bash
python app.py
```

**Option C: Using uvicorn directly**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

### 2. Web Interface

1. Open your browser and go to `http://localhost:8000`
2. **Upload Multiple PDFs**: Drag and drop up to 3 PDF files or click to select
3. **Configure Settings**:
   - **Questions per PDF**: Number of questions to generate per PDF (default: 10)
   - **Chunk Size**: Number of tokens per text chunk (default: 1000, read-only)
   - **Overlap**: Token overlap between chunks (default: 100, read-only)
4. Click "Generate MCQs" and wait for processing
5. **View Results**: 
   - See overall statistics and per-file results
   - Switch to "View Questions" tab to see all generated questions
   - Filter questions by file or success status

### 3. API Documentation

FastAPI provides automatic interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 4. Testing

Run the test suite to verify everything works:
```bash
python test_fastapi.py
```

## Project Structure

```
03_mcq_testing_app/
├── app.py                 # FastAPI web application (NEW)
├── flask_app.py          # Original Flask application (renamed)
├── run_fastapi.py        # FastAPI startup script (NEW)
├── test_fastapi.py       # FastAPI test suite (NEW)
├── pdf_processor.py      # PDF text extraction and chunking
├── mcq_generator.py      # AI model integration and MCQ generation
├── mcq_storage.py        # JSON storage and file management
├── logger_config.py      # Logging configuration
├── requirements.txt      # Python dependencies (updated)
├── test_app.py          # Original Flask test suite
├── templates/
│   └── index.html       # Web interface (redesigned)
├── logs/                # Log files (auto-created)
├── generated_mcq/       # Saved MCQ JSON files (auto-created)
├── uploads/             # Temporary file storage (auto-created)
└── README.md           # This file
```

## API Endpoints

### FastAPI Endpoints (v2.0)
- `GET /` - Main web interface with multiple PDF support
- `POST /generate` - Upload multiple PDFs and generate MCQs
- `GET /health` - Health check
- `GET /model-info` - Model status information
- `GET /mcq-files` - List all saved MCQ files
- `GET /mcq-files/{filename}` - Get specific MCQ file
- `DELETE /mcq-files/{filename}` - Delete specific MCQ file
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Legacy Flask Endpoints (v1.0)
- `GET /` - Original web interface (single PDF)
- `POST /upload` - Upload single PDF and generate MCQs
- `POST /test-generation` - Test MCQ generation with text input

## Configuration

### PDF Processing Settings

- **Chunk Size**: Controls how much text is processed at once (default: 1000)
  - Smaller chunks (400-600): More focused questions, faster processing
  - Larger chunks (800-1200): More comprehensive questions, slower processing

- **Overlap**: Ensures context continuity between chunks (default: 100)
  - Higher overlap: Better context preservation
  - Lower overlap: Faster processing, less redundancy

- **Smart Chunk Selection**: Intelligent algorithm selects chunks with 3-chunk difference
  - Ensures diverse question generation from different parts of the document
  - Prevents questions from overlapping content areas
  - Random selection from available chunks meeting the distance criteria

### Model Settings

The application supports two model configurations:

**GGUF Quantized Model (Default - Recommended for Mac)**:
- **Model**: `unsloth/gemma-3-4b-it-GGUF`
- **File**: `gemma-3-4b-it-Q4_0.gguf` (2.37 GB)
- **Memory Usage**: ~3-4 GB RAM
- **Performance**: Fast inference on CPU
- **Library**: llama-cpp-python

**Standard Transformers Model**:
- **Model**: `google/gemma-3-4b-it`
- **Device**: CPU (optimized for local development)
- **Precision**: Float32 (CPU-compatible)
- **Memory Usage**: ~8-12 GB RAM
- **Max Tokens**: 1024 per generation

### Logging and Storage

- **Log Files**: Automatically created in `./logs/` with timestamp format `mcq_generator_HH_MM_SS.log`
- **MCQ Storage**: Generated questions saved in `./generated_mcq/` as JSON files
- **File Naming**: `mcq_{source_filename}_{timestamp}.json`
- **Log Levels**: DEBUG (file), INFO (console), with detailed operation tracking

## Performance Notes

### Performance Comparison

**GGUF Quantized Model (Recommended)**:
- **First Load**: 30-60 seconds
- **Question Generation**: 10-30 seconds per question
- **Memory Usage**: ~3-4GB RAM
- **Model Size**: 2.37 GB download

**Standard Transformers Model**:
- **First Load**: 2-3 minutes
- **Question Generation**: 30-60 seconds per question
- **Memory Usage**: ~8-12GB RAM
- **Model Size**: 8+ GB download

### Optimization Tips
1. **Smaller Chunks**: Use 400-600 tokens for faster processing
2. **Fewer Questions**: Generate 1 question per chunk initially
3. **Model Caching**: Model stays loaded after first use
4. **Batch Processing**: Process multiple chunks sequentially

## Troubleshooting

### Common Issues

1. **Model Not Found**:
   ```
   Error: Model not found in cache
   ```
   - Ensure Gemma-3-4B-IT is downloaded to `~/.cache/huggingface/hub`
   - Run: `huggingface-cli download google/gemma-3-4B-it`

2. **Out of Memory**:
   ```
   Error: CUDA out of memory
   ```
   - The app is configured for CPU, but if you see this, restart the application
   - Ensure you have 8GB+ available RAM

3. **PDF Processing Error**:
   ```
   Error: No text could be extracted from PDF
   ```
   - Ensure the PDF contains selectable text (not just images)
   - Try a different PDF file

4. **Slow Performance**:
   - Reduce chunk size to 400-600 tokens
   - Generate only 1 question per chunk
   - Close other memory-intensive applications

### Debug Mode

Run with debug logging:
```bash
export FLASK_DEBUG=1
python app.py
```

## Next Steps

This local CPU version is perfect for:
- ✅ Development and testing
- ✅ Small-scale processing
- ✅ Understanding the workflow

For production deployment on EC2 with GPU:
- Use G4 or G5 instances for better performance
- Implement model quantization for memory efficiency
- Add batch processing capabilities
- Set up proper logging and monitoring

## License

This project is for educational and development purposes.
