"""
PDF Processing Module
Handles PDF text extraction and intelligent chunking for MCQ generation
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict
import tiktoken
from logger_config import get_logger

logger = get_logger("pdf_processor")


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize PDF processor with chunking parameters
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        logger.info(f"PDFProcessor initialized - chunk_size: {chunk_size}, overlap: {overlap}")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text
                text += "\n\n"  # Add spacing between pages
                logger.debug(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
                
            doc.close()
            logger.info(f"Text extraction completed - Total characters: {len(text)}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """
        Create intelligent chunks from text
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'start_sentence': len(chunks) * (self.chunk_size - self.overlap) if chunks else 0
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    # Get last few sentences for overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = len(self.tokenizer.encode(current_chunk))
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'start_sentence': len(chunks) * (self.chunk_size - self.overlap) if chunks else 0
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last portion of text that contains approximately overlap_tokens
        
        Args:
            text: Input text
            overlap_tokens: Number of tokens to include in overlap
            
        Returns:
            Overlap text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        # Get the last overlap_tokens
        overlap_tokens_list = tokens[-overlap_tokens:]
        return self.tokenizer.decode(overlap_tokens_list)
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed chunks
        """
        logger.info(f"Starting PDF processing: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        
        logger.info(f"Text extracted: {len(text)} characters")
        logger.info(f"Creating chunks with size {self.chunk_size} tokens...")
        
        chunks = self.create_chunks(text)
        
        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {chunk['token_count']} tokens")
        
        return chunks


if __name__ == "__main__":
    # Test the PDF processor
    processor = PDFProcessor(chunk_size=1000, overlap=100)
    
    # You can test with a sample PDF
    # chunks = processor.process_pdf("sample.pdf")
    # print(f"Processed {len(chunks)} chunks")
