"""PDF text extraction module.

This module provides functionality to extract text from PDF files,
preserving page numbers and structure.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import PyPDF2
from pathlib import Path


@dataclass
class Page:
    """Represents a single page from a PDF document."""
    page_number: int
    text: str
    char_count: int


def extract_text_from_pdf(file_path: str) -> List[Page]:
    """Extract text from all pages of a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Page objects containing text and metadata for each page
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PyPDF2.errors.PdfReadError: If the PDF is corrupted or invalid
        Exception: For other PDF processing errors
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")
    
    pages = []
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                raise ValueError("PDF is password-protected and cannot be processed")
            
            num_pages = len(pdf_reader.pages)
            
            if num_pages == 0:
                raise ValueError("PDF contains no pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Create Page object with 1-based page numbering
                pages.append(Page(
                    page_number=page_num + 1,
                    text=text,
                    char_count=len(text)
                ))
    
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    
    return pages
