"""Document processor module.

This module provides a unified interface for processing PDF documents,
combining text extraction and chunking functionality.
"""

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid

from .pdf_extractor import extract_text_from_pdf, Page
from .text_chunker import chunk_pages, Chunk


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with all chunks and metadata."""
    document_id: str
    filename: str
    chunks: List[Chunk]
    page_count: int
    total_chunks: int
    status: str
    error_message: Optional[str] = None


class DocumentProcessor:
    """Processes PDF documents by extracting text and creating chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks in tokens (default: 1000)
            chunk_overlap: Overlap between consecutive chunks in tokens (default: 100)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessedDocument:
        """Process a PDF document: extract text and create chunks.
        
        Args:
            file_path: Path to the PDF file
            document_id: Optional document identifier (generated if not provided)
            
        Returns:
            ProcessedDocument with all chunks and metadata
            
        Raises:
            ValueError: For invalid PDFs, unsupported formats, or processing errors
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        filename = path.name
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        try:
            # Extract text from PDF
            pages = self._extract_text(file_path)
            
            if not pages:
                return ProcessedDocument(
                    document_id=document_id,
                    filename=filename,
                    chunks=[],
                    page_count=0,
                    total_chunks=0,
                    status='error',
                    error_message='No text could be extracted from the PDF'
                )
            
            # Chunk the extracted text
            chunks = self._chunk_document(pages, document_id)
            
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                chunks=chunks,
                page_count=len(pages),
                total_chunks=len(chunks),
                status='ready',
                error_message=None
            )
        
        except FileNotFoundError as e:
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                chunks=[],
                page_count=0,
                total_chunks=0,
                status='error',
                error_message=f'File not found: {str(e)}'
            )
        
        except ValueError as e:
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                chunks=[],
                page_count=0,
                total_chunks=0,
                status='error',
                error_message=f'Invalid PDF: {str(e)}'
            )
        
        except Exception as e:
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                chunks=[],
                page_count=0,
                total_chunks=0,
                status='error',
                error_message=f'Error processing document: {str(e)}'
            )
    
    def _extract_text(self, file_path: str) -> List[Page]:
        """Extract text from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Page objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: For invalid or corrupted PDFs
        """
        return extract_text_from_pdf(file_path)
    
    def _chunk_document(self, pages: List[Page], document_id: str) -> List[Chunk]:
        """Chunk the extracted pages.
        
        Args:
            pages: List of Page objects
            document_id: Document identifier
            
        Returns:
            List of Chunk objects
        """
        return chunk_pages(
            pages=pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            document_id=document_id
        )
