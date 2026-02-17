"""
Unit tests for DocumentProcessor.
"""

import pytest
from pathlib import Path
from app.document_processor import DocumentProcessor, ProcessedDocument


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(chunk_size=1000, chunk_overlap=100)
    
    def test_processor_initialization(self, processor):
        """Test that processor initializes with correct parameters."""
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 100
    
    def test_process_nonexistent_file(self, processor):
        """Test processing a file that doesn't exist."""
        result = processor.process_document("nonexistent.pdf")
        
        assert result.status == 'error'
        assert 'File not found' in result.error_message
        assert result.page_count == 0
        assert result.total_chunks == 0
    
    def test_process_non_pdf_file(self, processor, tmp_path):
        """Test processing a non-PDF file."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not a PDF")
        
        result = processor.process_document(str(text_file))
        
        assert result.status == 'error'
        assert result.error_message is not None
