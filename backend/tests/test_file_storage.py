"""
Unit tests for file storage module.
"""

import os
import sqlite3
import tempfile
import shutil
from pathlib import Path
import pytest

from app.file_storage import (
    FileStorage, 
    DocumentMetadata, 
    FileSizeError,
    MAX_FILE_SIZE_BYTES
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def file_storage(temp_storage_dir):
    """Create a FileStorage instance with temporary directories."""
    storage_dir = os.path.join(temp_storage_dir, "documents")
    db_path = os.path.join(temp_storage_dir, "metadata.db")
    return FileStorage(storage_dir=storage_dir, db_path=db_path)


class TestFileStorage:
    """Test suite for FileStorage class."""

    def test_init_creates_directories(self, temp_storage_dir):
        """Test that initialization creates necessary directories."""
        storage_dir = os.path.join(temp_storage_dir, "documents")
        db_path = os.path.join(temp_storage_dir, "metadata.db")
        
        storage = FileStorage(storage_dir=storage_dir, db_path=db_path)
        
        assert os.path.exists(storage_dir)
        assert os.path.exists(db_path)

    def test_init_creates_database_table(self, file_storage):
        """Test that initialization creates the documents table."""
        conn = sqlite3.connect(file_storage.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='documents'
        """)
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == 'documents'

    def test_validate_file_size_accepts_valid_size(self, file_storage):
        """Test that valid file sizes are accepted."""
        # 1MB file should be accepted
        file_size = 1 * 1024 * 1024
        file_storage.validate_file_size(file_size)  # Should not raise

    def test_validate_file_size_rejects_oversized_file(self, file_storage):
        """Test that oversized files are rejected."""
        # 51MB file should be rejected
        file_size = 51 * 1024 * 1024
        
        with pytest.raises(FileSizeError) as exc_info:
            file_storage.validate_file_size(file_size)
        
        assert "exceeds maximum allowed size" in str(exc_info.value)

    def test_validate_file_size_accepts_max_size(self, file_storage):
        """Test that exactly 50MB is accepted."""
        file_size = MAX_FILE_SIZE_BYTES
        file_storage.validate_file_size(file_size)  # Should not raise

    def test_generate_document_id_returns_unique_ids(self, file_storage):
        """Test that generated document IDs are unique."""
        id1 = file_storage.generate_document_id()
        id2 = file_storage.generate_document_id()
        
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    def test_generate_document_id_returns_valid_uuid(self, file_storage):
        """Test that generated IDs are valid UUIDs."""
        import uuid
        
        doc_id = file_storage.generate_document_id()
        
        # Should be able to parse as UUID
        try:
            uuid.UUID(doc_id)
        except ValueError:
            pytest.fail("Generated ID is not a valid UUID")

    def test_save_file_creates_file_on_disk(self, file_storage):
        """Test that save_file creates a file on disk."""
        content = b"Test PDF content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        # Check file exists
        assert os.path.exists(metadata.file_path)
        
        # Check file content
        with open(metadata.file_path, 'rb') as f:
            saved_content = f.read()
        assert saved_content == content

    def test_save_file_creates_metadata_entry(self, file_storage):
        """Test that save_file creates a metadata entry in database."""
        content = b"Test PDF content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        # Retrieve metadata from database
        retrieved = file_storage.get_metadata(metadata.id)
        
        assert retrieved is not None
        assert retrieved.id == metadata.id
        assert retrieved.filename == filename
        assert retrieved.status == 'processing'

    def test_save_file_rejects_oversized_file(self, file_storage):
        """Test that save_file rejects files exceeding size limit."""
        # Create content larger than 50MB
        content = b"x" * (51 * 1024 * 1024)
        filename = "large.pdf"
        
        with pytest.raises(FileSizeError):
            file_storage.save_file(content, filename)

    def test_save_file_with_custom_document_id(self, file_storage):
        """Test that save_file accepts a custom document ID."""
        content = b"Test content"
        filename = "test.pdf"
        custom_id = "custom-doc-id-123"
        
        metadata = file_storage.save_file(content, filename, document_id=custom_id)
        
        assert metadata.id == custom_id

    def test_save_file_preserves_file_extension(self, file_storage):
        """Test that file extension is preserved in stored filename."""
        content = b"Test content"
        filename = "document.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        assert metadata.file_path.endswith('.pdf')

    def test_update_metadata_updates_page_count(self, file_storage):
        """Test updating page count in metadata."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        file_storage.update_metadata(metadata.id, page_count=10)
        
        updated = file_storage.get_metadata(metadata.id)
        assert updated.page_count == 10

    def test_update_metadata_updates_status(self, file_storage):
        """Test updating status in metadata."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        file_storage.update_metadata(metadata.id, status='ready')
        
        updated = file_storage.get_metadata(metadata.id)
        assert updated.status == 'ready'

    def test_update_metadata_updates_error_message(self, file_storage):
        """Test updating error message in metadata."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        error_msg = "Failed to process document"
        file_storage.update_metadata(
            metadata.id, 
            status='error', 
            error_message=error_msg
        )
        
        updated = file_storage.get_metadata(metadata.id)
        assert updated.status == 'error'
        assert updated.error_message == error_msg

    def test_update_metadata_updates_multiple_fields(self, file_storage):
        """Test updating multiple fields at once."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        file_storage.update_metadata(
            metadata.id,
            page_count=5,
            chunk_count=20,
            status='ready'
        )
        
        updated = file_storage.get_metadata(metadata.id)
        assert updated.page_count == 5
        assert updated.chunk_count == 20
        assert updated.status == 'ready'

    def test_get_metadata_returns_none_for_nonexistent_id(self, file_storage):
        """Test that get_metadata returns None for non-existent document."""
        result = file_storage.get_metadata("nonexistent-id")
        assert result is None

    def test_list_documents_returns_all_documents(self, file_storage):
        """Test that list_documents returns all uploaded documents."""
        # Upload multiple documents
        file_storage.save_file(b"Content 1", "doc1.pdf")
        file_storage.save_file(b"Content 2", "doc2.pdf")
        file_storage.save_file(b"Content 3", "doc3.pdf")
        
        documents = file_storage.list_documents()
        
        assert len(documents) == 3
        filenames = [doc.filename for doc in documents]
        assert "doc1.pdf" in filenames
        assert "doc2.pdf" in filenames
        assert "doc3.pdf" in filenames

    def test_list_documents_returns_empty_list_when_no_documents(self, file_storage):
        """Test that list_documents returns empty list when no documents exist."""
        documents = file_storage.list_documents()
        assert documents == []

    def test_list_documents_orders_by_upload_date_desc(self, file_storage):
        """Test that documents are ordered by upload date (newest first)."""
        import time
        
        # Upload documents with slight delay
        meta1 = file_storage.save_file(b"Content 1", "doc1.pdf")
        time.sleep(0.01)
        meta2 = file_storage.save_file(b"Content 2", "doc2.pdf")
        time.sleep(0.01)
        meta3 = file_storage.save_file(b"Content 3", "doc3.pdf")
        
        documents = file_storage.list_documents()
        
        # Newest should be first
        assert documents[0].id == meta3.id
        assert documents[1].id == meta2.id
        assert documents[2].id == meta1.id

    def test_delete_document_removes_file(self, file_storage):
        """Test that delete_document removes the file from disk."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        file_path = metadata.file_path
        
        # Verify file exists
        assert os.path.exists(file_path)
        
        # Delete document
        result = file_storage.delete_document(metadata.id)
        
        assert result is True
        assert not os.path.exists(file_path)

    def test_delete_document_removes_metadata(self, file_storage):
        """Test that delete_document removes metadata from database."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        # Delete document
        file_storage.delete_document(metadata.id)
        
        # Verify metadata is gone
        retrieved = file_storage.get_metadata(metadata.id)
        assert retrieved is None

    def test_delete_document_returns_false_for_nonexistent_id(self, file_storage):
        """Test that delete_document returns False for non-existent document."""
        result = file_storage.delete_document("nonexistent-id")
        assert result is False

    def test_delete_document_handles_missing_file(self, file_storage):
        """Test that delete_document handles case where file is already missing."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        # Manually delete the file
        os.remove(metadata.file_path)
        
        # Should still delete metadata without error
        result = file_storage.delete_document(metadata.id)
        assert result is True

    def test_get_file_path_returns_correct_path(self, file_storage):
        """Test that get_file_path returns the correct file path."""
        content = b"Test content"
        filename = "test.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        file_path = file_storage.get_file_path(metadata.id)
        assert file_path == metadata.file_path

    def test_get_file_path_returns_none_for_nonexistent_id(self, file_storage):
        """Test that get_file_path returns None for non-existent document."""
        file_path = file_storage.get_file_path("nonexistent-id")
        assert file_path is None

    def test_document_metadata_to_dict(self):
        """Test DocumentMetadata to_dict conversion."""
        metadata = DocumentMetadata(
            id="test-id",
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            upload_date="2024-01-01T00:00:00",
            page_count=10,
            chunk_count=50,
            status="ready"
        )
        
        result = metadata.to_dict()
        
        assert result['id'] == "test-id"
        assert result['filename'] == "test.pdf"
        assert result['page_count'] == 10
        assert result['status'] == "ready"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file(self, file_storage):
        """Test handling of empty file."""
        content = b""
        filename = "empty.pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        assert os.path.exists(metadata.file_path)
        with open(metadata.file_path, 'rb') as f:
            assert f.read() == b""

    def test_filename_with_special_characters(self, file_storage):
        """Test handling of filenames with special characters."""
        content = b"Test content"
        filename = "test file (1) [copy].pdf"
        
        metadata = file_storage.save_file(content, filename)
        
        assert metadata.filename == filename
        assert os.path.exists(metadata.file_path)

    def test_filename_without_extension(self, file_storage):
        """Test handling of filename without extension."""
        content = b"Test content"
        filename = "document"
        
        metadata = file_storage.save_file(content, filename)
        
        assert metadata.filename == filename
        assert os.path.exists(metadata.file_path)

    def test_multiple_files_same_name(self, file_storage):
        """Test that multiple files with same name don't conflict."""
        content1 = b"Content 1"
        content2 = b"Content 2"
        filename = "test.pdf"
        
        meta1 = file_storage.save_file(content1, filename)
        meta2 = file_storage.save_file(content2, filename)
        
        # Both files should exist with different paths
        assert os.path.exists(meta1.file_path)
        assert os.path.exists(meta2.file_path)
        assert meta1.file_path != meta2.file_path
        
        # Content should be preserved correctly
        with open(meta1.file_path, 'rb') as f:
            assert f.read() == content1
        with open(meta2.file_path, 'rb') as f:
            assert f.read() == content2
