"""
Integration tests for file storage module demonstrating complete workflows.
"""

import os
import tempfile
import shutil
import pytest

from app.file_storage import FileStorage, FileSizeError


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


def test_complete_document_lifecycle(file_storage):
    """
    Test complete document lifecycle: upload -> update -> retrieve -> delete.
    
    This test validates Requirements 1.1 and 11.1:
    - System accepts files up to 50MB
    - System assigns unique identifier to each document
    """
    # 1. Upload document
    content = b"Sample PDF content for testing"
    filename = "sample_document.pdf"
    
    metadata = file_storage.save_file(content, filename)
    
    # Verify unique ID was assigned (Requirement 11.1)
    assert metadata.id is not None
    assert len(metadata.id) > 0
    
    # Verify file was saved
    assert os.path.exists(metadata.file_path)
    assert metadata.filename == filename
    assert metadata.status == 'processing'
    
    # 2. Update metadata after processing
    file_storage.update_metadata(
        metadata.id,
        page_count=5,
        chunk_count=25,
        status='ready'
    )
    
    # 3. Retrieve and verify updated metadata
    updated = file_storage.get_metadata(metadata.id)
    assert updated.page_count == 5
    assert updated.chunk_count == 25
    assert updated.status == 'ready'
    
    # 4. List documents
    documents = file_storage.list_documents()
    assert len(documents) == 1
    assert documents[0].id == metadata.id
    
    # 5. Delete document
    result = file_storage.delete_document(metadata.id)
    assert result is True
    
    # Verify deletion
    assert not os.path.exists(metadata.file_path)
    assert file_storage.get_metadata(metadata.id) is None


def test_file_size_validation_requirement(file_storage):
    """
    Test file size validation (Requirement 1.1).
    
    System SHALL accept files up to 50MB in size.
    """
    # Test 1: Small file should be accepted
    small_content = b"Small file content"
    small_metadata = file_storage.save_file(small_content, "small.pdf")
    assert small_metadata is not None
    assert os.path.exists(small_metadata.file_path)
    
    # Test 2: Exactly 50MB should be accepted
    max_size_content = b"x" * (50 * 1024 * 1024)
    max_metadata = file_storage.save_file(max_size_content, "max_size.pdf")
    assert max_metadata is not None
    assert os.path.exists(max_metadata.file_path)
    
    # Test 3: Over 50MB should be rejected
    oversized_content = b"x" * (51 * 1024 * 1024)
    with pytest.raises(FileSizeError) as exc_info:
        file_storage.save_file(oversized_content, "oversized.pdf")
    
    assert "exceeds maximum allowed size" in str(exc_info.value)
    assert "50MB" in str(exc_info.value)


def test_unique_document_id_requirement(file_storage):
    """
    Test unique document ID generation (Requirement 11.1).
    
    System SHALL assign a unique identifier to each uploaded document.
    """
    # Upload multiple documents
    doc_ids = []
    for i in range(10):
        content = f"Document {i} content".encode()
        filename = f"document_{i}.pdf"
        metadata = file_storage.save_file(content, filename)
        doc_ids.append(metadata.id)
    
    # Verify all IDs are unique
    assert len(doc_ids) == len(set(doc_ids)), "Document IDs are not unique"
    
    # Verify all IDs are non-empty strings
    for doc_id in doc_ids:
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0


def test_multiple_documents_management(file_storage):
    """
    Test managing multiple documents simultaneously.
    """
    # Upload multiple documents
    documents = []
    for i in range(5):
        content = f"Content for document {i}".encode()
        filename = f"doc_{i}.pdf"
        metadata = file_storage.save_file(content, filename)
        documents.append(metadata)
    
    # Verify all documents are listed
    all_docs = file_storage.list_documents()
    assert len(all_docs) == 5
    
    # Update some documents to 'ready' status
    for i in [0, 2, 4]:
        file_storage.update_metadata(documents[i].id, status='ready')
    
    # Verify updates
    ready_count = 0
    processing_count = 0
    for doc in file_storage.list_documents():
        if doc.status == 'ready':
            ready_count += 1
        elif doc.status == 'processing':
            processing_count += 1
    
    assert ready_count == 3
    assert processing_count == 2
    
    # Delete one document
    file_storage.delete_document(documents[0].id)
    
    # Verify deletion
    remaining_docs = file_storage.list_documents()
    assert len(remaining_docs) == 4
    assert documents[0].id not in [doc.id for doc in remaining_docs]


def test_error_handling_workflow(file_storage):
    """
    Test error handling during document processing.
    """
    # Upload document
    content = b"Test content"
    filename = "test.pdf"
    metadata = file_storage.save_file(content, filename)
    
    # Simulate processing error
    error_message = "Failed to extract text from PDF"
    file_storage.update_metadata(
        metadata.id,
        status='error',
        error_message=error_message
    )
    
    # Retrieve and verify error state
    updated = file_storage.get_metadata(metadata.id)
    assert updated.status == 'error'
    assert updated.error_message == error_message
    
    # Document should still be in database
    all_docs = file_storage.list_documents()
    assert len(all_docs) == 1
    assert all_docs[0].status == 'error'


def test_database_persistence(temp_storage_dir):
    """
    Test that database persists across FileStorage instances.
    """
    storage_dir = os.path.join(temp_storage_dir, "documents")
    db_path = os.path.join(temp_storage_dir, "metadata.db")
    
    # Create first instance and upload document
    storage1 = FileStorage(storage_dir=storage_dir, db_path=db_path)
    content = b"Persistent content"
    filename = "persistent.pdf"
    metadata = storage1.save_file(content, filename)
    doc_id = metadata.id
    
    # Create second instance (simulating restart)
    storage2 = FileStorage(storage_dir=storage_dir, db_path=db_path)
    
    # Verify document is still accessible
    retrieved = storage2.get_metadata(doc_id)
    assert retrieved is not None
    assert retrieved.id == doc_id
    assert retrieved.filename == filename
    
    # Verify file still exists
    assert os.path.exists(retrieved.file_path)
    with open(retrieved.file_path, 'rb') as f:
        assert f.read() == content
