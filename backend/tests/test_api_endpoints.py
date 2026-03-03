"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import io

from app.main import app
from app.file_storage import DocumentMetadata, FileSizeError


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock all services."""
    with patch('app.dependencies.get_services') as mock:
        services = {
            'file_storage': Mock(),
            'document_processor': Mock(),
            'embedding_service': Mock(),
            'vector_store': Mock(),
            'llm_service': Mock(),
            'rag_engine': Mock()
        }
        mock.return_value = services
        yield services


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client, mock_services):
    """Test the health check endpoint."""
    # Mock LLM service health check
    mock_services['llm_service'].check_health = AsyncMock(return_value=True)
    mock_services['vector_store'].get_document_count.return_value = 5
    
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_upload_document_success(client, mock_services):
    """Test successful document upload."""
    # Mock file storage
    mock_metadata = DocumentMetadata(
        id="test-doc-id",
        filename="test.pdf",
        file_path="/path/to/test.pdf",
        upload_date="2024-01-01T00:00:00",
        status="processing"
    )
    mock_services['file_storage'].save_file.return_value = mock_metadata
    
    # Create a fake PDF file
    file_content = b"%PDF-1.4\nFake PDF content"
    files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
    
    response = client.post("/api/documents", files=files)
    assert response.status_code == 202
    data = response.json()
    assert data["document_id"] == "test-doc-id"
    assert data["filename"] == "test.pdf"
    assert data["status"] == "processing"


def test_upload_document_invalid_type(client, mock_services):
    """Test upload with invalid file type."""
    file_content = b"Not a PDF"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/api/documents", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "invalid_file_type" in str(data["detail"])


def test_upload_document_too_large(client, mock_services):
    """Test upload with file too large."""
    # Mock file storage to raise FileSizeError
    mock_services['file_storage'].save_file.side_effect = FileSizeError(
        "File size exceeds maximum allowed size"
    )
    
    file_content = b"%PDF-1.4\nFake PDF content"
    files = {"file": ("large.pdf", io.BytesIO(file_content), "application/pdf")}
    
    response = client.post("/api/documents", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_list_documents(client, mock_services):
    """Test listing documents."""
    # Mock file storage
    mock_docs = [
        DocumentMetadata(
            id="doc1",
            filename="test1.pdf",
            file_path="/path/to/test1.pdf",
            upload_date="2024-01-01T00:00:00",
            page_count=10,
            chunk_count=20,
            status="ready"
        ),
        DocumentMetadata(
            id="doc2",
            filename="test2.pdf",
            file_path="/path/to/test2.pdf",
            upload_date="2024-01-02T00:00:00",
            page_count=5,
            chunk_count=10,
            status="processing"
        )
    ]
    mock_services['file_storage'].list_documents.return_value = mock_docs
    
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert data["total"] == 2
    assert len(data["documents"]) == 2


def test_get_document_success(client, mock_services):
    """Test getting document details."""
    # Mock file storage
    mock_metadata = DocumentMetadata(
        id="test-doc-id",
        filename="test.pdf",
        file_path="/path/to/test.pdf",
        upload_date="2024-01-01T00:00:00",
        page_count=10,
        chunk_count=20,
        status="ready"
    )
    mock_services['file_storage'].get_metadata.return_value = mock_metadata
    
    response = client.get("/api/documents/test-doc-id")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-doc-id"
    assert data["filename"] == "test.pdf"
    assert data["status"] == "ready"


def test_get_document_not_found(client, mock_services):
    """Test getting non-existent document."""
    mock_services['file_storage'].get_metadata.return_value = None
    
    response = client.get("/api/documents/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_delete_document_success(client, mock_services):
    """Test deleting a document."""
    # Mock file storage
    mock_metadata = DocumentMetadata(
        id="test-doc-id",
        filename="test.pdf",
        file_path="/path/to/test.pdf",
        upload_date="2024-01-01T00:00:00",
        status="ready"
    )
    mock_services['file_storage'].get_metadata.return_value = mock_metadata
    mock_services['file_storage'].delete_document.return_value = True
    
    response = client.delete("/api/documents/test-doc-id")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "document_id" in data


def test_delete_document_not_found(client, mock_services):
    """Test deleting non-existent document."""
    mock_services['file_storage'].get_metadata.return_value = None
    
    response = client.delete("/api/documents/nonexistent")
    assert response.status_code == 404


def test_ask_question_success(client, mock_services):
    """Test asking a question."""
    from app.vector_store import Chunk, SearchResult
    
    # Mock RAG engine
    mock_services['rag_engine'].answer_question = AsyncMock(
        return_value="This is the answer to your question."
    )
    
    # Mock retrieve_context
    mock_chunk = Chunk(
        chunk_id="chunk1",
        text="This is relevant text from the document.",
        page_number=5,
        chunk_index=0,
        document_id="doc1",
        metadata={}
    )
    mock_result = SearchResult(chunk=mock_chunk, similarity_score=0.85)
    mock_services['rag_engine'].retrieve_context.return_value = [mock_result]
    
    # Mock file storage
    mock_metadata = DocumentMetadata(
        id="doc1",
        filename="test.pdf",
        file_path="/path/to/test.pdf",
        upload_date="2024-01-01T00:00:00",
        status="ready"
    )
    mock_services['file_storage'].get_metadata.return_value = mock_metadata
    
    request_data = {
        "question": "What is the main topic?",
        "max_chunks": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/questions", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "model_used" in data
    assert len(data["sources"]) == 1


def test_ask_question_invalid_document_id(client, mock_services):
    """Test asking question with invalid document ID."""
    mock_services['file_storage'].get_metadata.return_value = None
    
    request_data = {
        "question": "What is the main topic?",
        "document_ids": ["nonexistent"]
    }
    
    response = client.post("/api/questions", json=request_data)
    assert response.status_code == 400


def test_ask_question_document_not_ready(client, mock_services):
    """Test asking question when document is not ready."""
    mock_metadata = DocumentMetadata(
        id="doc1",
        filename="test.pdf",
        file_path="/path/to/test.pdf",
        upload_date="2024-01-01T00:00:00",
        status="processing"
    )
    mock_services['file_storage'].get_metadata.return_value = mock_metadata
    
    request_data = {
        "question": "What is the main topic?",
        "document_ids": ["doc1"]
    }
    
    response = client.post("/api/questions", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "document_not_ready" in str(data["detail"])


def test_ask_question_empty(client, mock_services):
    """Test asking an empty question."""
    request_data = {
        "question": ""
    }
    
    response = client.post("/api/questions", json=request_data)
    assert response.status_code == 422  # Validation error
