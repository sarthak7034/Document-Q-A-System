"""
Pydantic models for API request and response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status: processing, ready, or error")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    chunk_count: Optional[int] = Field(None, description="Number of chunks created")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "research_paper.pdf",
                "status": "processing",
                "page_count": None,
                "chunk_count": None,
                "error_message": None
            }
        }
    )


class DocumentMetadataResponse(BaseModel):
    """Response model for document metadata."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    upload_date: str = Field(..., description="Upload timestamp (ISO format)")
    page_count: Optional[int] = Field(None, description="Number of pages")
    chunk_count: Optional[int] = Field(None, description="Number of chunks")
    status: str = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if any")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "research_paper.pdf",
                "upload_date": "2024-01-15T10:30:00",
                "page_count": 25,
                "chunk_count": 48,
                "status": "ready",
                "error_message": None
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentMetadataResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="The question to ask about the documents", min_length=1)
    document_ids: Optional[List[str]] = Field(
        None, 
        description="Optional list of document IDs to search within (searches all if not provided)"
    )
    max_chunks: int = Field(
        5, 
        description="Maximum number of document chunks to retrieve",
        ge=1,
        le=20
    )
    temperature: float = Field(
        0.7,
        description="LLM temperature parameter (0.0 = deterministic, 2.0 = very creative)",
        ge=0.0,
        le=2.0
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the main findings of the research?",
                "document_ids": None,
                "max_chunks": 5,
                "temperature": 0.7
            }
        }
    )


class Source(BaseModel):
    """Source citation for an answer."""
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename")
    page_number: int = Field(..., description="Page number in the document")
    chunk_text: str = Field(..., description="Relevant text chunk")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[Source] = Field(..., description="Source citations used to generate the answer")
    model_used: str = Field(..., description="LLM model used for generation")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "The main findings of the research indicate that...",
                "sources": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_name": "research_paper.pdf",
                        "page_number": 5,
                        "chunk_text": "Our study demonstrates that...",
                        "similarity_score": 0.89
                    }
                ],
                "model_used": "llama2",
                "chunks_retrieved": 5
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    services: dict = Field(..., description="Status of individual services")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
