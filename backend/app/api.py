"""
API routes for the Document Q&A System.

This module defines all REST API endpoints for document management
and question answering.
"""

import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    DocumentUploadResponse,
    DocumentMetadataResponse,
    DocumentListResponse,
    QuestionRequest,
    QuestionResponse,
    Source,
    HealthResponse,
    ErrorResponse
)
from .dependencies import (
    get_file_storage,
    get_document_processor,
    get_embedding_service,
    get_vector_store,
    get_rag_engine,
    get_llm_service
)
from .file_storage import FileStorage, FileSizeError
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import ChromaVectorStore
from .rag_engine import RAGEngine
from .llm_service import LLMService
from .config import Config


logger = logging.getLogger(__name__)
router = APIRouter()


async def process_document_background(
    document_id: str,
    file_path: str,
    file_storage: FileStorage,
    document_processor: DocumentProcessor,
    embedding_service: EmbeddingService,
    vector_store: ChromaVectorStore
):
    """
    Background task to process uploaded document.
    
    This function:
    1. Processes the document (extract text and chunk)
    2. Generates embeddings for chunks
    3. Stores embeddings in vector store
    4. Updates document metadata
    """
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Process document
        processed_doc = document_processor.process_document(file_path, document_id)
        
        if processed_doc.status == 'error':
            logger.error(f"Document processing failed: {processed_doc.error_message}")
            file_storage.update_metadata(
                document_id=document_id,
                status='error',
                error_message=processed_doc.error_message
            )
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(processed_doc.chunks)} chunks")
        chunk_texts = [chunk.text for chunk in processed_doc.chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        # Store in vector store
        logger.info(f"Storing embeddings in vector store")
        vector_store.add_documents(processed_doc.chunks, embeddings)
        
        # Update metadata
        file_storage.update_metadata(
            document_id=document_id,
            page_count=processed_doc.page_count,
            chunk_count=processed_doc.total_chunks,
            status='ready'
        )
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        file_storage.update_metadata(
            document_id=document_id,
            status='error',
            error_message=str(e)
        )


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=202,
    tags=["Documents"],
    summary="Upload a PDF document",
    description="""
    Upload a PDF document for processing. The document will be:
    1. Validated for size (max 50MB) and type (PDF only)
    2. Saved to disk
    3. Processed in the background (text extraction, chunking, embedding generation)
    
    Returns immediately with document ID and 'processing' status.
    Use GET /api/documents/{id} to check processing status.
    """,
    responses={
        202: {"description": "Document accepted for processing"},
        400: {"model": ErrorResponse, "description": "Invalid file (wrong type, too large, etc.)"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload (max 50MB)"),
    file_storage: FileStorage = Depends(get_file_storage),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Upload and process a PDF document."""
    logger.info(f"Received upload request for file: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "message": "Only PDF files are supported",
                "details": {"filename": file.filename}
            }
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file and create metadata
        metadata = file_storage.save_file(
            file_content=file_content,
            filename=file.filename
        )
        
        logger.info(f"File saved with document ID: {metadata.id}")
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            document_id=metadata.id,
            file_path=metadata.file_path,
            file_storage=file_storage,
            document_processor=document_processor,
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        return DocumentUploadResponse(
            document_id=metadata.id,
            filename=metadata.filename,
            status=metadata.status,
            page_count=metadata.page_count,
            chunk_count=metadata.chunk_count,
            error_message=metadata.error_message
        )
        
    except FileSizeError as e:
        logger.warning(f"File size error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "file_too_large",
                "message": str(e),
                "details": {"max_size_mb": Config.MAX_FILE_SIZE_MB}
            }
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_error",
                "message": f"Failed to upload document: {str(e)}"
            }
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all documents",
    description="Retrieve a list of all uploaded documents with their metadata."
)
async def list_documents(
    file_storage: FileStorage = Depends(get_file_storage)
):
    """List all uploaded documents."""
    logger.info("Listing all documents")
    
    try:
        documents = file_storage.list_documents()
        
        return DocumentListResponse(
            documents=[
                DocumentMetadataResponse(
                    id=doc.id,
                    filename=doc.filename,
                    upload_date=doc.upload_date,
                    page_count=doc.page_count,
                    chunk_count=doc.chunk_count,
                    status=doc.status,
                    error_message=doc.error_message
                )
                for doc in documents
            ],
            total=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "list_error",
                "message": f"Failed to list documents: {str(e)}"
            }
        )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentMetadataResponse,
    tags=["Documents"],
    summary="Get document details",
    description="Retrieve detailed metadata for a specific document.",
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    }
)
async def get_document(
    document_id: str,
    file_storage: FileStorage = Depends(get_file_storage)
):
    """Get details for a specific document."""
    logger.info(f"Getting document details for: {document_id}")
    
    try:
        metadata = file_storage.get_metadata(document_id)
        
        if not metadata:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "document_not_found",
                    "message": f"Document with ID '{document_id}' not found"
                }
            )
        
        return DocumentMetadataResponse(
            id=metadata.id,
            filename=metadata.filename,
            upload_date=metadata.upload_date,
            page_count=metadata.page_count,
            chunk_count=metadata.chunk_count,
            status=metadata.status,
            error_message=metadata.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "get_error",
                "message": f"Failed to get document: {str(e)}"
            }
        )


@router.delete(
    "/documents/{document_id}",
    status_code=200,
    tags=["Documents"],
    summary="Delete a document",
    description="""
    Delete a document and all associated data:
    - Document file from disk
    - Document metadata from database
    - All chunks and embeddings from vector store
    """,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    }
)
async def delete_document(
    document_id: str,
    file_storage: FileStorage = Depends(get_file_storage),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Delete a document and all associated data."""
    logger.info(f"Deleting document: {document_id}")
    
    try:
        # Check if document exists
        metadata = file_storage.get_metadata(document_id)
        if not metadata:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "document_not_found",
                    "message": f"Document with ID '{document_id}' not found"
                }
            )
        
        # Delete from vector store
        logger.info(f"Deleting embeddings for document: {document_id}")
        vector_store.delete_document(document_id)
        
        # Delete file and metadata
        logger.info(f"Deleting file and metadata for document: {document_id}")
        file_storage.delete_document(document_id)
        
        logger.info(f"Document {document_id} deleted successfully")
        
        return {
            "message": f"Document '{metadata.filename}' deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "delete_error",
                "message": f"Failed to delete document: {str(e)}"
            }
        )


@router.post(
    "/questions",
    response_model=QuestionResponse,
    tags=["Questions"],
    summary="Ask a question about documents",
    description="""
    Ask a natural language question about your uploaded documents.
    
    The system will:
    1. Generate an embedding for your question
    2. Search for relevant document chunks
    3. Construct a prompt with the retrieved context
    4. Generate an answer using the local LLM
    
    You can optionally filter to specific documents using document_ids.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"}
    }
)
async def ask_question(
    request: QuestionRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    file_storage: FileStorage = Depends(get_file_storage)
):
    """Ask a question about uploaded documents."""
    logger.info(f"Received question: {request.question[:100]}...")
    
    try:
        # Validate document_ids if provided
        if request.document_ids:
            for doc_id in request.document_ids:
                metadata = file_storage.get_metadata(doc_id)
                if not metadata:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "invalid_document_id",
                            "message": f"Document with ID '{doc_id}' not found"
                        }
                    )
                if metadata.status != 'ready':
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "document_not_ready",
                            "message": f"Document '{metadata.filename}' is not ready (status: {metadata.status})"
                        }
                    )
        
        # Get answer from RAG engine
        answer = await rag_engine.answer_question(
            question=request.question,
            document_ids=request.document_ids,
            max_chunks=request.max_chunks,
            temperature=request.temperature
        )
        
        # Get sources
        sources_results = rag_engine.retrieve_context(
            question=request.question,
            document_ids=request.document_ids,
            max_chunks=request.max_chunks
        )
        
        # Build sources list
        sources = []
        for result in sources_results:
            # Get document metadata for filename
            metadata = file_storage.get_metadata(result.chunk.document_id)
            document_name = metadata.filename if metadata else "Unknown"
            
            sources.append(Source(
                document_id=result.chunk.document_id,
                document_name=document_name,
                page_number=result.chunk.page_number,
                chunk_text=result.chunk.text[:200] + "..." if len(result.chunk.text) > 200 else result.chunk.text,
                similarity_score=result.similarity_score
            ))
        
        logger.info(f"Question answered successfully with {len(sources)} sources")
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            model_used=Config.OLLAMA_MODEL,
            chunks_retrieved=len(sources)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        
        # Check if it's an LLM service error
        if "Ollama" in str(e) or "connection" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "llm_service_unavailable",
                    "message": f"LLM service is not accessible: {str(e)}",
                    "details": {
                        "service_url": Config.OLLAMA_BASE_URL,
                        "suggestion": "Ensure Ollama is running: docker-compose up ollama"
                    }
                }
            )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "question_error",
                "message": f"Failed to answer question: {str(e)}"
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="System health check",
    description="Check the health status of all system components."
)
async def health_check(
    llm_service: LLMService = Depends(get_llm_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store)
):
    """Check system health."""
    logger.info("Health check requested")
    
    try:
        # Check LLM service
        llm_healthy = await llm_service.check_health()
        
        # Check vector store (try to get count)
        try:
            doc_count = vector_store.get_document_count()
            vector_store_healthy = True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            doc_count = None
            vector_store_healthy = False
        
        # Determine overall status
        overall_status = "healthy" if (llm_healthy and vector_store_healthy) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services={
                "llm": {
                    "status": "healthy" if llm_healthy else "unhealthy",
                    "url": Config.OLLAMA_BASE_URL,
                    "model": Config.OLLAMA_MODEL
                },
                "vector_store": {
                    "status": "healthy" if vector_store_healthy else "unhealthy",
                    "type": Config.VECTOR_STORE_TYPE,
                    "document_count": doc_count
                },
                "embedding_service": {
                    "status": "healthy",
                    "model": Config.EMBEDDING_MODEL
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            services={
                "error": str(e)
            }
        )
