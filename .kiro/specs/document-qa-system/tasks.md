# Implementation Plan: Document Q&A System

## Overview

This implementation plan breaks down the Document Q&A System into incremental coding tasks. The approach follows a bottom-up strategy: build core components first (document processing, embeddings, vector store), then add the RAG engine, and finally the API layer with FastAPI and Swagger UI. Each task builds on previous work, with property-based tests integrated throughout to validate correctness early.

The implementation uses Python for the backend with FastAPI providing automatic Swagger UI for testing and interaction.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create backend directory structure (app/, tests/, data/)
  - Write requirements.txt with core dependencies (fastapi, uvicorn, sentence-transformers, chromadb, pypdf2, hypothesis, pytest, python-multipart)
  - Create .env.example with configuration templates
  - Set up pytest configuration and test directory structure
  - _Requirements: 10.1, 10.6_

- [ ] 2. Implement Document Processor
  - [ ] 2.1 Create PDF text extraction module
    - Write function to extract text from PDF using PyPDF2 or pdfplumber
    - Handle multi-page PDFs and preserve page numbers
    - Return structured data with page numbers and text content
    - _Requirements: 1.2_
  
  - [ ]* 2.2 Write property test for text extraction
    - **Property 2: Complete text extraction**
    - **Validates: Requirements 1.2**
  
  - [ ] 2.3 Implement text chunking with overlap
    - Write chunking function that splits text into 500-1000 token chunks
    - Implement 100-token overlap between consecutive chunks
    - Preserve sentence boundaries (don't split mid-sentence)
    - Store metadata (page number, chunk index) with each chunk
    - _Requirements: 1.3, 9.1, 9.2, 9.3_
  
  - [ ]* 2.4 Write property tests for chunking
    - **Property 3: Chunk size constraints**
    - **Property 4: Chunk overlap preservation**
    - **Property 5: Sentence boundary preservation**
    - **Property 21: Chunk metadata preservation**
    - **Validates: Requirements 1.3, 9.1, 9.2, 9.3**
  
  - [ ] 2.5 Create DocumentProcessor class
    - Combine extraction and chunking into unified interface
    - Add error handling for corrupted PDFs and unsupported formats
    - Return ProcessedDocument with all chunks and metadata
    - _Requirements: 1.2, 1.3, 1.7_
  
  - [ ]* 2.6 Write property test for error handling
    - **Property 14: Error message descriptiveness**
    - **Validates: Requirements 1.7**

- [ ] 3. Implement Embedding Service
  - [ ] 3.1 Create EmbeddingService class
    - Load sentence-transformers model (all-MiniLM-L6-v2)
    - Implement embed_text() for single text
    - Implement embed_batch() for multiple texts
    - Add get_embedding_dimension() method
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ]* 3.2 Write property test for embedding consistency
    - **Property 11: Embedding dimension consistency**
    - **Validates: Requirements 5.3**
  
  - [ ]* 3.3 Write property test for embedding generation
    - **Property 7: Question embedding generation**
    - **Validates: Requirements 2.1**

- [ ] 4. Implement Vector Store
  - [ ] 4.1 Create VectorStore abstract base class
    - Define interface: add_documents(), search(), delete_document(), get_document_count()
    - Define SearchResult data class
    - _Requirements: 3.1_
  
  - [ ] 4.2 Implement ChromaDB vector store
    - Create ChromaVectorStore class implementing VectorStore interface
    - Initialize persistent ChromaDB client
    - Implement add_documents() to store embeddings with metadata
    - Implement search() with cosine similarity
    - Implement delete_document() to remove all chunks for a document
    - _Requirements: 3.1, 3.4, 3.5_
  
  - [ ]* 4.3 Write property tests for vector store
    - **Property 6: Embedding generation round-trip**
    - **Property 8: Retrieval count accuracy**
    - **Property 12: Vector store persistence**
    - **Property 13: Complete document deletion**
    - **Validates: Requirements 1.4, 1.5, 2.2, 3.4, 3.5, 12.3**

- [ ] 5. Checkpoint - Core components complete
  - Ensure all tests pass for DocumentProcessor, EmbeddingService, and VectorStore
  - Verify that documents can be processed, embedded, and stored/retrieved
  - Ask the user if questions arise

- [ ] 6. Implement LLM Service
  - [ ] 6.1 Create LLMService class
    - Implement connection to Ollama HTTP API
    - Create generate() method supporting both streaming and non-streaming modes
    - Add check_health() to verify Ollama availability
    - Add list_models() to get available models
    - Handle connection errors with clear messages
    - _Requirements: 4.1, 4.3, 4.4, 4.5_
  
  - [ ] 6.2 Write unit tests for LLM service
    - Test health check with mocked Ollama
    - Test error handling when Ollama is unavailable
    - Test response parsing for both streaming and non-streaming modes
    - _Requirements: 4.4_

- [ ] 7. Implement RAG Engine
  - [ ] 7.1 Create RAGEngine class
    - Initialize with VectorStore, EmbeddingService, and LLMService
    - Implement retrieve_context() to get relevant chunks
    - Implement construct_prompt() with template
    - Implement answer_question() returning complete response
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 7.2 Write property tests for RAG engine
    - **Property 9: Prompt construction completeness**
    - **Property 10: Source citation inclusion**
    - **Validates: Requirements 2.3, 2.7**
  
  - [ ] 7.3 Add edge case handling
    - Handle case when no relevant chunks found (below similarity threshold)
    - Return appropriate message to user
    - _Requirements: 2.6_

- [ ] 8. Implement file upload and storage
  - [ ] 8.1 Create file storage module
    - Implement file size validation (max 50MB)
    - Generate unique document IDs
    - Save uploaded files to disk
    - Create SQLite database for document metadata
    - _Requirements: 1.1, 11.1_
  
  - [ ]* 8.2 Write property tests for file handling
    - **Property 1: File size validation**
    - **Property 17: Document ID uniqueness**
    - **Validates: Requirements 1.1, 11.1**

- [ ] 9. Implement FastAPI backend
  - [ ] 9.1 Create FastAPI application and configuration
    - Initialize FastAPI app with metadata for Swagger UI
    - Set up logging configuration
    - _Requirements: 6.5, 9.5_
  
  - [ ] 9.2 Implement document upload endpoint
    - Create POST /api/documents endpoint
    - Accept multipart file upload
    - Validate file type and size
    - Trigger document processing pipeline
    - Return document ID and status
    - _Requirements: 1.1, 6.1_
  
  - [ ] 9.3 Implement document management endpoints
    - Create GET /api/documents to list all documents
    - Create GET /api/documents/{id} for document details
    - Create DELETE /api/documents/{id} to remove documents
    - _Requirements: 6.2, 6.3, 11.2, 11.3_
  
  - [ ]* 9.4 Write property tests for document management
    - **Property 18: Document metadata completeness**
    - **Validates: Requirements 11.2**
  
  - [ ] 9.5 Implement question answering endpoint
    - Create POST /api/questions endpoint
    - Accept question text and optional document filters
    - Call RAG engine and return complete response with sources
    - _Requirements: 6.4_
  
  - [ ] 9.6 Add error handling middleware
    - Implement global exception handler
    - Return appropriate HTTP status codes
    - Format error responses as JSON
    - _Requirements: 6.7_
  
  - [ ]* 9.7 Write property tests for API error handling
    - **Property 15: API error response format**
    - **Validates: Requirements 6.7**
  
  - [ ] 9.8 Add request logging middleware
    - Log all API requests with timestamp, endpoint, method, status
    - Log errors with stack traces
    - _Requirements: 9.1, 9.2_
  
  - [ ]* 9.9 Write property test for logging
    - **Property 20: Request logging completeness**
    - **Validates: Requirements 9.1, 9.2**
  
  - [ ] 9.10 Configure Swagger UI
    - Customize Swagger UI title and description
    - Add API examples and documentation
    - Test file upload functionality in Swagger UI
    - _Requirements: 6.5, 6.6_

- [ ] 10. Implement configuration management
  - [ ] 10.1 Create configuration module
    - Load settings from environment variables
    - Provide defaults for all configuration options
    - Validate configuration on startup
    - Support chunk size, model names, vector store selection
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  
  - [ ]* 10.2 Write property test for configuration validation
    - **Property 16: Configuration validation**
    - **Validates: Requirements 10.7**

- [ ] 11. Checkpoint - Backend complete
  - Ensure all backend tests pass
  - Test API endpoints using Swagger UI at http://localhost:8000/docs
  - Verify document upload, processing, and Q&A flow works end-to-end
  - Test file upload widget in Swagger UI
  - Ask the user if questions arise

- [ ] 12. Implement Docker containerization
  - [ ] 12.1 Create backend Dockerfile
    - Use Python 3.10+ base image
    - Install dependencies from requirements.txt
    - Copy application code
    - Set up non-root user
    - Expose port 8000
    - _Requirements: 7.1, 7.2_
  
  - [ ] 12.2 Create docker-compose.yml
    - Define services: backend, ollama
    - Configure network connectivity between services
    - Set up volume mounts for persistent data
    - Configure environment variables
    - Add health checks for all services
    - Expose port 8000 for backend API and Swagger UI
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  
  - [ ] 12.3 Create setup documentation
    - Write README with setup instructions
    - Document how to pull Ollama models
    - Provide docker-compose commands
    - Include instructions for accessing Swagger UI
    - Include troubleshooting section
    - _Requirements: 7.7_

- [ ] 13. Add multi-document search support
  - [ ] 13.1 Update RAG engine for multi-document queries
    - Modify search to query across all documents by default
    - Support optional document_ids filter
    - Ensure results can come from multiple documents
    - _Requirements: 11.5_
  
  - [ ]* 13.2 Write property test for multi-document search
    - **Property 19: Multi-document search**
    - **Validates: Requirements 11.5**

- [ ] 14. Final integration and testing
  - [ ] 14.1 Run full integration tests
    - Test complete flow: upload → process → query → answer
    - Test with multiple documents
    - Test document deletion and cleanup
    - Test error scenarios
    - _Requirements: All_
  
  - [ ] 14.2 Run property-based test suite
    - Execute all property tests with 100+ iterations
    - Verify all properties pass
    - Fix any discovered edge cases
    - _Requirements: All testable requirements_
  
  - [ ] 14.3 Test Docker deployment
    - Build all Docker images
    - Start services with docker-compose
    - Verify service connectivity
    - Test volume persistence across restarts
    - Verify health checks work
    - Access Swagger UI and test all endpoints
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 15. Final checkpoint - System complete
  - Ensure all tests pass (unit, property, integration)
  - Verify Docker deployment works end-to-end
  - Verify Swagger UI is fully functional for testing
  - Review documentation completeness
  - Ask the user if questions arise or if ready for deployment

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Property tests validate universal correctness properties across many inputs
- Unit tests validate specific examples, edge cases, and integration points
- The implementation follows a bottom-up approach: core components → API → Docker deployment
- All property tests should run with minimum 100 iterations
- Use hypothesis library for property-based testing in Python
- Mock external services (Ollama) in unit tests for reliability
- FastAPI automatically generates Swagger UI at /docs for interactive API testing
- No frontend development required - Swagger UI provides all testing capabilities
