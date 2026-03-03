"""
Dependency injection for FastAPI.

This module provides singleton instances of services for use in API endpoints.
"""

import logging
from functools import lru_cache

from .config import Config
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import ChromaVectorStore
from .llm_service import LLMService
from .rag_engine import RAGEngine
from .file_storage import FileStorage


logger = logging.getLogger(__name__)


@lru_cache()
def get_services():
    """
    Get singleton instances of all services.
    
    This function is cached to ensure we only create one instance of each service.
    
    Returns:
        Dictionary containing all initialized services
    """
    logger.info("Initializing services...")
    
    # Initialize file storage
    file_storage = FileStorage(
        storage_dir=Config.UPLOAD_DIR,
        db_path=Config.DB_PATH
    )
    logger.info("File storage initialized")
    
    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    logger.info("Document processor initialized")
    
    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=Config.EMBEDDING_MODEL
    )
    logger.info(f"Embedding service initialized with model: {Config.EMBEDDING_MODEL}")
    
    # Initialize vector store
    if Config.VECTOR_STORE_TYPE == "chromadb":
        vector_store = ChromaVectorStore(
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        logger.info(f"ChromaDB vector store initialized at: {Config.CHROMA_PERSIST_DIR}")
    else:
        raise ValueError(f"Unsupported vector store type: {Config.VECTOR_STORE_TYPE}")
    
    # Initialize LLM service
    llm_service = LLMService(
        base_url=Config.OLLAMA_BASE_URL,
        model_name=Config.OLLAMA_MODEL
    )
    logger.info(f"LLM service initialized: {Config.OLLAMA_MODEL} at {Config.OLLAMA_BASE_URL}")
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_service=llm_service
    )
    logger.info("RAG engine initialized")
    
    return {
        'file_storage': file_storage,
        'document_processor': document_processor,
        'embedding_service': embedding_service,
        'vector_store': vector_store,
        'llm_service': llm_service,
        'rag_engine': rag_engine
    }


def get_file_storage() -> FileStorage:
    """Get file storage service."""
    return get_services()['file_storage']


def get_document_processor() -> DocumentProcessor:
    """Get document processor service."""
    return get_services()['document_processor']


def get_embedding_service() -> EmbeddingService:
    """Get embedding service."""
    return get_services()['embedding_service']


def get_vector_store() -> ChromaVectorStore:
    """Get vector store service."""
    return get_services()['vector_store']


def get_llm_service() -> LLMService:
    """Get LLM service."""
    return get_services()['llm_service']


def get_rag_engine() -> RAGEngine:
    """Get RAG engine service."""
    return get_services()['rag_engine']
