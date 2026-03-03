"""
Main FastAPI application for Document Q&A System.

This module initializes the FastAPI app with automatic Swagger UI,
sets up logging, and configures all API routes.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .api import router
from .dependencies import get_services
from .middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware


# Configure logging
def setup_logging():
    """Configure application logging."""
    log_level = getattr(logging, Config.LOG_LEVEL.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler (if configured)
    handlers = [console_handler]
    if Config.LOG_FILE:
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    This initializes services on startup and cleans up on shutdown.
    """
    # Validate configuration first
    Config.validate()
    Config.ensure_directories()
    
    logger.info("Starting Document Q&A System...")
    logger.info(f"Configuration: Vector Store={Config.VECTOR_STORE_TYPE}, "
                f"Embedding Model={Config.EMBEDDING_MODEL}, "
                f"LLM={Config.OLLAMA_MODEL}")
    
    # Initialize services (this will load models, etc.)
    try:
        services = get_services()
        logger.info("Services initialized successfully")
        
        # Check LLM health
        llm_healthy = await services['llm_service'].check_health()
        if llm_healthy:
            logger.info(f"LLM service is healthy at {Config.OLLAMA_BASE_URL}")
        else:
            logger.warning(
                f"LLM service is not accessible at {Config.OLLAMA_BASE_URL}. "
                "Question answering will not work until Ollama is started."
            )
    except Exception as e:
        logger.error(f"Error initializing services: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down Document Q&A System...")


# Create FastAPI application
app = FastAPI(
    title="Document Q&A System",
    description="""
    A locally-hosted backend for PDF document Q&A using RAG (Retrieval Augmented Generation).
    
    ## Features
    
    * **Upload PDF Documents**: Upload documents up to 50MB for processing
    * **Ask Questions**: Query your documents using natural language
    * **Manage Documents**: List, view details, and delete documents
    * **Local Processing**: All AI operations run locally for privacy
    
    ## How It Works
    
    1. Upload a PDF document via the `/api/documents` endpoint
    2. The system extracts text, chunks it, and generates embeddings
    3. Ask questions via the `/api/questions` endpoint
    4. The system retrieves relevant chunks and generates answers using a local LLM
    
    ## Technology Stack
    
    * **FastAPI**: Modern Python web framework
    * **Sentence Transformers**: Local embedding generation
    * **ChromaDB**: Vector database for similarity search
    * **Ollama**: Local LLM inference
    
    ## Getting Started
    
    1. Ensure Ollama is running: `docker-compose up ollama` or `ollama serve`
    2. Upload a document using the file upload widget below
    3. Ask questions about your documents
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add error handling middleware
app.add_middleware(ErrorHandlingMiddleware)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API and links to documentation.
    """
    return {
        "message": "Document Q&A System API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=Config.BACKEND_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )
