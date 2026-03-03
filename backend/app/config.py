"""
Configuration module for the Document Q&A System.

This module loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # Backend settings
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/documents")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Vector Store settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    
    # Embeddings settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # LLM settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_CHUNKS_FOR_CONTEXT: int = int(os.getenv("MAX_CHUNKS_FOR_CONTEXT", "5"))
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "./logs/app.log")
    
    # Database settings
    DB_PATH: str = os.getenv("DB_PATH", "./data/metadata.db")
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        errors = []
        
        # Validate chunk size
        if cls.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got {cls.CHUNK_SIZE}")
        
        # Validate chunk overlap
        if cls.CHUNK_OVERLAP < 0:
            errors.append(f"CHUNK_OVERLAP must be non-negative, got {cls.CHUNK_OVERLAP}")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )
        
        # Validate max file size
        if cls.MAX_FILE_SIZE_MB <= 0:
            errors.append(f"MAX_FILE_SIZE_MB must be positive, got {cls.MAX_FILE_SIZE_MB}")
        
        # Validate vector store type
        if cls.VECTOR_STORE_TYPE not in ["chromadb", "faiss"]:
            errors.append(
                f"VECTOR_STORE_TYPE must be 'chromadb' or 'faiss', "
                f"got '{cls.VECTOR_STORE_TYPE}'"
            )
        
        # Validate temperature
        if not 0.0 <= cls.LLM_TEMPERATURE <= 2.0:
            errors.append(
                f"LLM_TEMPERATURE must be between 0.0 and 2.0, "
                f"got {cls.LLM_TEMPERATURE}"
            )
        
        # Validate max tokens
        if cls.LLM_MAX_TOKENS <= 0:
            errors.append(f"LLM_MAX_TOKENS must be positive, got {cls.LLM_MAX_TOKENS}")
        
        # Validate max chunks for context
        if cls.MAX_CHUNKS_FOR_CONTEXT <= 0:
            errors.append(
                f"MAX_CHUNKS_FOR_CONTEXT must be positive, "
                f"got {cls.MAX_CHUNKS_FOR_CONTEXT}"
            )
        
        # Validate embedding device
        valid_devices = ["cpu", "cuda", "mps"]
        if cls.EMBEDDING_DEVICE.lower() not in valid_devices:
            errors.append(
                f"EMBEDDING_DEVICE must be one of {valid_devices}, "
                f"got '{cls.EMBEDDING_DEVICE}'"
            )
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, "
                f"got '{cls.LOG_LEVEL}'"
            )
        
        if errors:
            raise ValueError(
                "Invalid configuration:\n" + "\n".join(f"  - {err}" for err in errors)
            )
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.UPLOAD_DIR,
            cls.CHROMA_PERSIST_DIR,
            Path(cls.DB_PATH).parent,
        ]
        
        if cls.LOG_FILE:
            directories.append(Path(cls.LOG_FILE).parent)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Note: Validation is called explicitly by the application at startup
# This allows tests to patch configuration values before validation
# Config.validate()
# Config.ensure_directories()
