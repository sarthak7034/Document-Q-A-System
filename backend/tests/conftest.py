"""
Pytest configuration and shared fixtures.
"""

import sys
import os
from pathlib import Path
import pytest

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))


@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config class attributes before each test."""
    from app.config import Config
    
    # Store original values
    original_values = {
        'CHUNK_SIZE': int(os.getenv("CHUNK_SIZE", "1000")),
        'CHUNK_OVERLAP': int(os.getenv("CHUNK_OVERLAP", "100")),
        'MAX_FILE_SIZE_MB': int(os.getenv("MAX_FILE_SIZE_MB", "50")),
        'LLM_TEMPERATURE': float(os.getenv("LLM_TEMPERATURE", "0.7")),
        'LLM_MAX_TOKENS': int(os.getenv("LLM_MAX_TOKENS", "512")),
        'MAX_CHUNKS_FOR_CONTEXT': int(os.getenv("MAX_CHUNKS_FOR_CONTEXT", "5")),
        'VECTOR_STORE_TYPE': os.getenv("VECTOR_STORE_TYPE", "chromadb"),
        'EMBEDDING_DEVICE': os.getenv("EMBEDDING_DEVICE", "cpu"),
        'LOG_LEVEL': os.getenv("LOG_LEVEL", "INFO"),
    }
    
    # Reset to original values before test
    for key, value in original_values.items():
        setattr(Config, key, value)
    
    yield
    
    # Reset to original values after test
    for key, value in original_values.items():
        setattr(Config, key, value)

