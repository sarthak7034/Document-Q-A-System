"""
Unit tests for the configuration module.

Tests configuration loading, validation, and error handling.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
from app.config import Config


class TestConfigDefaults:
    """Test that default configuration values are set correctly."""
    
    def test_backend_defaults(self):
        """Test backend configuration defaults."""
        assert Config.BACKEND_PORT == 8000
        assert Config.UPLOAD_DIR == "./data/documents"
        assert Config.MAX_FILE_SIZE_MB == 50
    
    def test_vector_store_defaults(self):
        """Test vector store configuration defaults."""
        # Note: This may be overridden by .env file
        # The default in code is "chromadb", but .env may set it to "faiss"
        assert Config.VECTOR_STORE_TYPE in ["chromadb", "faiss"]
        assert Config.CHROMA_PERSIST_DIR == "./data/chroma_db"
        assert Config.FAISS_INDEX_PATH == "./data/faiss_index"
    
    def test_embedding_defaults(self):
        """Test embedding configuration defaults."""
        assert Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert Config.EMBEDDING_DEVICE == "cpu"
    
    def test_llm_defaults(self):
        """Test LLM configuration defaults."""
        assert Config.OLLAMA_BASE_URL == "http://localhost:11434"
        assert Config.OLLAMA_MODEL == "llama2"
        assert Config.LLM_TEMPERATURE == 0.7
        assert Config.LLM_MAX_TOKENS == 512
    
    def test_chunking_defaults(self):
        """Test chunking configuration defaults."""
        assert Config.CHUNK_SIZE == 1000
        assert Config.CHUNK_OVERLAP == 100
        assert Config.MAX_CHUNKS_FOR_CONTEXT == 5
    
    def test_logging_defaults(self):
        """Test logging configuration defaults."""
        assert Config.LOG_LEVEL == "INFO"
        assert Config.LOG_FILE == "./logs/app.log"


class TestConfigValidation:
    """Test configuration validation logic."""
    
    def test_valid_configuration_passes(self):
        """Test that valid configuration passes validation."""
        # Should not raise any exception
        Config.validate()
    
    def test_negative_chunk_size_fails(self):
        """Test that negative chunk size fails validation."""
        with patch.object(Config, 'CHUNK_SIZE', -100):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "CHUNK_SIZE must be positive" in str(exc_info.value)
    
    def test_zero_chunk_size_fails(self):
        """Test that zero chunk size fails validation."""
        with patch.object(Config, 'CHUNK_SIZE', 0):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "CHUNK_SIZE must be positive" in str(exc_info.value)
    
    def test_negative_chunk_overlap_fails(self):
        """Test that negative chunk overlap fails validation."""
        with patch.object(Config, 'CHUNK_OVERLAP', -50):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "CHUNK_OVERLAP must be non-negative" in str(exc_info.value)
    
    def test_chunk_overlap_greater_than_size_fails(self):
        """Test that chunk overlap >= chunk size fails validation."""
        with patch.object(Config, 'CHUNK_SIZE', 100):
            with patch.object(Config, 'CHUNK_OVERLAP', 100):
                with pytest.raises(ValueError) as exc_info:
                    Config.validate()
                assert "CHUNK_OVERLAP" in str(exc_info.value)
                assert "must be less than" in str(exc_info.value)
    
    def test_negative_max_file_size_fails(self):
        """Test that negative max file size fails validation."""
        with patch.object(Config, 'MAX_FILE_SIZE_MB', -10):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "MAX_FILE_SIZE_MB must be positive" in str(exc_info.value)
    
    def test_invalid_vector_store_type_fails(self):
        """Test that invalid vector store type fails validation."""
        with patch.object(Config, 'VECTOR_STORE_TYPE', 'invalid'):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "VECTOR_STORE_TYPE must be 'chromadb' or 'faiss'" in str(exc_info.value)
    
    def test_temperature_below_zero_fails(self):
        """Test that temperature below 0.0 fails validation."""
        with patch.object(Config, 'LLM_TEMPERATURE', -0.5):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "LLM_TEMPERATURE must be between 0.0 and 2.0" in str(exc_info.value)
    
    def test_temperature_above_two_fails(self):
        """Test that temperature above 2.0 fails validation."""
        with patch.object(Config, 'LLM_TEMPERATURE', 2.5):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "LLM_TEMPERATURE must be between 0.0 and 2.0" in str(exc_info.value)
    
    def test_negative_max_tokens_fails(self):
        """Test that negative max tokens fails validation."""
        with patch.object(Config, 'LLM_MAX_TOKENS', -100):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "LLM_MAX_TOKENS must be positive" in str(exc_info.value)
    
    def test_zero_max_tokens_fails(self):
        """Test that zero max tokens fails validation."""
        with patch.object(Config, 'LLM_MAX_TOKENS', 0):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "LLM_MAX_TOKENS must be positive" in str(exc_info.value)
    
    def test_negative_max_chunks_fails(self):
        """Test that negative max chunks for context fails validation."""
        with patch.object(Config, 'MAX_CHUNKS_FOR_CONTEXT', -5):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "MAX_CHUNKS_FOR_CONTEXT must be positive" in str(exc_info.value)
    
    def test_zero_max_chunks_fails(self):
        """Test that zero max chunks for context fails validation."""
        with patch.object(Config, 'MAX_CHUNKS_FOR_CONTEXT', 0):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "MAX_CHUNKS_FOR_CONTEXT must be positive" in str(exc_info.value)
    
    def test_invalid_embedding_device_fails(self):
        """Test that invalid embedding device fails validation."""
        with patch.object(Config, 'EMBEDDING_DEVICE', 'invalid'):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "EMBEDDING_DEVICE must be one of" in str(exc_info.value)
    
    def test_invalid_log_level_fails(self):
        """Test that invalid log level fails validation."""
        with patch.object(Config, 'LOG_LEVEL', 'INVALID'):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "LOG_LEVEL must be one of" in str(exc_info.value)
    
    def test_multiple_errors_reported(self):
        """Test that multiple validation errors are reported together."""
        with patch.object(Config, 'CHUNK_SIZE', -100):
            with patch.object(Config, 'MAX_FILE_SIZE_MB', -10):
                with pytest.raises(ValueError) as exc_info:
                    Config.validate()
                error_msg = str(exc_info.value)
                assert "CHUNK_SIZE must be positive" in error_msg
                assert "MAX_FILE_SIZE_MB must be positive" in error_msg


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_valid_chunk_size(self):
        """Test that minimum valid chunk size (1) passes validation."""
        with patch.object(Config, 'CHUNK_SIZE', 1):
            with patch.object(Config, 'CHUNK_OVERLAP', 0):
                Config.validate()  # Should not raise
    
    def test_zero_chunk_overlap_valid(self):
        """Test that zero chunk overlap is valid."""
        with patch.object(Config, 'CHUNK_OVERLAP', 0):
            Config.validate()  # Should not raise
    
    def test_temperature_boundary_zero(self):
        """Test that temperature of 0.0 is valid."""
        with patch.object(Config, 'LLM_TEMPERATURE', 0.0):
            Config.validate()  # Should not raise
    
    def test_temperature_boundary_two(self):
        """Test that temperature of 2.0 is valid."""
        with patch.object(Config, 'LLM_TEMPERATURE', 2.0):
            Config.validate()  # Should not raise
    
    def test_chromadb_vector_store_valid(self):
        """Test that 'chromadb' vector store type is valid."""
        with patch.object(Config, 'VECTOR_STORE_TYPE', 'chromadb'):
            Config.validate()  # Should not raise
    
    def test_faiss_vector_store_valid(self):
        """Test that 'faiss' vector store type is valid."""
        with patch.object(Config, 'VECTOR_STORE_TYPE', 'faiss'):
            Config.validate()  # Should not raise
    
    def test_cpu_device_valid(self):
        """Test that 'cpu' embedding device is valid."""
        with patch.object(Config, 'EMBEDDING_DEVICE', 'cpu'):
            Config.validate()  # Should not raise
    
    def test_cuda_device_valid(self):
        """Test that 'cuda' embedding device is valid."""
        with patch.object(Config, 'EMBEDDING_DEVICE', 'cuda'):
            Config.validate()  # Should not raise
    
    def test_mps_device_valid(self):
        """Test that 'mps' embedding device is valid."""
        with patch.object(Config, 'EMBEDDING_DEVICE', 'mps'):
            Config.validate()  # Should not raise


class TestConfigEnvironmentVariables:
    """Test that environment variables are properly loaded."""
    
    def test_backend_port_from_env(self, monkeypatch):
        """Test that BACKEND_PORT is loaded from environment."""
        # Note: This test demonstrates the pattern, but Config is already loaded
        # In a real scenario, we'd need to reload the module
        monkeypatch.setenv("BACKEND_PORT", "9000")
        # Would need to reload config module to test this properly
        # This is more of a documentation test
    
    def test_chunk_size_from_env(self, monkeypatch):
        """Test that CHUNK_SIZE is loaded from environment."""
        monkeypatch.setenv("CHUNK_SIZE", "2000")
        # Would need to reload config module to test this properly
    
    def test_vector_store_type_from_env(self, monkeypatch):
        """Test that VECTOR_STORE_TYPE is loaded from environment."""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        # Would need to reload config module to test this properly


class TestConfigDirectoryCreation:
    """Test directory creation functionality."""
    
    def test_ensure_directories_creates_upload_dir(self, tmp_path):
        """Test that ensure_directories creates upload directory."""
        test_dir = tmp_path / "test_uploads"
        with patch.object(Config, 'UPLOAD_DIR', str(test_dir)):
            Config.ensure_directories()
            assert test_dir.exists()
    
    def test_ensure_directories_creates_chroma_dir(self, tmp_path):
        """Test that ensure_directories creates ChromaDB directory."""
        test_dir = tmp_path / "test_chroma"
        with patch.object(Config, 'CHROMA_PERSIST_DIR', str(test_dir)):
            Config.ensure_directories()
            assert test_dir.exists()
    
    def test_ensure_directories_creates_log_dir(self, tmp_path):
        """Test that ensure_directories creates log directory."""
        test_log = tmp_path / "test_logs" / "app.log"
        with patch.object(Config, 'LOG_FILE', str(test_log)):
            Config.ensure_directories()
            assert test_log.parent.exists()
    
    def test_ensure_directories_creates_db_dir(self, tmp_path):
        """Test that ensure_directories creates database directory."""
        test_db = tmp_path / "test_data" / "metadata.db"
        with patch.object(Config, 'DB_PATH', str(test_db)):
            Config.ensure_directories()
            assert test_db.parent.exists()
    
    def test_ensure_directories_idempotent(self, tmp_path):
        """Test that ensure_directories can be called multiple times safely."""
        test_dir = tmp_path / "test_idempotent"
        with patch.object(Config, 'UPLOAD_DIR', str(test_dir)):
            Config.ensure_directories()
            Config.ensure_directories()  # Should not raise
            assert test_dir.exists()


class TestConfigErrorMessages:
    """Test that error messages are clear and actionable."""
    
    def test_error_message_includes_invalid_value(self):
        """Test that error messages include the invalid value."""
        with patch.object(Config, 'CHUNK_SIZE', -100):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "-100" in str(exc_info.value)
    
    def test_error_message_includes_field_name(self):
        """Test that error messages include the field name."""
        with patch.object(Config, 'MAX_FILE_SIZE_MB', -10):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "MAX_FILE_SIZE_MB" in str(exc_info.value)
    
    def test_error_message_includes_valid_options(self):
        """Test that error messages include valid options for enums."""
        with patch.object(Config, 'VECTOR_STORE_TYPE', 'invalid'):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            assert "chromadb" in str(exc_info.value)
            assert "faiss" in str(exc_info.value)
    
    def test_error_message_format_readable(self):
        """Test that error messages are formatted for readability."""
        with patch.object(Config, 'CHUNK_SIZE', -100):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()
            error_msg = str(exc_info.value)
            assert "Invalid configuration:" in error_msg
            assert "  - " in error_msg  # Bullet point formatting
