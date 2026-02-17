"""
Unit tests for EmbeddingService.
"""

import pytest
import numpy as np
from app.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Tests for EmbeddingService class."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    def test_service_initialization(self, embedding_service):
        """Test that service initializes correctly."""
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
        assert embedding_service.dimension > 0
        assert embedding_service.model is not None
    
    def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        dimension = embedding_service.get_embedding_dimension()
        assert dimension == 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    
    def test_embed_text_single(self, embedding_service):
        """Test embedding a single text."""
        text = "This is a test sentence for embedding."
        embedding = embedding_service.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()
    
    def test_embed_text_consistency(self, embedding_service):
        """Test that same text produces same embedding."""
        text = "Consistent embedding test."
        embedding1 = embedding_service.embed_text(text)
        embedding2 = embedding_service.embed_text(text)
        
        # Embeddings should be identical for same text
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_embed_batch(self, embedding_service):
        """Test embedding multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedding_service.embed_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()
    
    def test_embed_batch_empty(self, embedding_service):
        """Test embedding empty list."""
        embeddings = embedding_service.embed_batch([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0
    
    def test_embed_different_texts_different_embeddings(self, embedding_service):
        """Test that different texts produce different embeddings."""
        text1 = "Machine learning is fascinating."
        text2 = "The weather is nice today."
        
        embedding1 = embedding_service.embed_text(text1)
        embedding2 = embedding_service.embed_text(text2)
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2)
