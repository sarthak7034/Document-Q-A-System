"""
Embedding Service for generating vector embeddings from text.

This module provides the EmbeddingService class that uses sentence-transformers
to generate embeddings for document chunks and questions.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    This class loads a pre-trained sentence-transformers model and provides
    methods to generate embeddings for single texts or batches of texts.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingService with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is "all-MiniLM-L6-v2" (384 dimensions, fast, good quality).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed.
        
        Returns:
            A numpy array containing the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            A numpy array of shape (len(texts), embedding_dimension) containing
            the embedding vectors for all input texts.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.
        
        Returns:
            The number of dimensions in the embedding vectors produced by this model.
        """
        return self.dimension
