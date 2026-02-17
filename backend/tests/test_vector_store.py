"""
Unit tests for VectorStore implementations.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from app.vector_store import VectorStore, ChromaVectorStore, Chunk, SearchResult


class TestChromaVectorStore:
    """Tests for ChromaDB vector store implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a ChromaVectorStore instance for testing."""
        return ChromaVectorStore(persist_directory=temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                chunk_id="doc1_chunk_0",
                text="This is the first chunk about machine learning.",
                page_number=1,
                chunk_index=0,
                document_id="doc1",
                metadata={"char_count": 48}
            ),
            Chunk(
                chunk_id="doc1_chunk_1",
                text="This is the second chunk about deep learning.",
                page_number=1,
                chunk_index=1,
                document_id="doc1",
                metadata={"char_count": 46}
            ),
            Chunk(
                chunk_id="doc2_chunk_0",
                text="This chunk is from a different document about AI.",
                page_number=1,
                chunk_index=0,
                document_id="doc2",
                metadata={"char_count": 50}
            )
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create 3 embeddings with dimension 384 (typical for sentence transformers)
        np.random.seed(42)
        return np.random.rand(3, 384).astype(np.float32)
    
    def test_add_documents(self, vector_store, sample_chunks, sample_embeddings):
        """Test adding documents to the vector store."""
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        # Verify documents were added
        count = vector_store.get_document_count()
        assert count == 2  # Two unique documents
    
    def test_add_documents_empty(self, vector_store):
        """Test adding empty list of documents."""
        vector_store.add_documents([], np.array([]))
        count = vector_store.get_document_count()
        assert count == 0
    
    def test_add_documents_mismatch(self, vector_store, sample_chunks):
        """Test error when chunks and embeddings don't match."""
        wrong_embeddings = np.random.rand(2, 384).astype(np.float32)
        
        with pytest.raises(ValueError, match="must match"):
            vector_store.add_documents(sample_chunks, wrong_embeddings)
    
    def test_search(self, vector_store, sample_chunks, sample_embeddings):
        """Test searching for similar chunks."""
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        # Search with the first embedding
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        
        # First result should be most similar (likely the same chunk)
        if results:
            assert results[0].chunk.chunk_id == "doc1_chunk_0"
            assert 0.0 <= results[0].similarity_score <= 1.0
    
    def test_search_with_document_filter(self, vector_store, sample_chunks, sample_embeddings):
        """Test searching with document ID filter."""
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        # Search only in doc2
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=5, document_ids=["doc2"])
        
        assert len(results) == 1
        assert results[0].chunk.document_id == "doc2"
    
    def test_search_empty_store(self, vector_store):
        """Test searching in empty store."""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=5)
        
        assert len(results) == 0
    
    def test_delete_document(self, vector_store, sample_chunks, sample_embeddings):
        """Test deleting a document."""
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        # Verify initial state
        assert vector_store.get_document_count() == 2
        
        # Delete doc1
        vector_store.delete_document("doc1")
        
        # Verify doc1 is gone
        assert vector_store.get_document_count() == 1
        
        # Search should only return doc2 chunks
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=5)
        
        assert all(r.chunk.document_id == "doc2" for r in results)
    
    def test_delete_nonexistent_document(self, vector_store):
        """Test deleting a document that doesn't exist."""
        # Should not raise an error
        vector_store.delete_document("nonexistent")
        assert vector_store.get_document_count() == 0
    
    def test_get_document_count(self, vector_store, sample_chunks, sample_embeddings):
        """Test getting document count."""
        assert vector_store.get_document_count() == 0
        
        vector_store.add_documents(sample_chunks, sample_embeddings)
        assert vector_store.get_document_count() == 2
        
        # Add more chunks from existing document
        new_chunk = Chunk(
            chunk_id="doc1_chunk_2",
            text="Another chunk from doc1.",
            page_number=2,
            chunk_index=2,
            document_id="doc1",
            metadata={}
        )
        new_embedding = np.random.rand(1, 384).astype(np.float32)
        vector_store.add_documents([new_chunk], new_embedding)
        
        # Count should still be 2 (same documents)
        assert vector_store.get_document_count() == 2
    
    def test_persistence(self, temp_dir, sample_chunks, sample_embeddings):
        """Test that data persists across instances."""
        # Create first instance and add data
        store1 = ChromaVectorStore(persist_directory=temp_dir)
        store1.add_documents(sample_chunks, sample_embeddings)
        
        # Create second instance with same directory
        store2 = ChromaVectorStore(persist_directory=temp_dir)
        
        # Data should be persisted
        assert store2.get_document_count() == 2
        
        # Search should work
        query_embedding = sample_embeddings[0]
        results = store2.search(query_embedding, top_k=1)
        assert len(results) > 0
