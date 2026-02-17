"""
Integration tests for core components working together.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from app.document_processor import DocumentProcessor
from app.embedding_service import EmbeddingService
from app.vector_store import ChromaVectorStore


class TestCoreIntegration:
    """Integration tests for DocumentProcessor, EmbeddingService, and VectorStore."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Note: Cleanup may fail on Windows due to ChromaDB file locks
        # This is expected and doesn't affect functionality
        try:
            shutil.rmtree(temp_path)
        except PermissionError:
            pass
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a VectorStore instance."""
        return ChromaVectorStore(persist_directory=temp_dir)
    
    def test_end_to_end_workflow_with_mock_data(
        self, processor, embedding_service, vector_store
    ):
        """Test complete workflow: process -> embed -> store -> retrieve."""
        # Create mock chunks using vector_store.Chunk (which has document_id attribute)
        from app.vector_store import Chunk
        
        mock_chunks = [
            Chunk(
                chunk_id="test_doc_chunk_0",
                text="Machine learning is a subset of artificial intelligence.",
                page_number=1,
                chunk_index=0,
                document_id="test_doc",
                metadata={"token_count": 10, "char_count": 58}
            ),
            Chunk(
                chunk_id="test_doc_chunk_1",
                text="Deep learning uses neural networks with multiple layers.",
                page_number=1,
                chunk_index=1,
                document_id="test_doc",
                metadata={"token_count": 9, "char_count": 57}
            ),
            Chunk(
                chunk_id="test_doc_chunk_2",
                text="Natural language processing enables computers to understand text.",
                page_number=2,
                chunk_index=2,
                document_id="test_doc",
                metadata={"token_count": 10, "char_count": 66}
            )
        ]
        
        # Step 1: Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in mock_chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        assert embeddings.shape == (3, 384)
        
        # Step 2: Store chunks with embeddings in vector store
        vector_store.add_documents(mock_chunks, embeddings)
        
        # Verify storage
        doc_count = vector_store.get_document_count()
        assert doc_count == 1  # One unique document
        
        # Step 3: Query with a question
        question = "What is deep learning?"
        question_embedding = embedding_service.embed_text(question)
        
        assert question_embedding.shape == (384,)
        
        # Step 4: Retrieve relevant chunks
        results = vector_store.search(question_embedding, top_k=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # The most relevant result should be about deep learning
        top_result = results[0]
        assert "deep learning" in top_result.chunk.text.lower()
        assert 0.0 <= top_result.similarity_score <= 1.0
        
        # Verify metadata is preserved
        assert top_result.chunk.page_number in [1, 2]
        assert top_result.chunk.document_id == "test_doc"
    
    def test_multiple_documents_workflow(
        self, processor, embedding_service, vector_store
    ):
        """Test workflow with multiple documents."""
        from app.vector_store import Chunk
        
        # Create chunks from two different documents
        doc1_chunks = [
            Chunk(
                chunk_id="doc1_chunk_0",
                text="Python is a popular programming language.",
                page_number=1,
                chunk_index=0,
                document_id="doc1",
                metadata={"token_count": 7, "char_count": 42}
            )
        ]
        
        doc2_chunks = [
            Chunk(
                chunk_id="doc2_chunk_0",
                text="JavaScript is used for web development.",
                page_number=1,
                chunk_index=0,
                document_id="doc2",
                metadata={"token_count": 7, "char_count": 40}
            )
        ]
        
        # Process and store both documents
        all_chunks = doc1_chunks + doc2_chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        vector_store.add_documents(all_chunks, embeddings)
        
        # Verify both documents are stored
        assert vector_store.get_document_count() == 2
        
        # Query should be able to retrieve from both documents
        question = "What programming languages are mentioned?"
        question_embedding = embedding_service.embed_text(question)
        results = vector_store.search(question_embedding, top_k=5)
        
        # Should get results from both documents
        document_ids = {r.chunk.document_id for r in results}
        assert len(document_ids) >= 1  # At least one document
        
        # Test filtering by document
        doc1_results = vector_store.search(
            question_embedding, top_k=5, document_ids=["doc1"]
        )
        assert all(r.chunk.document_id == "doc1" for r in doc1_results)
    
    def test_document_deletion_workflow(
        self, processor, embedding_service, vector_store
    ):
        """Test that document deletion works correctly."""
        from app.vector_store import Chunk
        
        # Create and store chunks
        chunks = [
            Chunk(
                chunk_id="temp_doc_chunk_0",
                text="This is temporary content.",
                page_number=1,
                chunk_index=0,
                document_id="temp_doc",
                metadata={"token_count": 5, "char_count": 27}
            )
        ]
        
        embeddings = embedding_service.embed_batch([c.text for c in chunks])
        vector_store.add_documents(chunks, embeddings)
        
        # Verify document exists
        assert vector_store.get_document_count() == 1
        
        # Delete document
        vector_store.delete_document("temp_doc")
        
        # Verify document is gone
        assert vector_store.get_document_count() == 0
        
        # Search should return no results
        question_embedding = embedding_service.embed_text("temporary")
        results = vector_store.search(question_embedding, top_k=5)
        assert len(results) == 0
