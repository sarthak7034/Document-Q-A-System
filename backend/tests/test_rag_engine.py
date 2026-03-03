"""
Unit tests for RAG Engine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from app.rag_engine import RAGEngine, PROMPT_TEMPLATE
from app.vector_store import Chunk, SearchResult


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    return store


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock()
    service.embed_text.return_value = np.array([0.1, 0.2, 0.3])
    return service


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock()
    service.generate = AsyncMock(return_value="This is a test answer.")
    return service


@pytest.fixture
def rag_engine(mock_vector_store, mock_embedding_service, mock_llm_service):
    """Create a RAG engine with mocked dependencies."""
    return RAGEngine(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_service=mock_llm_service
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="doc1_chunk1",
            text="This is the first chunk about Python programming.",
            page_number=1,
            chunk_index=0,
            document_id="doc1",
            metadata={"filename": "python_guide.pdf"}
        ),
        Chunk(
            chunk_id="doc1_chunk2",
            text="This is the second chunk about data structures.",
            page_number=2,
            chunk_index=1,
            document_id="doc1",
            metadata={"filename": "python_guide.pdf"}
        ),
        Chunk(
            chunk_id="doc2_chunk1",
            text="This chunk is from a different document about algorithms.",
            page_number=1,
            chunk_index=0,
            document_id="doc2",
            metadata={"filename": "algorithms.pdf"}
        )
    ]


@pytest.fixture
def sample_search_results(sample_chunks):
    """Create sample search results."""
    return [
        SearchResult(chunk=sample_chunks[0], similarity_score=0.95),
        SearchResult(chunk=sample_chunks[1], similarity_score=0.85),
        SearchResult(chunk=sample_chunks[2], similarity_score=0.75)
    ]


class TestRAGEngineInitialization:
    """Tests for RAG Engine initialization."""
    
    def test_initialization(self, mock_vector_store, mock_embedding_service, mock_llm_service):
        """Test that RAG engine initializes with correct dependencies."""
        engine = RAGEngine(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service
        )
        
        assert engine.vector_store == mock_vector_store
        assert engine.embedding_service == mock_embedding_service
        assert engine.llm_service == mock_llm_service


class TestRetrieveContext:
    """Tests for retrieve_context method."""
    
    def test_retrieve_context_basic(self, rag_engine, mock_embedding_service, 
                                    mock_vector_store, sample_search_results):
        """Test basic context retrieval."""
        question = "What is Python?"
        mock_vector_store.search.return_value = sample_search_results
        
        results = rag_engine.retrieve_context(question)
        
        # Verify embedding was generated for question
        mock_embedding_service.embed_text.assert_called_once_with(question)
        
        # Verify vector store search was called
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]['top_k'] == 5
        assert call_args[1]['document_ids'] is None
        
        # Verify results
        assert results == sample_search_results
    
    def test_retrieve_context_with_document_filter(self, rag_engine, mock_vector_store,
                                                   sample_search_results):
        """Test context retrieval with document ID filter."""
        question = "What is Python?"
        document_ids = ["doc1", "doc2"]
        mock_vector_store.search.return_value = sample_search_results
        
        results = rag_engine.retrieve_context(question, document_ids=document_ids)
        
        # Verify document_ids were passed to search
        call_args = mock_vector_store.search.call_args
        assert call_args[1]['document_ids'] == document_ids
        assert results == sample_search_results
    
    def test_retrieve_context_custom_max_chunks(self, rag_engine, mock_vector_store,
                                               sample_search_results):
        """Test context retrieval with custom max_chunks."""
        question = "What is Python?"
        max_chunks = 10
        mock_vector_store.search.return_value = sample_search_results
        
        results = rag_engine.retrieve_context(question, max_chunks=max_chunks)
        
        # Verify max_chunks was passed as top_k
        call_args = mock_vector_store.search.call_args
        assert call_args[1]['top_k'] == max_chunks
    
    def test_retrieve_context_empty_results(self, rag_engine, mock_vector_store):
        """Test context retrieval when no results found."""
        question = "What is Python?"
        mock_vector_store.search.return_value = []
        
        results = rag_engine.retrieve_context(question)
        
        assert results == []


class TestConstructPrompt:
    """Tests for construct_prompt method."""
    
    def test_construct_prompt_basic(self, rag_engine, sample_search_results):
        """Test basic prompt construction."""
        question = "What is Python?"
        
        prompt = rag_engine.construct_prompt(question, sample_search_results)
        
        # Verify prompt contains question
        assert question in prompt
        
        # Verify prompt contains all chunk texts
        for result in sample_search_results:
            assert result.chunk.text in prompt
        
        # Verify prompt contains document metadata
        assert "doc1" in prompt
        assert "Page: 1" in prompt
        assert "Page: 2" in prompt
    
    def test_construct_prompt_empty_chunks(self, rag_engine):
        """Test prompt construction with no chunks."""
        question = "What is Python?"
        
        prompt = rag_engine.construct_prompt(question, [])
        
        # Verify prompt still contains question
        assert question in prompt
        
        # Verify prompt uses template structure
        assert "Context from documents:" in prompt
        assert "Instructions:" in prompt
    
    def test_construct_prompt_format(self, rag_engine, sample_search_results):
        """Test that prompt follows the expected template format."""
        question = "What is Python?"
        
        prompt = rag_engine.construct_prompt(question, sample_search_results)
        
        # Verify template structure
        assert "You are a helpful assistant" in prompt
        assert "Context from documents:" in prompt
        assert "Question:" in prompt
        assert "Instructions:" in prompt
        assert "Answer:" in prompt


class TestAnswerQuestion:
    """Tests for answer_question method."""
    
    @pytest.mark.asyncio
    async def test_answer_question_basic(self, rag_engine, mock_vector_store,
                                        mock_llm_service, sample_search_results):
        """Test basic question answering."""
        question = "What is Python?"
        mock_vector_store.search.return_value = sample_search_results
        
        answer = await rag_engine.answer_question(question)
        
        # Verify LLM was called
        mock_llm_service.generate.assert_called_once()
        call_args = mock_llm_service.generate.call_args
        
        # Verify prompt was passed
        assert 'prompt' in call_args[1]
        assert question in call_args[1]['prompt']
        
        # Verify default parameters
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 512
        assert call_args[1]['stream'] is False
        
        # Verify answer returned
        assert answer == "This is a test answer."
    
    @pytest.mark.asyncio
    async def test_answer_question_custom_parameters(self, rag_engine, mock_vector_store,
                                                     mock_llm_service, sample_search_results):
        """Test question answering with custom parameters."""
        question = "What is Python?"
        mock_vector_store.search.return_value = sample_search_results
        
        answer = await rag_engine.answer_question(
            question=question,
            document_ids=["doc1"],
            max_chunks=3,
            temperature=0.5,
            max_tokens=256
        )
        
        # Verify custom parameters were used
        call_args = mock_llm_service.generate.call_args
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 256
    
    @pytest.mark.asyncio
    async def test_answer_question_no_relevant_chunks(self, rag_engine, mock_vector_store,
                                                      mock_llm_service):
        """Test question answering when no relevant chunks found."""
        question = "What is Python?"
        mock_vector_store.search.return_value = []
        
        answer = await rag_engine.answer_question(question)
        
        # Verify appropriate message returned
        assert "couldn't find any relevant information" in answer
        
        # Verify LLM was NOT called
        mock_llm_service.generate.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_answer_question_below_similarity_threshold(self, rag_engine, 
                                                              mock_vector_store,
                                                              mock_llm_service,
                                                              sample_chunks):
        """Test question answering when all chunks are below similarity threshold."""
        question = "What is Python?"
        
        # Create search results with low similarity scores
        low_similarity_results = [
            SearchResult(chunk=sample_chunks[0], similarity_score=0.1),
            SearchResult(chunk=sample_chunks[1], similarity_score=0.15)
        ]
        mock_vector_store.search.return_value = low_similarity_results
        
        answer = await rag_engine.answer_question(
            question=question,
            similarity_threshold=0.3
        )
        
        # Verify appropriate message returned
        assert "couldn't find any relevant information" in answer
        
        # Verify LLM was NOT called
        mock_llm_service.generate.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_answer_question_mixed_similarity_scores(self, rag_engine,
                                                           mock_vector_store,
                                                           mock_llm_service,
                                                           sample_chunks):
        """Test that only chunks above threshold are used."""
        question = "What is Python?"
        
        # Create search results with mixed similarity scores
        mixed_results = [
            SearchResult(chunk=sample_chunks[0], similarity_score=0.9),  # Above threshold
            SearchResult(chunk=sample_chunks[1], similarity_score=0.2),  # Below threshold
            SearchResult(chunk=sample_chunks[2], similarity_score=0.5)   # Above threshold
        ]
        mock_vector_store.search.return_value = mixed_results
        
        answer = await rag_engine.answer_question(
            question=question,
            similarity_threshold=0.3
        )
        
        # Verify LLM was called (some chunks above threshold)
        mock_llm_service.generate.assert_called_once()
        
        # Verify only high-similarity chunks were included in prompt
        call_args = mock_llm_service.generate.call_args
        prompt = call_args[1]['prompt']
        
        assert sample_chunks[0].text in prompt  # High similarity
        assert sample_chunks[1].text not in prompt  # Low similarity
        assert sample_chunks[2].text in prompt  # Medium-high similarity


class TestPromptTemplate:
    """Tests for the prompt template."""
    
    def test_prompt_template_structure(self):
        """Test that the prompt template has the expected structure."""
        assert "You are a helpful assistant" in PROMPT_TEMPLATE
        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE
        assert "Instructions:" in PROMPT_TEMPLATE
        assert "Answer:" in PROMPT_TEMPLATE
    
    def test_prompt_template_formatting(self):
        """Test that the prompt template can be formatted correctly."""
        context = "Test context"
        question = "Test question"
        
        formatted = PROMPT_TEMPLATE.format(context=context, question=question)
        
        assert context in formatted
        assert question in formatted
        assert "{context}" not in formatted
        assert "{question}" not in formatted
