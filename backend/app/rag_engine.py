"""
RAG Engine for orchestrating retrieval and generation.

This module provides the RAGEngine class that combines vector search with
LLM generation to answer questions based on document context.
"""

from typing import List, Optional
from .vector_store import VectorStore, SearchResult
from .embedding_service import EmbeddingService
from .llm_service import LLMService


# Prompt template for RAG
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based solely on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite specific parts of the context when relevant

Answer:"""


class RAGEngine:
    """
    RAG Engine that orchestrates retrieval and generation.
    
    This class combines vector search to retrieve relevant document chunks
    with LLM generation to produce answers based on the retrieved context.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            vector_store: Vector store for similarity search
            embedding_service: Service for generating embeddings
            llm_service: Service for LLM text generation
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
    
    async def answer_question(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
        similarity_threshold: float = 0.3
    ) -> str:
        """
        Generate answer to a question using RAG.
        
        This method:
        1. Retrieves relevant chunks from the vector store
        2. Constructs a prompt with the retrieved context
        3. Generates an answer using the LLM
        
        Args:
            question: The question to answer
            document_ids: Optional list of document IDs to search within
            max_chunks: Maximum number of chunks to retrieve (default: 5)
            temperature: LLM temperature parameter (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 512)
            similarity_threshold: Minimum similarity score for relevant chunks (default: 0.3)
            
        Returns:
            The generated answer as a string, or a message indicating no relevant
            information was found if all chunks are below the similarity threshold
        """
        # Retrieve relevant context
        context_chunks = self.retrieve_context(
            question=question,
            document_ids=document_ids,
            max_chunks=max_chunks
        )
        
        # Filter chunks by similarity threshold
        relevant_chunks = [
            chunk for chunk in context_chunks
            if chunk.similarity_score >= similarity_threshold
        ]
        
        # Handle case when no relevant chunks found
        if not relevant_chunks:
            return (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Please try rephrasing your question or "
                "upload documents that contain information related to your query."
            )
        
        # Construct prompt with context
        prompt = self.construct_prompt(
            question=question,
            context_chunks=relevant_chunks
        )
        
        # Generate answer using LLM
        answer = await self.llm_service.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        return answer
    
    def retrieve_context(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            question: The question to find context for
            document_ids: Optional list of document IDs to search within
            max_chunks: Maximum number of chunks to retrieve (default: 5)
            
        Returns:
            List of SearchResult objects with relevant chunks
        """
        # Generate embedding for the question
        question_embedding = self.embedding_service.embed_text(question)
        
        # Search for similar chunks
        results = self.vector_store.search(
            query_embedding=question_embedding,
            top_k=max_chunks,
            document_ids=document_ids
        )
        
        return results
    
    def construct_prompt(
        self,
        question: str,
        context_chunks: List[SearchResult]
    ) -> str:
        """
        Build prompt with question and retrieved context.
        
        Args:
            question: The question to answer
            context_chunks: List of retrieved SearchResult objects
            
        Returns:
            Formatted prompt string ready for LLM
        """
        # Format context from chunks
        context_parts = []
        for i, result in enumerate(context_chunks, 1):
            chunk = result.chunk
            context_parts.append(
                f"[Document: {chunk.document_id}, Page: {chunk.page_number}]\n"
                f"{chunk.text}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Fill in the template
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        return prompt
