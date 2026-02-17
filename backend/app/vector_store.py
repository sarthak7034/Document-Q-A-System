"""
Vector Store module for managing document embeddings and similarity search.

This module provides an abstract base class for vector store implementations
and concrete implementations for ChromaDB.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    document_id: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result with chunk and similarity score."""
    chunk: Chunk
    similarity_score: float


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Store chunks with their embeddings.
        
        Args:
            chunks: List of document chunks to store
            embeddings: Numpy array of embeddings corresponding to chunks
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Find most similar chunks to the query embedding.
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            document_ids: Optional filter to search only specific documents
            
        Returns:
            List of SearchResult objects ordered by similarity (highest first)
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        Remove all chunks for a document.
        
        Args:
            document_id: ID of the document to delete
        """
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """
        Return total number of indexed documents.
        
        Returns:
            Count of unique documents in the store
        """
        pass



class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory path for persistent storage
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Store chunks with their embeddings in ChromaDB.
        
        Args:
            chunks: List of document chunks to store
            embeddings: Numpy array of embeddings corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        if len(chunks) == 0:
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Convert embeddings to list format for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Find most similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            document_ids: Optional filter to search only specific documents
            
        Returns:
            List of SearchResult objects ordered by similarity (highest first)
        """
        # Build where filter if document_ids provided
        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        # Convert results to SearchResult objects
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (cosine similarity)
                # ChromaDB returns squared L2 distance for cosine space
                # Convert to similarity: similarity = 1 - distance
                similarity_score = 1.0 - distance
                
                # Reconstruct Chunk object
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=document,
                    page_number=metadata['page_number'],
                    chunk_index=metadata['chunk_index'],
                    document_id=metadata['document_id'],
                    metadata={
                        k: v for k, v in metadata.items() 
                        if k not in ['document_id', 'page_number', 'chunk_index']
                    }
                )
                
                search_results.append(
                    SearchResult(chunk=chunk, similarity_score=similarity_score)
                )
        
        return search_results
    
    def delete_document(self, document_id: str) -> None:
        """
        Remove all chunks for a document.
        
        Args:
            document_id: ID of the document to delete
        """
        # Query for all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        # Delete if any chunks found
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def get_document_count(self) -> int:
        """
        Return total number of indexed documents.
        
        Returns:
            Count of unique documents in the store
        """
        # Get all items from collection
        all_items = self.collection.get()
        
        if not all_items['metadatas']:
            return 0
        
        # Extract unique document_ids
        document_ids = set(
            metadata['document_id'] 
            for metadata in all_items['metadatas']
        )
        
        return len(document_ids)
