"""Text chunking module with overlap and sentence boundary preservation.

This module provides functionality to split text into chunks with configurable
size and overlap, while preserving sentence boundaries.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    token_count: int
    char_count: int
    metadata: Dict[str, Any]


def count_tokens(text: str) -> int:
    """Estimate token count by splitting on whitespace.
    
    This is a simple approximation. For more accurate token counting,
    consider using a tokenizer like tiktoken.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving sentence boundaries.
    
    Uses regex to split on common sentence-ending punctuation followed by
    whitespace or end of string.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Pattern matches sentence-ending punctuation (. ! ?) followed by space or end
    # Handles common abbreviations like Dr., Mr., etc.
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    
    sentences = re.split(sentence_pattern, text)
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    page_number: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    document_id: str = ""
) -> List[Chunk]:
    """Split text into overlapping chunks while preserving sentence boundaries.
    
    Args:
        text: Input text to chunk
        page_number: Page number this text came from
        chunk_size: Target chunk size in tokens (default: 1000)
        chunk_overlap: Overlap between chunks in tokens (default: 100)
        document_id: Document identifier for chunk IDs
        
    Returns:
        List of Chunk objects with metadata
    """
    if not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    chunks = []
    chunk_index = 0
    
    current_chunk_sentences = []
    current_token_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_token_count > 0 and current_token_count + sentence_tokens > chunk_size:
            # Create chunk from accumulated sentences
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_id = f"{document_id}_page{page_number}_chunk{chunk_index}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                page_number=page_number,
                chunk_index=chunk_index,
                token_count=current_token_count,
                char_count=len(chunk_text),
                metadata={
                    'document_id': document_id,
                    'page_number': page_number,
                    'chunk_index': chunk_index
                }
            ))
            
            chunk_index += 1
            
            # Calculate overlap: keep sentences that total ~chunk_overlap tokens
            overlap_sentences = []
            overlap_tokens = 0
            
            # Work backwards from current position to build overlap
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                sent = current_chunk_sentences[j]
                sent_tokens = count_tokens(sent)
                
                if overlap_tokens + sent_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            i += 1
    
    # Handle remaining sentences
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunk_id = f"{document_id}_page{page_number}_chunk{chunk_index}"
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            page_number=page_number,
            chunk_index=chunk_index,
            token_count=current_token_count,
            char_count=len(chunk_text),
            metadata={
                'document_id': document_id,
                'page_number': page_number,
                'chunk_index': chunk_index
            }
        ))
    
    return chunks


def chunk_pages(
    pages: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    document_id: str = ""
) -> List[Chunk]:
    """Chunk multiple pages of text.
    
    Args:
        pages: List of Page objects with page_number and text attributes
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        document_id: Document identifier
        
    Returns:
        List of all chunks from all pages
    """
    all_chunks = []
    global_chunk_index = 0
    
    for page in pages:
        page_chunks = chunk_text(
            text=page.text,
            page_number=page.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=document_id
        )
        
        # Update chunk indices to be globally unique
        for chunk in page_chunks:
            chunk.chunk_index = global_chunk_index
            chunk.chunk_id = f"{document_id}_chunk{global_chunk_index}"
            chunk.metadata['chunk_index'] = global_chunk_index
            global_chunk_index += 1
        
        all_chunks.extend(page_chunks)
    
    return all_chunks
