"""
File storage module for managing document uploads and metadata.

This module handles:
- File size validation (max 50MB)
- Unique document ID generation
- File saving to disk
- SQLite database for document metadata
"""

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


# Constants
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@dataclass
class DocumentMetadata:
    """Document metadata model."""
    id: str
    filename: str
    file_path: str
    upload_date: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    status: str = 'processing'  # processing, ready, error
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FileSizeError(Exception):
    """Raised when file size exceeds the maximum allowed size."""
    pass


class FileStorage:
    """
    Manages file storage and document metadata.
    
    Responsibilities:
    - Validate file sizes before accepting uploads
    - Generate unique document IDs (UUIDs)
    - Save files to the data/documents/ directory
    - Create and manage SQLite database for document metadata
    """

    def __init__(self, 
                 storage_dir: str = "data/documents",
                 db_path: str = "data/metadata.db"):
        """
        Initialize file storage.
        
        Args:
            storage_dir: Directory to store uploaded documents
            db_path: Path to SQLite database file
        """
        self.storage_dir = Path(storage_dir)
        self.db_path = Path(db_path)
        
        # Create directories if they don't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Create the documents table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                page_count INTEGER,
                chunk_count INTEGER,
                status TEXT CHECK(status IN ('processing', 'ready', 'error')),
                error_message TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def validate_file_size(self, file_size: int) -> None:
        """
        Validate that file size is within acceptable limits.
        
        Args:
            file_size: Size of file in bytes
            
        Raises:
            FileSizeError: If file size exceeds MAX_FILE_SIZE_BYTES
        """
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise FileSizeError(
                f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)"
            )

    def generate_document_id(self) -> str:
        """
        Generate a unique document ID.
        
        Returns:
            Unique UUID string
        """
        return str(uuid.uuid4())

    def save_file(self, 
                  file_content: bytes, 
                  filename: str,
                  document_id: Optional[str] = None) -> DocumentMetadata:
        """
        Save uploaded file to disk and create metadata entry.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            DocumentMetadata object with document information
            
        Raises:
            FileSizeError: If file size exceeds maximum
        """
        # Validate file size
        file_size = len(file_content)
        self.validate_file_size(file_size)
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self.generate_document_id()
        
        # Create file path with document ID to avoid filename conflicts
        file_extension = Path(filename).suffix
        stored_filename = f"{document_id}{file_extension}"
        file_path = self.storage_dir / stored_filename
        
        # Save file to disk
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Create metadata entry
        metadata = DocumentMetadata(
            id=document_id,
            filename=filename,
            file_path=str(file_path),
            upload_date=datetime.utcnow().isoformat(),
            status='processing'
        )
        
        # Save metadata to database
        self._save_metadata(metadata)
        
        return metadata

    def _save_metadata(self, metadata: DocumentMetadata) -> None:
        """
        Save document metadata to database.
        
        Args:
            metadata: DocumentMetadata object to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents 
            (id, filename, file_path, upload_date, page_count, chunk_count, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.id,
            metadata.filename,
            metadata.file_path,
            metadata.upload_date,
            metadata.page_count,
            metadata.chunk_count,
            metadata.status,
            metadata.error_message
        ))
        
        conn.commit()
        conn.close()

    def update_metadata(self, 
                       document_id: str,
                       page_count: Optional[int] = None,
                       chunk_count: Optional[int] = None,
                       status: Optional[str] = None,
                       error_message: Optional[str] = None) -> None:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID to update
            page_count: Number of pages in document
            chunk_count: Number of chunks created
            status: Document processing status
            error_message: Error message if status is 'error'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)
        
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if updates:
            params.append(document_id)
            query = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()

    def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            DocumentMetadata object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DocumentMetadata(
                id=row['id'],
                filename=row['filename'],
                file_path=row['file_path'],
                upload_date=row['upload_date'],
                page_count=row['page_count'],
                chunk_count=row['chunk_count'],
                status=row['status'],
                error_message=row['error_message']
            )
        
        return None

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all documents in the database.
        
        Returns:
            List of DocumentMetadata objects
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            DocumentMetadata(
                id=row['id'],
                filename=row['filename'],
                file_path=row['file_path'],
                upload_date=row['upload_date'],
                page_count=row['page_count'],
                chunk_count=row['chunk_count'],
                status=row['status'],
                error_message=row['error_message']
            )
            for row in rows
        ]

    def delete_document(self, document_id: str) -> bool:
        """
        Delete document file and metadata.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        # Get metadata to find file path
        metadata = self.get_metadata(document_id)
        if not metadata:
            return False
        
        # Delete file from disk
        file_path = Path(metadata.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete metadata from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        conn.close()
        
        return True

    def get_file_path(self, document_id: str) -> Optional[str]:
        """
        Get the file path for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            File path string or None if not found
        """
        metadata = self.get_metadata(document_id)
        return metadata.file_path if metadata else None
