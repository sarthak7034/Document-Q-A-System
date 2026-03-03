# Running the FastAPI Backend

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Start the FastAPI Server

```bash
# From the backend directory
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Swagger UI

Open your browser and navigate to:

**http://localhost:8000/docs**

This will open the interactive Swagger UI where you can:
- Upload PDF documents
- Ask questions about your documents
- Manage documents (list, view, delete)
- Test all API endpoints

### 4. Alternative Documentation

ReDoc documentation is also available at:

**http://localhost:8000/redoc**

## API Endpoints

### Document Management

- **POST /api/documents** - Upload a PDF document (max 50MB)
- **GET /api/documents** - List all uploaded documents
- **GET /api/documents/{id}** - Get document details
- **DELETE /api/documents/{id}** - Delete a document

### Question Answering

- **POST /api/questions** - Ask a question about your documents

### Health Check

- **GET /api/health** - Check system health status

## Configuration

The application can be configured using environment variables or a `.env` file:

```bash
# Backend settings
BACKEND_PORT=8000
UPLOAD_DIR=./data/documents
MAX_FILE_SIZE_MB=50

# Vector Store settings
VECTOR_STORE_TYPE=chromadb
CHROMA_PERSIST_DIR=./data/chroma_db

# Embeddings settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# LLM settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# Chunking settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
MAX_CHUNKS_FOR_CONTEXT=5

# Logging settings
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Testing the API

### Using Swagger UI

1. Navigate to http://localhost:8000/docs
2. Click on "POST /api/documents"
3. Click "Try it out"
4. Click "Choose File" and select a PDF
5. Click "Execute"
6. Copy the `document_id` from the response
7. Wait for processing to complete (check status with GET /api/documents/{id})
8. Go to "POST /api/questions"
9. Click "Try it out"
10. Enter your question in the request body
11. Click "Execute"

### Using curl

Upload a document:
```bash
curl -X POST "http://localhost:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

Ask a question:
```bash
curl -X POST "http://localhost:8000/api/questions" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "max_chunks": 5,
    "temperature": 0.7
  }'
```

## Prerequisites

### Ollama (for Question Answering)

The question answering feature requires Ollama to be running:

```bash
# Install Ollama from https://ollama.ai

# Pull a model
ollama pull llama2

# Start Ollama (it usually runs as a service)
ollama serve
```

If Ollama is not running, you can still upload and manage documents, but question answering will not work.

## Logs

Logs are written to:
- Console (stdout)
- File: `./logs/app.log` (if configured)

Log level can be set via the `LOG_LEVEL` environment variable (DEBUG, INFO, WARNING, ERROR, CRITICAL).

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, change the port:

```bash
BACKEND_PORT=8001 python -m app.main
```

### Ollama Connection Error

If you see "LLM service is not accessible":
1. Ensure Ollama is installed and running
2. Check the `OLLAMA_BASE_URL` configuration
3. Verify the model is pulled: `ollama list`

### File Upload Errors

- Ensure the file is a PDF
- Check file size is under 50MB
- Verify the `data/documents` directory exists and is writable

### Import Errors

If you see import errors, ensure you've installed the package:

```bash
pip install -e ".[dev]"
```
