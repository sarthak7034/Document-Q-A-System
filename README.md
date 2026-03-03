# Document Q&A System

A locally-hosted backend application that enables users to upload PDF documents and ask natural language questions about their content via REST API using Retrieval Augmented Generation (RAG).

## Features

- 📄 Upload and process PDF documents
- 🤖 Ask questions using natural language
- 🔍 RAG-based retrieval with vector search
- 🏠 100% local execution (privacy-first)
- 🚀 FastAPI backend with auto-generated Swagger UI
- 🐳 Docker containerization for easy deployment

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- 4GB+ RAM recommended
- 10GB+ disk space for models

### 1. Start the Services

```bash
docker-compose up -d
```

This will start:
- **Backend API** on port 8000
- **Ollama LLM service** on port 11434

### 2. Pull an Ollama Model

After the services are running, pull a language model:

```bash
# Pull Llama 2 (recommended, ~4GB)
docker exec -it document-qa-ollama ollama pull llama2

# Or pull Mistral (~4GB)
docker exec -it document-qa-ollama ollama pull mistral

# Or pull a smaller model like Phi (~1.6GB)
docker exec -it document-qa-ollama ollama pull phi
```

### 3. Access Swagger UI

Open your browser and navigate to:

**http://localhost:8000/docs**

You can now:
- Upload PDF documents using the interactive file upload widget
- Ask questions about your documents
- View API responses with source citations

## Docker Commands

### Start Services
```bash
# Start in detached mode
docker-compose up -d

# Start with logs visible
docker-compose up
```

### Stop Services
```bash
# Stop services
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

### View Logs
```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs ollama
```

### Check Service Health
```bash
# Check running services
docker-compose ps

# Check backend health
curl http://localhost:8000/api/health

# Check Ollama health
curl http://localhost:11434/api/tags
```

### Rebuild After Code Changes
```bash
# Rebuild backend image
docker-compose build backend

# Rebuild and restart
docker-compose up -d --build
```

## API Endpoints

### Document Management
- `POST /api/documents` - Upload a PDF document
- `GET /api/documents` - List all uploaded documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete a document

### Question Answering
- `POST /api/questions` - Submit a question and get an answer

### System
- `GET /api/health` - Health check
- `GET /api/models` - List available Ollama models
- `GET /docs` - Swagger UI (interactive API documentation)

## Example Usage

### 1. Upload a Document

Using Swagger UI at http://localhost:8000/docs:
1. Navigate to `POST /api/documents`
2. Click "Try it out"
3. Click "Choose File" and select a PDF
4. Click "Execute"

Or using curl:
```bash
curl -X POST "http://localhost:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### 2. Ask a Question

Using Swagger UI:
1. Navigate to `POST /api/questions`
2. Click "Try it out"
3. Enter your question in the request body
4. Click "Execute"

Or using curl:
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

## Configuration

Environment variables can be configured in `docker-compose.yml`:

### Backend Configuration
- `BACKEND_PORT` - API port (default: 8000)
- `MAX_FILE_SIZE_MB` - Maximum upload size (default: 50)

### Vector Store
- `VECTOR_STORE_TYPE` - chromadb or faiss (default: chromadb)

### Embeddings
- `EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)
- `EMBEDDING_DEVICE` - cpu or cuda (default: cpu)

### LLM
- `OLLAMA_BASE_URL` - Ollama service URL (default: http://ollama:11434)
- `OLLAMA_MODEL` - Model name (default: llama2)
- `LLM_TEMPERATURE` - Generation temperature (default: 0.7)
- `LLM_MAX_TOKENS` - Max tokens to generate (default: 512)

### Chunking
- `CHUNK_SIZE` - Chunk size in characters (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 100)
- `MAX_CHUNKS_FOR_CONTEXT` - Chunks to retrieve (default: 5)

## Data Persistence

Docker volumes are used for persistent storage:

- `backend_data` - Uploaded documents and vector database
- `backend_logs` - Application logs
- `ollama_data` - Downloaded Ollama models

Data persists across container restarts. To remove all data:
```bash
docker-compose down -v
```

## Troubleshooting

### Backend won't start
**Problem:** Backend container exits immediately

**Solutions:**
1. Check logs: `docker-compose logs backend`
2. Verify Ollama is running: `docker-compose ps`
3. Ensure port 8000 is not in use: `netstat -an | grep 8000`

### Ollama connection error
**Problem:** "LLM service is not accessible"

**Solutions:**
1. Verify Ollama is healthy: `docker-compose ps`
2. Check Ollama logs: `docker-compose logs ollama`
3. Ensure you've pulled a model: `docker exec -it document-qa-ollama ollama list`
4. Pull a model if needed: `docker exec -it document-qa-ollama ollama pull llama2`

### Model not found
**Problem:** "Model 'llama2' not found"

**Solution:**
```bash
# Pull the model
docker exec -it document-qa-ollama ollama pull llama2

# Verify it's available
docker exec -it document-qa-ollama ollama list
```

### Slow response times
**Problem:** Questions take a long time to answer

**Solutions:**
1. Use a smaller model (phi instead of llama2)
2. Reduce `LLM_MAX_TOKENS` in docker-compose.yml
3. Reduce `MAX_CHUNKS_FOR_CONTEXT` to retrieve fewer chunks
4. Allocate more RAM to Docker (Docker Desktop settings)

### Out of memory
**Problem:** Container crashes or system becomes unresponsive

**Solutions:**
1. Use a smaller model: `docker exec -it document-qa-ollama ollama pull phi`
2. Update `OLLAMA_MODEL=phi` in docker-compose.yml
3. Increase Docker memory limit (Docker Desktop settings)
4. Close other applications to free up RAM

### Permission errors
**Problem:** "Permission denied" when accessing volumes

**Solution:**
```bash
# Stop services
docker-compose down

# Remove volumes and restart
docker-compose down -v
docker-compose up -d
```

### Port already in use
**Problem:** "Port 8000 is already allocated"

**Solutions:**
1. Stop the conflicting service
2. Or change the port in docker-compose.yml:
   ```yaml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

### Cannot access Swagger UI
**Problem:** Browser shows "Connection refused" at localhost:8000

**Solutions:**
1. Verify backend is running: `docker-compose ps`
2. Check backend health: `curl http://localhost:8000/api/health`
3. View backend logs: `docker-compose logs backend`
4. Ensure you're using the correct URL: http://localhost:8000/docs

## Development Setup

For local development without Docker, see [backend/README.md](backend/README.md).

## Architecture

The system uses a RAG (Retrieval Augmented Generation) architecture:

1. **Document Processing**: PDFs are parsed, chunked, and embedded
2. **Vector Storage**: Embeddings stored in ChromaDB for similarity search
3. **Retrieval**: Questions are embedded and matched against document chunks
4. **Generation**: Retrieved context is sent to Ollama LLM for answer generation

## Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **LLM**: Ollama (Llama 2, Mistral, Phi, etc.)
- **Containerization**: Docker & Docker Compose

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space for models

## License

MIT

## Contributing

This is a learning project for understanding RAG architecture, embeddings, and local LLM deployment. Feel free to explore and modify!
