# Document Q&A System - Backend

A locally-hosted backend application that enables users to upload PDF documents and ask natural language questions about their content via REST API using Retrieval Augmented Generation (RAG).

## Project Structure

```
backend/
├── app/                    # Application source code
│   └── __init__.py
├── tests/                  # Test suite
│   ├── __init__.py
│   └── conftest.py        # Pytest fixtures and configuration
├── data/                   # Data storage (created at runtime)
│   ├── documents/         # Uploaded PDF files
│   ├── chroma_db/         # ChromaDB persistent storage
│   └── metadata.db        # SQLite metadata database
├── logs/                   # Application logs (created at runtime)
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
├── .env.example           # Environment variable templates
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run tests:
```bash
pytest
```

## Development

- All application code goes in `app/`
- All tests go in `tests/`
- Use pytest for testing with hypothesis for property-based tests
- Follow the implementation plan in `.kiro/specs/document-qa-system/tasks.md`

## API Documentation

Once the backend is running, access Swagger UI at:
- http://localhost:8000/docs

## Requirements

- Python 3.10+
- Ollama (for LLM inference)
- 4GB+ RAM recommended
