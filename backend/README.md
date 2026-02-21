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
├── pyproject.toml          # Project metadata, dependencies & tool configuration
├── .env.example           # Environment variable templates
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install core dependencies:
```bash
pip install .
```

3. Install development & testing dependencies:
```bash
pip install ".[dev]"
```

> **Windows Users:** If you encounter build errors, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for alternative installation methods.

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run tests:
```bash
pytest
```

## Development

- All application code goes in `app/`
- All tests go in `tests/`
- Use pytest for testing with hypothesis for property-based tests
- Pytest and Hypothesis are configured in `pyproject.toml` under `[tool.pytest.ini_options]` and `[tool.hypothesis]`
- Follow the implementation plan in `.kiro/specs/document-qa-system/tasks.md`

## API Documentation

Once the backend is running, access Swagger UI at:
- http://localhost:8000/docs

## Requirements

- Python 3.10–3.13 (Python 3.14 has compatibility issues with ChromaDB)
- Ollama (for LLM inference)
- 4GB+ RAM recommended

## Known Issues

### Python 3.14 Compatibility
ChromaDB currently has compatibility issues with Python 3.14 due to its dependency on Pydantic v1. If you're using Python 3.14, you may encounter errors when running tests.

**Workaround:** Use Python 3.10, 3.11, 3.12, or 3.13 for development and testing until ChromaDB releases a Python 3.14-compatible version.

The vector store implementation itself is correct and will work properly once the ChromaDB compatibility issue is resolved.
