"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return "This is a sample text for testing purposes. " * 100
