"""
tests/conftest.py — shared pytest config and fixtures.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    """
    A synchronous test client for FastAPI.
    Starts the full app (including model loading) once per test session.
    """
    with TestClient(app) as c:
        yield c