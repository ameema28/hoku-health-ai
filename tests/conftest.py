"""Shared pytest fixtures for the test suite."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app


# Use SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Create a TestClient with overridden DB dependency."""
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def mock_emergency_detector():
    """Mock emergency detector for controlled testing."""
    from unittest.mock import patch
    
    def mock_detect(message):
        if message and "emergency" in message.lower():
            return True
        return False
    
    with patch("app.ai.chatbot.detect_emergency", side_effect=mock_detect):
        yield

@pytest.fixture(autouse=True)
def _isolate_chatbot_cache():
    """Prevent ResponseCache singleton from leaking between tests."""
    from unittest.mock import patch
    with patch("app.ai.chatbot.ResponseCache") as MockCache:
        instance = MockCache.return_value
        instance.get.return_value = None
        instance.set.return_value = None
        yield

@pytest.fixture
def auth_headers(client):
    """
    Authenticated headers for endpoints that require JWT.
    Overrides get_current_user so any Bearer token works in tests.
    """
    from app.core.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"id": 1, "email": "test@hoku.health"}
    return {"Authorization": "Bearer test-token"}       