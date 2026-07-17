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


# Day 4: Intent-specific fixtures
@pytest.fixture
def mock_intent_classifier():
    """Mock IntentClassifier that returns predictable intents."""
    from unittest.mock import MagicMock
    from app.ai.intent_classifier import IntentEnum
    
    classifier = MagicMock()
    
    async def mock_classify(message):
        msg_lower = message.lower()
        if "headache" in msg_lower or "fever" in msg_lower:
            return (IntentEnum.SYMPTOM, 0.95)
        elif "book" in msg_lower or "appointment" in msg_lower:
            return (IntentEnum.BOOKING, 0.92)
        elif "medicine" in msg_lower or "paracetamol" in msg_lower:
            return (IntentEnum.MEDICATION, 0.93)
        elif "emergency" in msg_lower or "chest pain" in msg_lower or "breathe" in msg_lower:
            return (IntentEnum.EMERGENCY, 0.99)
        else:
            return (IntentEnum.GENERAL, 0.88)
    
    classifier.classify_intent = mock_classify
    return classifier


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