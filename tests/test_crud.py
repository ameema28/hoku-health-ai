#!/usr/bin/env python3
"""
Hoku Health Care - CRUD Verification Script.

Quick local test to verify chat_history CRUD operations without a full
FastAPI server. Uses SQLite for portability.

Usage:
    python test_crud.py
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.crud import (
    create_chat_history,
    get_chat_history_by_user,
    get_chat_history_count,
    get_recent_chat_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEST_DB_URL = "sqlite:///./test_chat_history.db"


def main() -> None:
    """Run CRUD verification tests."""
    engine = create_engine(TEST_DB_URL, echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        logger.info("=== Test 1: Create chat history ===")
        entry = create_chat_history(
            db=db,
            user_id=1,
            message="I have a headache and fever",
            ai_response=(
                "I understand you have a headache and fever. "
                "Please consult a doctor for proper diagnosis."
            ),
            intent="symptom",
        )
        logger.info("Created entry: %s", entry.to_dict())

        logger.info("=== Test 2: Create second entry ===")
        entry2 = create_chat_history(
            db=db,
            user_id=1,
            message="Thank you",
            ai_response="You're welcome! Take care.",
            intent="general_health",
        )
        logger.info("Created entry: %s", entry2.to_dict())

        logger.info("=== Test 3: Get history by user (paginated) ===")
        history = get_chat_history_by_user(db, user_id=1, limit=10, skip=0)
        logger.info("Retrieved %d entries", len(history))
        for h in history:
            logger.info("  - %s", h.to_dict())

        logger.info("=== Test 4: Count history ===")
        count = get_chat_history_count(db, user_id=1)
        logger.info("Total count: %d", count)

        logger.info("=== Test 5: Recent history ===")
        recent = get_recent_chat_history(db, user_id=1, limit=5)
        logger.info("Recent entries: %d", len(recent))

        logger.info("=== Test 6: Empty user ===")
        empty = get_chat_history_by_user(db, user_id=999, limit=10, skip=0)
        logger.info("Entries for user 999: %d", len(empty))

        logger.info("=== All CRUD tests passed ===")

    except Exception as exc:
        logger.error("Test failed: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
