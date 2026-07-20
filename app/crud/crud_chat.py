"""
Hoku Health Care - Chat History CRUD Operations.

Database access layer for chat_history using SQLAlchemy 2.0 select() syntax.
All operations are atomic (commit/rollback handled) and fully logged.
"""

import logging
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseOperationException
from app.models.models_chat import ChatHistory

logger = logging.getLogger(__name__)


def create_chat_history(
    db: Session,
    user_id: int,
    message: str,
    ai_response: Optional[str],
    intent: Optional[str],
) -> ChatHistory:
    """
    Atomically create a new chat history entry.

    Args:
        db: SQLAlchemy database session.
        user_id: Foreign key to the users table.
        message: Sanitized user message.
        ai_response: AI-generated response text.
        intent: Classified intent string.

    Returns:
        ChatHistory: The persisted instance with refreshed ID.

    Raises:
        DatabaseOperationException: If the commit fails.
    """
    try:
        entry = ChatHistory(
            user_id=user_id,
            message=message,
            ai_response=ai_response,
            intent=intent,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        logger.info(
            "Created chat_history id=%s for user_id=%s",
            entry.id,
            user_id,
        )
        return entry
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to create chat_history for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to persist chat history"
        ) from exc


def get_chat_history_by_user(
    db: Session,
    user_id: int,
    limit: int = 50,
    skip: int = 0,
) -> List[ChatHistory]:
    """
    Retrieve paginated chat history for a specific user.

    Results are ordered by created_at descending (newest first) to support
    typical "latest conversations" UI patterns.

    Args:
        db: SQLAlchemy database session.
        user_id: User to filter by.
        limit: Maximum number of records to return.
        skip: Number of records to offset (pagination).

    Returns:
        List[ChatHistory]: Matching history entries.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d chat history entries for user_id=%s (skip=%d, limit=%d)",
            len(result),
            user_id,
            skip,
            limit,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve chat history for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to retrieve chat history"
        ) from exc


def get_chat_history_count(db: Session, user_id: int) -> int:
    """
    Count total chat history entries for a user.

    Args:
        db: SQLAlchemy database session.
        user_id: User to filter by.

    Returns:
        int: Total number of chat history rows.

    Raises:
        DatabaseOperationException: If the count query fails.
    """
    try:
        stmt = select(func.count(ChatHistory.id)).where(
            ChatHistory.user_id == user_id
        )
        count = db.execute(stmt).scalar_one()
        logger.info(
            "Chat history count for user_id=%s: %d",
            user_id,
            count,
        )
        return count
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to count chat history for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to count chat history"
        ) from exc


def get_recent_chat_history(
    db: Session,
    user_id: int,
    limit: int = 10,
) -> List[ChatHistory]:
    """
    Get the most recent COMPLETE chat history entries for a user.

    Only returns turns where ai_response is not NULL, ensuring the
    memory loader receives valid conversation pairs.

    Args:
        db: SQLAlchemy database session.
        user_id: User to filter by.
        limit: Maximum number of recent records.

    Returns:
        List[ChatHistory]: Recent complete entries ordered by created_at descending.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .where(ChatHistory.ai_response.isnot(None))
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d recent complete chat entries for user_id=%s",
            len(result),
            user_id,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve recent chat for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to retrieve recent chat history"
        ) from exc


def user_exists(db: Session, user_id: int) -> bool:
    """
    Verify that a user exists in the database.

    .. note::
        This is a placeholder implementation. The Backend Lead (Talha) owns
        the User model. Once available, replace the body with a proper query
        against the users table.

    Args:
        db: SQLAlchemy database session.
        user_id: User ID to verify.

    Returns:
        bool: True if the user exists.
    """
    # TODO: Replace with actual User model query when Backend completes it.
    return True