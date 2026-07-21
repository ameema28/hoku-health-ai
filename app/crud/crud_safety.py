"""
Hoku Health Care - Safety Log CRUD Operations (Day 7).

Database access layer for safety_logs using SQLAlchemy 2.0 select() syntax.
All operations are atomic and fully logged.
"""

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseOperationException
from app.models.safety_log import SafetyLog

logger = logging.getLogger(__name__)


def log_safety_violation(
    db: Session,
    user_id: Optional[int],
    message: str,
    ai_response: str,
    violation_type: str,
    severity: str,
) -> SafetyLog:
    """
    Atomically create a new safety log entry.

    Args:
        db: SQLAlchemy database session.
        user_id: Optional user ID associated with the event.
        message: The user message that triggered the safety event.
        ai_response: The AI response that was audited.
        violation_type: Category of safety event.
        severity: Impact level ("high", "moderate", "low").

    Returns:
        SafetyLog: The persisted instance with refreshed ID.

    Raises:
        DatabaseOperationException: If the commit fails.
    """
    try:
        entry = SafetyLog(
            user_id=user_id,
            message=message,
            ai_response=ai_response,
            violation_type=violation_type,
            severity=severity,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        logger.info(
            "Created safety_log id=%s for user_id=%s, violation_type='%s', severity='%s'",
            entry.id,
            user_id,
            violation_type,
            severity,
        )
        return entry
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to create safety_log for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to persist safety log"
        ) from exc


def get_safety_logs_by_user(
    db: Session,
    user_id: int,
    limit: int = 50,
    skip: int = 0,
) -> List[SafetyLog]:
    """
    Retrieve paginated safety logs for a specific user.

    Results are ordered by created_at descending (newest first).

    Args:
        db: SQLAlchemy database session.
        user_id: User to filter by.
        limit: Maximum number of records to return.
        skip: Number of records to offset (pagination).

    Returns:
        List[SafetyLog]: Matching safety log entries.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(SafetyLog)
            .where(SafetyLog.user_id == user_id)
            .order_by(SafetyLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d safety log entries for user_id=%s (skip=%d, limit=%d)",
            len(result),
            user_id,
            skip,
            limit,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve safety logs for user_id=%s: %s",
            user_id,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to retrieve safety logs"
        ) from exc


def get_safety_logs_by_type(
    db: Session,
    violation_type: str,
    limit: int = 50,
    skip: int = 0,
) -> List[SafetyLog]:
    """
    Retrieve safety logs filtered by violation type.

    Args:
        db: SQLAlchemy database session.
        violation_type: Violation type to filter by.
        limit: Maximum number of records to return.
        skip: Number of records to offset.

    Returns:
        List[SafetyLog]: Matching safety log entries.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(SafetyLog)
            .where(SafetyLog.violation_type == violation_type)
            .order_by(SafetyLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d safety logs for violation_type='%s'",
            len(result),
            violation_type,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve safety logs for type='%s': %s",
            violation_type,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to retrieve safety logs by type"
        ) from exc


def get_safety_logs_by_severity(
    db: Session,
    severity: str,
    limit: int = 50,
    skip: int = 0,
) -> List[SafetyLog]:
    """
    Retrieve safety logs filtered by severity level.

    Args:
        db: SQLAlchemy database session.
        severity: Severity level to filter by ("high", "moderate", "low").
        limit: Maximum number of records to return.
        skip: Number of records to offset.

    Returns:
        List[SafetyLog]: Matching safety log entries.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(SafetyLog)
            .where(SafetyLog.severity == severity)
            .order_by(SafetyLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d safety logs for severity='%s'",
            len(result),
            severity,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve safety logs for severity='%s': %s",
            severity,
            exc,
        )
        raise DatabaseOperationException(
            "Failed to retrieve safety logs by severity"
        ) from exc