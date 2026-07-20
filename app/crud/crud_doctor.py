"""
Hoku Health Care - Doctor CRUD Operations (Day 6).

Database access layer for doctors and doctor_availability tables.
All operations are atomic and fully logged.
"""

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseOperationException
from app.models.models_doctor import Doctor
from app.models.doctor_availability import DoctorAvailability

logger = logging.getLogger(__name__)


def get_doctors_by_specialty(db: Session, specialty: str) -> List[Doctor]:
    """
    Retrieve available doctors filtered by specialty, ordered by
    experience descending.

    Args:
        db: SQLAlchemy database session.
        specialty: Medical specialty name (case-insensitive match).

    Returns:
        List[Doctor]: Available doctors for the specialty.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = (
            select(Doctor)
            .where(Doctor.specialty.ilike(specialty))
            .where(Doctor.is_available == True)  # noqa: E712
            .order_by(Doctor.experience_years.desc())
        )
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d available doctors for specialty='%s'",
            len(result),
            specialty,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error("Failed to retrieve doctors for specialty='%s': %s", specialty, exc)
        raise DatabaseOperationException("Failed to retrieve doctors") from exc


def get_doctor_by_id(db: Session, doctor_id: int) -> Optional[Doctor]:
    """
    Retrieve a single doctor by primary key.

    Args:
        db: SQLAlchemy database session.
        doctor_id: Doctor primary key.

    Returns:
        Doctor | None: The doctor instance if found.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = select(Doctor).where(Doctor.id == doctor_id)
        result = db.execute(stmt).scalar_one_or_none()
        if result:
            logger.info("Retrieved doctor id=%d, specialty='%s'", result.id, result.specialty)
        else:
            logger.warning("Doctor id=%d not found", doctor_id)
        return result
    except Exception as exc:
        db.rollback()
        logger.error("Failed to retrieve doctor id=%d: %s", doctor_id, exc)
        raise DatabaseOperationException("Failed to retrieve doctor") from exc


def get_doctor_availability(
    db: Session,
    doctor_id: int,
    include_booked: bool = False,
) -> List[DoctorAvailability]:
    """
    Retrieve availability slots for a specific doctor.

    Args:
        db: SQLAlchemy database session.
        doctor_id: Doctor primary key.
        include_booked: If False, only returns unbooked slots.

    Returns:
        List[DoctorAvailability]: Time slots ordered by day_of_week, start_time.

    Raises:
        DatabaseOperationException: If the query fails.
    """
    try:
        stmt = select(DoctorAvailability).where(DoctorAvailability.doctor_id == doctor_id)
        if not include_booked:
            stmt = stmt.where(DoctorAvailability.is_booked == False)  # noqa: E712

        stmt = stmt.order_by(DoctorAvailability.day_of_week, DoctorAvailability.start_time)
        result = db.execute(stmt).scalars().all()
        logger.info(
            "Retrieved %d availability slots for doctor_id=%d (include_booked=%s)",
            len(result),
            doctor_id,
            include_booked,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to retrieve availability for doctor_id=%d: %s",
            doctor_id,
            exc,
        )
        raise DatabaseOperationException("Failed to retrieve doctor availability") from exc