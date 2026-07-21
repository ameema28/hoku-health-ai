#!/usr/bin/env python3
"""
Hoku Health Care - Database Initialization Script (Day 7).

Run this once after deleting the SQLite database to recreate all tables.
Not needed in production (Alembic handles migrations there).

Day 7 update: Registers SafetyLog model with Base.metadata.
"""

from app.core.database import Base, engine

# Import all models so they register with Base.metadata
from app.models.models_chat import ChatHistory  # noqa: F401
from app.models.vector_store import VectorStore  # noqa: F401  (Day 5)
from app.models.models_doctor import Doctor  # noqa: F401  (Day 6)
from app.models.doctor_availability import DoctorAvailability  # noqa: F401  (Day 6)
from app.models.safety_log import SafetyLog  # noqa: F401  (Day 7)

from sqlalchemy.orm import Session
from app.core.database import SessionLocal


def init_database() -> None:
    """Create all SQLAlchemy tables in the configured database."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def seed_sample_doctors(db: Session) -> None:
    """
    Populate the doctors table with 5 sample specialists and availability slots.

    Safe to run multiple times — uses add_all with commit/rollback.
    """
    from app.models.models_doctor import Doctor
    from app.models.doctor_availability import DoctorAvailability

    sample_doctors = [
        Doctor(
            user_id=None,
            specialty="General Physician",
            experience_years=10,
            is_available=True,
            license_number="GP-PAK-2024-001",
        ),
        Doctor(
            user_id=None,
            specialty="Cardiologist",
            experience_years=15,
            is_available=True,
            license_number="CD-PAK-2024-002",
        ),
        Doctor(
            user_id=None,
            specialty="Dermatologist",
            experience_years=8,
            is_available=True,
            license_number="DER-PAK-2024-003",
        ),
        Doctor(
            user_id=None,
            specialty="Child Specialist",
            experience_years=12,
            is_available=True,
            license_number="PED-PAK-2024-004",
        ),
        Doctor(
            user_id=None,
            specialty="Psychiatrist",
            experience_years=20,
            is_available=True,
            license_number="PSY-PAK-2024-005",
        ),
    ]

    db.add_all(sample_doctors)
    db.commit()

    # Refresh to get generated IDs
    for doc in sample_doctors:
        db.refresh(doc)

    # Add availability slots for each doctor
    availability_slots = []
    for doc in sample_doctors:
        availability_slots.extend([
            DoctorAvailability(
                doctor_id=doc.id,
                day_of_week=0,  # Monday
                start_time="09:00",
                end_time="12:00",
                is_booked=False,
            ),
            DoctorAvailability(
                doctor_id=doc.id,
                day_of_week=0,
                start_time="14:00",
                end_time="17:00",
                is_booked=False,
            ),
            DoctorAvailability(
                doctor_id=doc.id,
                day_of_week=2,  # Wednesday
                start_time="10:00",
                end_time="13:00",
                is_booked=False,
            ),
            DoctorAvailability(
                doctor_id=doc.id,
                day_of_week=4,  # Friday
                start_time="09:00",
                end_time="15:00",
                is_booked=False,
            ),
        ])

    db.add_all(availability_slots)
    db.commit()

    print(f"Seeded {len(sample_doctors)} doctors with {len(availability_slots)} availability slots.")


if __name__ == "__main__":
    init_database()

    # Optional: seed sample doctors for local development
    db = SessionLocal()
    try:
        seed_sample_doctors(db)
    finally:
        db.close()