"""
Hoku Health Care - Doctor Model (Day 6).

SQLAlchemy 2.0 mapped model for the doctors table, storing specialist
information, availability, and experience for patient-facing suggestions.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Index, Integer, String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Doctor(Base):
    """
    Represents a medical specialist in the Hoku Health Care network.

    Attributes:
        id: Primary key.
        user_id: Optional link to the users table (nullable until Backend Lead completes User model).
        specialty: Medical specialty (e.g., "Cardiologist", "Dermatologist").
        experience_years: Years of professional practice.
        is_available: Whether the doctor is currently accepting patients.
        license_number: Optional medical license identifier.
        created_at: UTC timestamp of record creation.
    """

    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    experience_years: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    license_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_doctors_specialty", "specialty"),
        Index("idx_doctors_available", "is_available"),
    )

    def __repr__(self) -> str:
        return (
            f"<Doctor(id={self.id}, specialty='{self.specialty}', "
            f"experience={self.experience_years}, available={self.is_available})>"
        )