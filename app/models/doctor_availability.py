"""
Hoku Health Care - Doctor Availability Model (Day 6).

SQLAlchemy 2.0 mapped model for doctor weekly availability slots.
Stores day_of_week (0=Monday), start/end times as TEXT for SQLite
compatibility, and booking status.
"""

from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DoctorAvailability(Base):
    """
    Represents a single weekly time slot for a doctor.

    Attributes:
        id: Primary key.
        doctor_id: Foreign key to doctors.id.
        day_of_week: 0=Monday, 1=Tuesday, ..., 6=Sunday.
        start_time: Slot start time (TEXT, e.g., "09:00").
        end_time: Slot end time (TEXT, e.g., "17:00").
        is_booked: Whether this slot is already reserved.
    """

    __tablename__ = "doctor_availability"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("doctors.id", ondelete="CASCADE"),
        nullable=False,
    )
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[str] = mapped_column(String(10), nullable=False)
    end_time: Mapped[str] = mapped_column(String(10), nullable=False)
    is_booked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<DoctorAvailability(id={self.id}, doctor_id={self.doctor_id}, "
            f"day={self.day_of_week}, {self.start_time}-{self.end_time}, "
            f"booked={self.is_booked})>"
        )