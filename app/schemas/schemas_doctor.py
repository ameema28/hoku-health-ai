"""
Hoku Health Care - Doctor Pydantic Schemas (Day 6).

Request/response models for doctor data and specialist suggestions.
All schemas use Pydantic v2 ConfigDict.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DoctorBase(BaseModel):
    """Base schema with shared doctor fields."""
    model_config = ConfigDict(from_attributes=True)

    specialty: str = Field(..., description="Medical specialty name.")
    experience_years: int = Field(0, ge=0, description="Years of professional experience.")
    is_available: bool = Field(True, description="Whether the doctor is currently accepting patients.")
    license_number: Optional[str] = Field(None, description="Medical license number.")


class DoctorRead(DoctorBase):
    """Schema for reading a doctor record (includes DB-generated fields)."""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Doctor primary key.")
    user_id: Optional[int] = Field(None, description="Linked user ID if applicable.")
    created_at: datetime = Field(..., description="Record creation timestamp.")


class DoctorAvailability(BaseModel):
    """Schema for a single weekly availability slot."""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Slot primary key.")
    doctor_id: int = Field(..., description="Linked doctor ID.")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday through 6=Sunday.")
    start_time: str = Field(..., description="Slot start time (e.g., '09:00').")
    end_time: str = Field(..., description="Slot end time (e.g., '17:00').")
    is_booked: bool = Field(False, description="Whether the slot is reserved.")


class DoctorSuggestion(BaseModel):
    """
    Schema for a specialist suggestion embedded in a chat response.

    Contains the mapped specialty, top doctor details, and availability.
    """
    model_config = ConfigDict(from_attributes=True)

    specialist: str = Field(..., description="Mapped medical specialty.")
    doctor_name: Optional[str] = Field(None, description="Suggested doctor's display name.")
    experience: Optional[int] = Field(None, ge=0, description="Doctor's years of experience.")
    availability: Optional[List[DoctorAvailability]] = Field(
        None,
        description="Upcoming available slots for the suggested doctor.",
    )
    doctor_id: Optional[int] = Field(None, description="Suggested doctor's primary key.")