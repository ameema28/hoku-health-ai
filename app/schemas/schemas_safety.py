"""
Hoku Health Care - Safety Pydantic Schemas (Day 7).

Request/response models for safety log operations and safety check results.
All schemas use Pydantic v2 ConfigDict.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SafetyLogCreate(BaseModel):
    """
    Schema for creating a new safety log entry.

    Attributes:
        user_id: Optional user ID associated with the event.
        message: The user message that triggered the safety event.
        ai_response: The AI response that was audited.
        violation_type: Category of safety event.
        severity: Impact level ("high", "moderate", "low").
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: Optional[int] = Field(
        None,
        description="User ID associated with the safety event.",
    )
    message: str = Field(
        ...,
        description="The user message that triggered the safety event.",
    )
    ai_response: str = Field(
        ...,
        description="The AI response that was audited.",
    )
    violation_type: str = Field(
        ...,
        description="Category of safety event.",
        examples=["diagnosis_attempt", "prescription_advice", "missing_disclaimer", "emergency_triggered"],
    )
    severity: str = Field(
        ...,
        description="Impact level: high, moderate, or low.",
        examples=["high", "moderate", "low"],
    )


class SafetyLogRead(BaseModel):
    """
    Schema for reading a safety log record (includes DB-generated fields).

    Attributes:
        id: Database primary key.
        user_id: Associated user ID.
        message: The user message.
        ai_response: The AI response.
        violation_type: Category of safety event.
        severity: Impact level.
        created_at: ISO timestamp of the event.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Safety log primary key.")
    user_id: Optional[int] = Field(None, description="Associated user ID.")
    message: str = Field(..., description="The user message.")
    ai_response: str = Field(..., description="The AI response.")
    violation_type: str = Field(..., description="Category of safety event.")
    severity: str = Field(..., description="Impact level.")
    created_at: Optional[datetime] = Field(None, description="Event timestamp.")


class SafetyCheckResult(BaseModel):
    """
    Schema representing the outcome of a safety guardrails check.

    Attributes:
        is_safe: Whether the response passed all safety checks.
        violations: List of detected violation types.
        sanitized_response: The cleaned response after safety processing.
        severity: Overall severity of the most critical violation.
    """
    model_config = ConfigDict(from_attributes=True)

    is_safe: bool = Field(
        ...,
        description="True if the response passed all safety checks.",
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of detected violation type strings.",
    )
    sanitized_response: str = Field(
        ...,
        description="The response after safety sanitization.",
    )
    severity: str = Field(
        "low",
        description="Overall severity: high, moderate, or low.",
    )