"""
Hoku Health Care - Chat Pydantic Schemas.

Request/response models for the AI chatbot API and chat history
persistence layer. Ensures runtime validation and OpenAPI documentation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ChatMessageRequest(BaseModel):
    """
    Incoming chat message from a patient or doctor.

    Attributes:
        message: The user's health-related question or symptom description.
        userId: The authenticated user's database ID.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's health question or symptom description.",
        examples=["I have a headache and fever for 3 days"],
    )
    userId: int = Field(
        ...,
        gt=0,
        description="Authenticated user's database ID.",
        examples=[123],
    )


class ChatMessageResponse(BaseModel):
    """
    AI chatbot response with optional clinical guidance metadata.

    Attributes:
        reply: The AI's conversational response.
        suggestedSpecialist: Recommended specialist type, if applicable.
        severity: Assessed severity level (mild, moderate, severe).
        shouldSeeDoctor: Whether the user should seek professional care.
    """

    reply: str = Field(..., description="AI-generated conversational response.")
    suggestedSpecialist: Optional[str] = Field(
        None,
        description="Recommended medical specialist based on the query.",
    )
    severity: str = Field(
        "mild",
        description="Assessed severity level: mild, moderate, or severe.",
    )
    shouldSeeDoctor: bool = Field(
        False,
        description="Flag indicating if professional consultation is advised.",
    )


class ChatHistoryCreate(BaseModel):
    """
    Internal schema for inserting a new chat history record.

    Not exposed directly via API; used by the service layer.
    """

    user_id: int
    message: str
    ai_response: Optional[str] = None
    intent: Optional[str] = None


class ChatHistoryRead(BaseModel):
    """
    Schema for retrieving chat history entries.

    Attributes:
        id: Database primary key.
        user_id: Associated user ID.
        message: Original user message.
        ai_response: AI response text.
        intent: Detected intent.
        created_at: ISO timestamp of the conversation turn.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    message: str
    ai_response: Optional[str] = None
    intent: Optional[str] = None
    created_at: Optional[datetime] = None
