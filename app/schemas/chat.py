"""
Hoku Health Care - Chat Pydantic Schemas (Day 4).

Request/response models for the AI chatbot API, chat history
persistence, and session retrieval. All schemas use Pydantic v2
ConfigDict for ORM mode.

Day 4 additions:
- IntentEnum for type-safe intent classification
- intent and confidence fields in ChatMessageResponse (optional for
  backward compatibility)
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IntentEnum(str):
    """
    Type hint helper for intent string values.
    Not a formal Enum to maintain JSON serialization simplicity,
    but provides documentation value.
    """
    SYMPTOM = "symptom"
    BOOKING = "booking"
    MEDICATION = "medication"
    GENERAL = "general"
    EMERGENCY = "emergency"


class ChatMessageRequest(BaseModel):
    """
    Incoming chat message from a patient or doctor.

    Attributes:
        message: The user's health-related question or symptom description.
        userId: The authenticated user's database ID.
    """
    model_config = ConfigDict(from_attributes=True)

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

    Day 4 additions:
    - intent: Classified intent of the user's message
    - confidence: Confidence score of the intent classification

    Attributes:
        reply: The AI's conversational response.
        suggestedSpecialist: Recommended specialist type, if applicable.
        severity: Assessed severity level (mild, moderate, severe).
        shouldSeeDoctor: Whether the user should seek professional care.
        intent: Classified intent category (Day 4).
        confidence: Intent classification confidence score (Day 4).
    """
    model_config = ConfigDict(from_attributes=True)

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
    # Day 4: Optional for backward compatibility with existing clients
    intent: Optional[str] = Field(
        None,
        description="Classified intent: symptom, booking, medication, general, or emergency.",
    )
    confidence: Optional[float] = Field(
        None,
        description="Intent classification confidence score (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )


class ChatHistoryCreate(BaseModel):
    """
    Internal schema for inserting a new chat history record.
    Not exposed directly via API; used by the service and CRUD layers.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    message: str
    ai_response: Optional[str] = None
    intent: Optional[str] = None


class ChatHistoryRead(BaseModel):
    """
    Schema for retrieving a single chat history entry.

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


class ChatHistoryItem(BaseModel):
    """
    Single message within a chat session, normalized for frontend display.

    Attributes:
        role: Sender identifier ('human' or 'ai').
        content: Message text.
        timestamp: When the message was recorded.
    """
    model_config = ConfigDict(from_attributes=True)

    role: Literal["human", "ai"] = Field(
        ...,
        description="Message sender role.",
    )
    content: str = Field(..., description="Message text.")
    timestamp: datetime = Field(..., description="Message timestamp.")


class ChatSessionResponse(BaseModel):
    """
    Complete chat session payload for the authenticated user.

    Attributes:
        user_id: The user's database ID.
        messages: Chronologically ordered list of chat messages.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: int = Field(..., description="User ID.")
    messages: List[ChatHistoryItem] = Field(
        default_factory=list,
        description="Chat messages in chronological order.",
    )