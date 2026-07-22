"""
Hoku Health Care - Static Fallback Responses (Day 8).

Pre-written, zero-latency fallback responses returned instantly (< 1ms)
when the LLM or any pipeline stage times out. These responses maintain
clinical safety standards even under degraded performance conditions.

Design rationale:
- Static strings require zero computation — no LLM, no DB, no parsing
- Every fallback includes the mandatory clinical disclaimer
- Emergency fallback is more urgent than general fallback
- Booking fallback directs users to the self-service portal

Saved execution time: ~2000-3000ms by skipping LLM generation entirely
when the pipeline is under time pressure.
"""

# ------------------------------------------------------------------
# GENERAL FALLBACK
# ------------------------------------------------------------------
# Returned when the main LLM times out or fails for non-emergency queries.
# Emphasizes that the system is experiencing issues while maintaining
# the mandatory clinical safety disclaimer.
FALLBACK_GENERAL: str = (
    "I'm sorry, I'm having trouble processing your request in time. "
    "Please consult a doctor for proper diagnosis."
)

# ------------------------------------------------------------------
# EMERGENCY FALLBACK
# ------------------------------------------------------------------
# Returned ONLY if emergency detection itself somehow fails or times out
# (extremely unlikely — Tier 1 regex is < 1ms). This is a last-resort
# safety net that still directs the user to emergency services.
FALLBACK_EMERGENCY: str = (
    "This may be serious. Please contact emergency services immediately "
    "or visit the nearest ER. Please consult a doctor for proper diagnosis."
)

# ------------------------------------------------------------------
# BOOKING FALLBACK
# ------------------------------------------------------------------
# Returned when the pipeline times out during a booking-intent query.
# Directs the user to the self-service dashboard rather than leaving
# them without guidance.
FALLBACK_BOOKING: str = (
    "You can book an appointment directly through the patient dashboard. "
    "Please consult a doctor for proper diagnosis."
)

# ------------------------------------------------------------------
# RAG TIMEOUT FALLBACK
# ------------------------------------------------------------------
# Returned when RAG retrieval times out but the LLM is still available.
# The chatbot proceeds with general knowledge instead of grounded FAQs.
FALLBACK_RAG_TIMEOUT: str = (
    "I'm looking into your question. In the meantime, please consult a "
    "doctor for proper diagnosis."
)

# ------------------------------------------------------------------
# SAFETY 3-STRIKE FALLBACK
# ------------------------------------------------------------------
# Re-exported from config for consistency. This is the absolute last
# resort when even safety sanitization fails repeatedly.
from app.ai.config import ai_settings

FALLBACK_SAFETY_3_STRIKE: str = ai_settings.SAFETY_FALLBACK_RESPONSE


def get_fallback_for_intent(intent: str) -> str:
    """
    Select the most appropriate fallback response based on intent.

    Args:
        intent: Classified intent string (e.g., "booking", "emergency").

    Returns:
        str: The matching fallback response, defaulting to FALLBACK_GENERAL.
    """
    intent_lower = (intent or "").lower()

    if intent_lower == "emergency":
        return FALLBACK_EMERGENCY
    if intent_lower == "booking":
        return FALLBACK_BOOKING

    return FALLBACK_GENERAL