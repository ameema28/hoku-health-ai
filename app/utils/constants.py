"""
Hoku Health Care - Application Constants.

Centralized constants to avoid magic strings and ensure consistency
across the AI chatbot and related services.
"""

# Application Identity
APP_NAME: str = "Hoku Health Care"

# Input Constraints
MAX_MESSAGE_LENGTH: int = 1000

# Performance Constraints (NFR-02)
AI_RESPONSE_TIMEOUT: int = 4  # seconds

# Clinical Safety
SAFETY_DISCLAIMER: str = "Please consult a doctor for proper diagnosis."

# Groq Model Selection Strategy
# Fast model for intent classification and entity extraction (low latency)
GROQ_FAST_MODEL: str = "llama3-8b-8192"
# Main model for high-quality patient-facing responses
GROQ_MAIN_MODEL: str = "llama3-70b-8192"
