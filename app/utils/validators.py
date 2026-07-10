"""
Hoku Health Care - Input Validators.

Shared validation utilities for sanitizing user input and ensuring
data integrity before processing by AI or persistence layers.
"""

import html
import logging
import re

from app.utils.constants import MAX_MESSAGE_LENGTH

logger = logging.getLogger(__name__)


def sanitize_message(message: str) -> str:
    """
    Sanitize a user message for safe processing.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Escape HTML entities to prevent XSS.
    3. Collapse excessive whitespace.
    4. Truncate to MAX_MESSAGE_LENGTH.

    Args:
        message: Raw user input string.

    Returns:
        str: Cleaned, safe message string.
    """
    if not isinstance(message, str):
        logger.warning("Non-string message received: %s", type(message))
        return ""

    # Strip whitespace
    cleaned = message.strip()

    # Escape HTML to prevent injection
    cleaned = html.escape(cleaned)

    # Collapse multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Truncate to maximum allowed length
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        logger.info("Message truncated from %d to %d chars", len(cleaned), MAX_MESSAGE_LENGTH)
        cleaned = cleaned[:MAX_MESSAGE_LENGTH]

    return cleaned


def validate_user_id(user_id: int) -> bool:
    """
    Validate that a user ID is a positive integer.

    Args:
        user_id: The user ID to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        return isinstance(user_id, int) and user_id > 0
    except Exception:
        return False
