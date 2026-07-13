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
    2. Escape HTML entities to prevent XSS injection.
    3. Collapse excessive whitespace into single spaces.
    4. Truncate to MAX_MESSAGE_LENGTH.

    Args:
        message: Raw user input string.

    Returns:
        str: Cleaned, safe message string.
    """
    if not isinstance(message, str):
        logger.warning(
            "Non-string message received: %s",
            type(message),
        )
        return ""

    # Strip whitespace
    cleaned = message.strip()

    # Escape HTML to prevent injection
    cleaned = html.escape(cleaned)

    # Collapse multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Truncate to maximum allowed length
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        logger.info(
            "Message truncated from %d to %d chars",
            len(cleaned),
            MAX_MESSAGE_LENGTH,
        )
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
        is_valid = isinstance(user_id, int) and user_id > 0
        if not is_valid:
            logger.warning(
                "Invalid user_id received: %s",
                user_id,
            )
        return is_valid
    except Exception:
        return False


def validate_message_length(message: str) -> bool:
    """
    Validate that a message is within acceptable length bounds.

    Args:
        message: The message string to validate.

    Returns:
        bool: True if length is valid (1 to MAX_MESSAGE_LENGTH), False otherwise.
    """
    if not isinstance(message, str):
        logger.warning(
            "Non-string message in length validation: %s",
            type(message),
        )
        return False

    length = len(message.strip())
    is_valid = 0 < length <= MAX_MESSAGE_LENGTH
    if not is_valid:
        logger.warning(
            "Message length validation failed: length=%d (max=%d)",
            length,
            MAX_MESSAGE_LENGTH,
        )
    return is_valid
