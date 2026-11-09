"""
Hoku Health Care - Input Validators.

Shared validation utilities for sanitizing user input and ensuring
data integrity before processing by AI or persistence layers.
"""

import logging
import re

from app.utils.constants import MAX_MESSAGE_LENGTH

logger = logging.getLogger(__name__)


#: Control characters except tab/newline/carriage return (those become spaces).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_message(message: str) -> str:
    """
    Normalise a user message for the chat pipeline.

    Steps: drop control characters, collapse whitespace runs into single
    spaces, strip the ends. The text is NOT HTML-escaped: it is plain text
    for the model, memory and history, and every client must render it as
    text (the demo page uses ``textContent``). It is not truncated either;
    ``validate_message_length`` rejects text over MAX_MESSAGE_LENGTH.

    Args:
        message: Raw user input string.

    Returns:
        str: The normalised message ("" for non-string input).
    """
    if not isinstance(message, str):
        logger.warning(
            "Non-string message received: %s",
            type(message),
        )
        return ""

    cleaned = _CONTROL_CHARS.sub("", message)
    return re.sub(r"\s+", " ", cleaned).strip()


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
