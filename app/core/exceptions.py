"""
Hoku Health Care - Custom Exceptions.

Domain-specific exceptions for the AI chatbot module.
Provides structured error types that map cleanly to HTTP status codes.
"""

from fastapi import HTTPException, status


class ChatbotException(Exception):
    """
    Base exception for chatbot-related business logic errors.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str = "Chatbot error occurred") -> None:
        """
        Initialize with a descriptive error message.

        Args:
            message: Explanation of what went wrong.
        """
        self.message = message
        super().__init__(self.message)


class UserNotFoundException(HTTPException):
    """
    Raised when a referenced user does not exist in the database.

    Maps to HTTP 404 Not Found.
    """

    def __init__(self, detail: str = "User not found") -> None:
        """
        Initialize with 404 status.

        Args:
            detail: Error message returned to the client.
        """
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class DatabaseOperationException(HTTPException):
    """
    Raised when a database transaction fails unexpectedly.

    Maps to HTTP 500 Internal Server Error.
    """

    def __init__(self, detail: str = "Database operation failed") -> None:
        """
        Initialize with 500 status.

        Args:
            detail: Error message returned to the client.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )
