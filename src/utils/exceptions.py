"""Custom exception classes for the application."""

from typing import Optional


class AgentRecommendationError(Exception):
    """Base exception for agent recommendation system."""
    pass


class ConfigurationError(AgentRecommendationError):
    """Raised when there's a configuration error."""
    pass


class ValidationError(AgentRecommendationError):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


class APIError(AgentRecommendationError):
    """Raised when external API calls fail."""
    pass


class RetrievalError(AgentRecommendationError):
    """Raised when agent retrieval fails."""
    pass


class RankingError(AgentRecommendationError):
    """Raised when agent ranking fails."""
    pass

