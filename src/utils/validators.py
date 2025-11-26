"""Input validation utilities for production use."""

import re
from typing import Optional
from flask import jsonify


def validate_goal(goal: str, max_length: int = 2000) -> tuple[bool, Optional[str]]:
    """
    Validate goal input.
    
    Args:
        goal: Goal string to validate
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not goal:
        return False, "Goal cannot be empty"
    
    if not isinstance(goal, str):
        return False, "Goal must be a string"
    
    goal = goal.strip()
    if not goal:
        return False, "Goal cannot be only whitespace"
    
    if len(goal) > max_length:
        return False, f"Goal exceeds maximum length of {max_length} characters"
    
    # Check for potentially malicious patterns
    if re.search(r'<script|javascript:|onerror=|onload=', goal, re.IGNORECASE):
        return False, "Goal contains potentially unsafe content"
    
    return True, None


def validate_top_k(top_k: Optional[int], max_k: int = 20) -> tuple[bool, Optional[str], int]:
    """
    Validate top_k parameter.
    
    Args:
        top_k: Number of recommendations requested
        max_k: Maximum allowed value
        
    Returns:
        Tuple of (is_valid, error_message, validated_top_k)
    """
    if top_k is None:
        return True, None, 5  # Default
    
    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        return False, "top_k must be an integer", 5
    
    if top_k < 1:
        return False, "top_k must be at least 1", 5
    
    if top_k > max_k:
        return False, f"top_k cannot exceed {max_k}", max_k
    
    return True, None, top_k


def sanitize_error_message(error: Exception, include_details: bool = False) -> str:
    """
    Sanitize error messages for production.
    
    Args:
        error: Exception object
        include_details: Whether to include detailed error info (only for internal logging)
        
    Returns:
        Safe error message for client
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Don't expose internal details to clients
    if not include_details:
        # Generic messages for common errors
        if "GOOGLE_API_KEY" in error_msg:
            return "API configuration error. Please contact support."
        if "Connection" in error_type or "Timeout" in error_type:
            return "Service temporarily unavailable. Please try again."
        if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            return "Service configuration error. Please contact support."
        
        # Return sanitized message
        return f"An error occurred: {error_type}"
    
    # For internal logging, include full details
    return f"{error_type}: {error_msg}"

