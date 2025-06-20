"""
Text Processing and Formatting Utilities

This module provides lightweight, dependency-free helper functions for common
text processing and formatting tasks. It is designed to be:

- Fast and efficient with compiled regex patterns
- Safe with graceful error handling  
- Generic and reusable across all project components
- Memory-efficient for large text processing

Typical use cases include:
- Cleaning up LLM responses for display
- Formatting JSON code blocks in user interfaces
- Preparing text for logging or file output
- Standardizing text representation across the application

All functions in this module are generic and not specific to any single
task (e.g., Task-0).
"""

from __future__ import annotations

import json
import re
from typing import Final

__all__ = [
    "process_response_for_display",
    "format_code_blocks",
    "truncate_text",
    "clean_whitespace",
]

# Pre-compiled regex patterns for performance optimization
# Matches content in triple backticks with optional language specification
_CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:json|javascript|python|bash|shell)?\s*(.*?)\s*```", 
    re.DOTALL | re.IGNORECASE
)

# Pattern for excessive whitespace normalization
_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r'\s+')


def clean_whitespace(text: str) -> str:
    """
    Normalizes whitespace in text by collapsing multiple spaces, tabs, and
    newlines into single spaces.
    
    Args:
        text: The input text to clean.
        
    Returns:
        Text with normalized whitespace.
        
    Example:
        >>> clean_whitespace("Hello\\n\\n   world\\t\\t!")
        'Hello world !'
    """
    if not text:
        return text
    return _WHITESPACE_PATTERN.sub(' ', text.strip())


def truncate_text(text: str, max_length: int = 2000, suffix: str = "... [truncated]") -> str:
    """
    Truncates text to a maximum length with an optional suffix.
    
    Args:
        text: The text to potentially truncate.
        max_length: Maximum allowed length before truncation.
        suffix: String to append when text is truncated.
        
    Returns:
        The original text if under max_length, otherwise truncated text with suffix.
        
    Example:
        >>> truncate_text("A very long string", max_length=10)
        'A very lon... [truncated]'
    """
    if not text or len(text) <= max_length:
        return text
    
    # Ensure we don't exceed max_length even with the suffix
    truncate_at = max(0, max_length - len(suffix))
    return text[:truncate_at] + suffix


def process_response_for_display(response: str, max_length: int = 2000) -> str:
    """
    Processes a raw text response for clean display in user interfaces.

    This is a comprehensive formatting pipeline that:
    1. Formats any code blocks (especially JSON) for readability
    2. Truncates the response to prevent UI overflow
    3. Handles edge cases gracefully

    Args:
        response: The raw string response from a source like an LLM.
        max_length: The maximum allowed length of the output string.

    Returns:
        The processed and potentially truncated string, ready for display.
        
    Example:
        >>> response = 'Here is some JSON: ```{"key": "value"}```'
        >>> process_response_for_display(response)
        'Here is some JSON: ```json\\n{\\n  "key": "value"\\n}\\n```'
    """
    if not response:
        return ""
    
    try:
        # First, format any code blocks for better readability
        formatted_response = format_code_blocks(response)
        
        # Then truncate if necessary
        return truncate_text(formatted_response, max_length)
        
    except Exception:
        # Fallback to basic truncation if formatting fails
        return truncate_text(response, max_length)


def format_code_blocks(text: str) -> str:
    """
    Finds all code blocks (```...```) in a string and formats them for readability.

    The function intelligently handles different content types:
    - Valid JSON: Pretty-printed with proper indentation
    - Other code: Returned as clean code blocks
    - Malformed content: Safely preserved as-is

    Args:
        text: The input string that may contain one or more code blocks.

    Returns:
        The text with all found code blocks properly formatted.
        
    Example:
        >>> text = 'Check this: ```{"name":"John","age":30}```'
        >>> format_code_blocks(text)
        'Check this: ```json\\n{\\n  "name": "John",\\n  "age": 30\\n}\\n```'
    """
    if not text:
        return text

    def _format_single_block(match: re.Match[str]) -> str:
        """Process a single regex match for code block formatting."""
        code_content = match.group(1).strip()
        
        if not code_content:
            return "```\n```"  # Empty code block

        # Attempt JSON formatting first (most common use case)
        try:
            # Try to parse as JSON
            json_obj = json.loads(code_content)
            formatted_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
            return f"```json\n{formatted_json}\n```"
            
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return as generic formatted code block
            return f"```\n{code_content}\n```"

    # Apply formatting to all code blocks in the text
    return _CODE_BLOCK_PATTERN.sub(_format_single_block, text)
