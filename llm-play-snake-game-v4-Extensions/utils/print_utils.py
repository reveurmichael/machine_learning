"""
Print and Display Utilities for Snake Game AI
============================================

This module centralizes print-related utilities including logging, colorama support,
and display formatting. It provides a single source of truth for all print-related
functionality across the project.

Design Philosophy:
- Centralized print utilities: All print-related functions in one place
- Future colorama support: Ready for colored output when needed
- Simple logging: Lightweight logging patterns following SUPREME_RULES
- Universal utilities: Used across Task-0 and all extensions

Educational Value:
- Shows centralized utility patterns
- Demonstrates simple logging without complex frameworks
- Provides foundation for enhanced terminal output

Future Extensions:
- Colorama integration for colored terminal output
- Progress bars and status indicators
- Formatted table output
- Terminal clearing and cursor management
"""

from typing import Callable

__all__ = [
    "create_logger",
    "print_info",
    "print_warning", 
    "print_error",
    "print_success",
]


def create_logger(prefix: str = "App") -> Callable[[str], None]:
    """Create a simple logging function with a prefix.
    
    Args:
        prefix: Prefix for log messages
        
    Returns:
        A logging function that can be called with messages
        
    Educational Value: Shows simple logging pattern following SUPREME_RULES
    Extension Pattern: Extensions can create their own loggers
    
    Example:
        >>> log = create_logger("WebApp")
        >>> log("Server started")
        # Outputs: [WebApp] Server started
    """
    return lambda msg: print(f"[{prefix}] {msg}")


def print_info(message: str, prefix: str = "INFO") -> None:
    """Print informational message with standardized format.
    
    Args:
        message: Message to print
        prefix: Prefix for the message (default: INFO)
        
    Educational Value: Shows standardized info message formatting
    Extension Pattern: Extensions can use for consistent info display
    """
    print(f"[{prefix}] {message}")


def print_warning(message: str) -> None:
    """Print warning message with standardized format.
    
    Args:
        message: Warning message to print
        
    Educational Value: Shows standardized warning message formatting
    Extension Pattern: Extensions can use for consistent warning display
    """
    print(f"[WARNING] {message}")


def print_error(message: str) -> None:
    """Print error message with standardized format.
    
    Args:
        message: Error message to print
        
    Educational Value: Shows standardized error message formatting
    Extension Pattern: Extensions can use for consistent error display
    """
    print(f"[ERROR] {message}")


def print_success(message: str) -> None:
    """Print success message with standardized format.
    
    Args:
        message: Success message to print
        
    Educational Value: Shows standardized success message formatting
    Extension Pattern: Extensions can use for consistent success display
    """
    print(f"[SUCCESS] {message}")


# Future colorama integration (when needed):
#
# def print_colored(message: str, color: str = "white") -> None:
#     """Print colored message using colorama (future implementation)."""
#     # Implementation will use colorama when needed
#     pass
#
# def print_progress_bar(current: int, total: int, width: int = 50) -> None:
#     """Print progress bar (future implementation)."""
#     # Implementation for progress indicators
#     pass 