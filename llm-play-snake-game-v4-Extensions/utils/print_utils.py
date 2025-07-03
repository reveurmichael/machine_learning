"""
Print and Display Utilities for Snake Game AI
============================================

This module centralizes print-related utilities including logging, colorama support,
and display formatting. It provides a single source of truth for all print-related
functionality across the project.
"""

from typing import Callable
from colorama import Fore, Style, init as _colorama_init

# Initialize Colorama (auto-reset to avoid manual Style.RESET_ALL)
_colorama_init(autoreset=True)

# Default emoji constants for consistent usage
# These can be overridden by passing custom emoji to print functions
DEFAULT_EMOJI_SUCCESS = "✅"
DEFAULT_EMOJI_WARNING = "⚠️"
DEFAULT_EMOJI_ERROR   = "❌"
DEFAULT_EMOJI_IMPORTANT = "🔔"


__all__ = [
    "print_info",
    "print_warning", 
    "print_error",
    "print_success",
    "print_important",
    # Default emoji constants
    "DEFAULT_EMOJI_SUCCESS",
    "DEFAULT_EMOJI_WARNING", 
    "DEFAULT_EMOJI_ERROR",
    "DEFAULT_EMOJI_IMPORTANT",
]


def print_info(message: str, prefix: str = "INFO") -> None:
    """Print informational message with standardized format and cyan color.
    
    Args:
        message: Message to print
        prefix: Prefix for the message (default: INFO)
        
    Educational Value: Shows standardized info message formatting
    Extension Pattern: Extensions can use for consistent info display
    """
    print(Fore.CYAN + f"[{prefix}] {message}" + Style.RESET_ALL)


def print_warning(message: str, emoji: str = DEFAULT_EMOJI_WARNING) -> None:
    """Print warning message with standardized format and yellow color.
    
    Args:
        message: Warning message to print
        emoji: Custom emoji to use (default: ⚠️)
        
    Educational Value: Shows standardized warning message formatting
    Extension Pattern: Extensions can use for consistent warning display
    """
    print(Fore.YELLOW + f"{emoji} {message}" + Style.RESET_ALL)


def print_error(message: str, emoji: str = DEFAULT_EMOJI_ERROR) -> None:
    """Print error message with standardized format and red color.
    
    Args:
        message: Error message to print
        emoji: Custom emoji to use (default: ❌)
        
    Educational Value: Shows standardized error message formatting
    Extension Pattern: Extensions can use for consistent error display
    """
    print(Fore.RED + f"{emoji} {message}" + Style.RESET_ALL)


def print_success(message: str, emoji: str = DEFAULT_EMOJI_SUCCESS) -> None:
    """Print success message with standardized format and green color.
    
    Args:
        message: Success message to print
        emoji: Custom emoji to use (default: ✅)
        
    Educational Value: Shows standardized success message formatting
    Extension Pattern: Extensions can use for consistent success display
    """
    print(Fore.GREEN + f"{emoji} {message}" + Style.RESET_ALL)


def print_important(message: str, emoji: str = DEFAULT_EMOJI_IMPORTANT) -> None:
    """Print important message with standardized format and magenta color.
    
    Args:
        message: Important message to print
        emoji: Custom emoji to use (default: 🔔)
        
    Educational Value: Shows standardized important message formatting
    Extension Pattern: Extensions can use for consistent important message display
    """
    print(Fore.MAGENTA + f"{emoji} {message}" + Style.RESET_ALL)
