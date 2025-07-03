"""
Print and Display Utilities for Snake Game AI
============================================

This module centralizes print-related utilities including logging, colorama support,
and display formatting. It provides a single source of truth for all print-related
functionality across the project.

Design Philosophy:
- Centralized print utilities: All print-related functions in one place
- Colorama integration: Colored output for better user experience
- Simple logging: Lightweight logging patterns following SUPREME_RULES
- Universal utilities: Used across Task-0 and all extensions

Educational Value:
- Shows centralized utility patterns
- Demonstrates simple logging without complex frameworks
- Provides foundation for enhanced terminal output

Future Extensions:
- Progress bars and status indicators
- Formatted table output
- Terminal clearing and cursor management
"""

from typing import Callable
from colorama import Fore, Style, init as _colorama_init

# Initialize Colorama (auto-reset to avoid manual Style.RESET_ALL)
_colorama_init(autoreset=True)

# Emoji constants for consistent usage
EMOJI_SUCCESS = "✅"
EMOJI_WARNING = "⚠️"
EMOJI_ERROR   = "❌"

__all__ = [
    "print_info",
    "print_warning", 
    "print_error",
    "print_success",
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


def print_warning(message: str) -> None:
    """Print warning message with standardized format and yellow color.
    
    Args:
        message: Warning message to print
        
    Educational Value: Shows standardized warning message formatting
    Extension Pattern: Extensions can use for consistent warning display
    """
    print(Fore.YELLOW + f"{EMOJI_WARNING} {message}" + Style.RESET_ALL)


def print_error(message: str) -> None:
    """Print error message with standardized format and red color.
    
    Args:
        message: Error message to print
        
    Educational Value: Shows standardized error message formatting
    Extension Pattern: Extensions can use for consistent error display
    """
    print(Fore.RED + f"{EMOJI_ERROR} {message}" + Style.RESET_ALL)


def print_success(message: str) -> None:
    """Print success message with standardized format and green color.
    
    Args:
        message: Success message to print
        
    Educational Value: Shows standardized success message formatting
    Extension Pattern: Extensions can use for consistent success display
    """
    print(Fore.GREEN + f"{EMOJI_SUCCESS} {message}" + Style.RESET_ALL)

