"""
Input Validation Utilities
--------------------------

Common validation functions used across different components.
Provides standardized validation patterns following SSOT principles.

Design Philosophy:
- SSOT: Single location for validation logic
- Reusable: Common patterns used across web, CLI, and GUI
- Educational: Clear validation examples
- Consistent: Standardized error messages and handling

Educational Value:
- Shows proper input validation patterns
- Demonstrates error handling best practices
- Provides template for extension validation needs

Extension Pattern:
Extensions can use these validation functions and add their own
following the same patterns for consistency across the project.
"""

import argparse
from pathlib import Path
from typing import Optional

from utils.print_utils import print_info


def validate_grid_size(grid_size: int) -> int:
    """Validate grid size parameter.
    
    Args:
        grid_size: Grid size to validate
        
    Returns:
        Validated grid size
        
    Raises:
        ValueError: If grid size is invalid
        
    Educational Value: Shows input validation patterns
    Extension Pattern: Extensions can copy this validation approach
    """
    if not isinstance(grid_size, int) or grid_size < 5 or grid_size > 50:
        raise ValueError(f"Grid size must be between 5 and 50, got: {grid_size}")
    return grid_size


def validate_port(port: Optional[int]) -> Optional[int]:
    """Validate port number parameter.
    
    Args:
        port: Port number to validate (can be None for auto-detection)
        
    Returns:
        Validated port number or None
        
    Raises:
        ValueError: If port is invalid
        
    Educational Value: Shows optional parameter validation
    Extension Pattern: Extensions can use this for network configuration
    """
    if port is None:
        return None
    
    if not isinstance(port, int) or port < 1024 or port > 65535:
        raise ValueError(f"Port must be between 1024 and 65535, got: {port}")
    return port


def validate_positive_int(value: int, name: str, min_value: int = 1) -> int:
    """Validate positive integer parameter.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (default: 1)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is invalid
        
    Educational Value: Shows generic validation patterns
    Extension Pattern: Extensions can use this for various integer validations
    """
    if not isinstance(value, int) or value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def validate_string_not_empty(value: str, name: str) -> str:
    """Validate string is not empty.
    
    Args:
        value: String value to validate
        name: Parameter name for error messages
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If string is empty or not a string
        
    Educational Value: Shows string validation patterns
    Extension Pattern: Extensions can use this for string validations
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got: {repr(value)}")
    return value.strip()


def validate_choice(value: str, choices: list[str], name: str) -> str:
    """Validate value is in allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed choices
        name: Parameter name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value not in choices
        
    Educational Value: Shows choice validation patterns
    Extension Pattern: Extensions can use this for enum-like validations
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got: {value}")
    return value


def validate_llm_provider(provider: str) -> str:
    """Validate LLM provider name using registry in ``llm.providers``.

    Args:
        provider: LLM provider name to validate

    Returns:
        Validated provider name

    Raises:
        ValueError: If provider is invalid

    Educational Value:
        - Demonstrates decoupling validation from hard-coded lists
        - Uses single source of truth (LLM provider registry)
    """
    from llm.providers import list_providers  # Local import to avoid circular deps

    valid_providers = list_providers()
    return validate_choice(provider, valid_providers, "LLM provider")


def validate_llm_model(model: str) -> str:
    """Validate LLM model name.
    
    Args:
        model: LLM model name to validate
        
    Returns:
        Validated model name
        
    Raises:
        ValueError: If model is invalid
        
    Educational Value: Shows LLM model validation patterns
    Extension Pattern: Extensions can use this for model configuration
    """
    return validate_string_not_empty(model, "LLM model")


def validate_log_directory(log_dir: str) -> str:
    """Validate log directory exists and is accessible.
    
    Args:
        log_dir: Log directory path to validate
        
    Returns:
        Validated log directory path
        
    Raises:
        ValueError: If log directory is invalid
        
    Educational Value: Shows file system validation patterns
    Extension Pattern: Extensions can use this for file/directory validation
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise ValueError(f"Log directory does not exist: {log_dir}")
    if not log_path.is_dir():
        raise ValueError(f"Log path is not a directory: {log_dir}")
    return str(log_path)


def validate_game_number(game_number: int) -> int:
    """Validate game number parameter.
    
    Args:
        game_number: Game number to validate
        
    Returns:
        Validated game number
        
    Raises:
        ValueError: If game number is invalid
        
    Educational Value: Shows game-specific validation patterns
    Extension Pattern: Extensions can use this for game parameter validation
    """
    return validate_positive_int(game_number, "Game number", min_value=1)


# Centralized validate_arguments functions for different script types

def validate_human_web_arguments(args: argparse.Namespace) -> None:
    """Validate arguments for human web interface scripts.
    
    Args:
        args: Parsed command line arguments
        
    Educational Value: Shows centralized validation for human web scripts
    Extension Pattern: Extensions can copy this pattern for validation
    """
    # Validate grid size
    args.grid_size = validate_grid_size(args.grid_size)
    print_info(f"[HumanWebValidation] Grid size validated: {args.grid_size}x{args.grid_size}")
    
    # Validate port
    args.port = validate_port(args.port)
    if args.port:
        print_info(f"[HumanWebValidation] Port validated: {args.port}")
    else:
        print_info("[HumanWebValidation] Port will be auto-detected (dynamic allocation)")


def validate_llm_web_arguments(args: argparse.Namespace) -> None:
    """Validate arguments for LLM web interface scripts.
    
    Args:
        args: Parsed command line arguments
        
    Educational Value: Shows centralized validation for LLM web scripts
    Extension Pattern: Extensions can copy this pattern for LLM validation
    """
    # Validate grid size
    args.grid_size = validate_grid_size(args.grid_size)
    print_info(f"[LLMWebValidation] Grid size validated: {args.grid_size}x{args.grid_size}")
    
    # Validate LLM provider
    args.provider = validate_llm_provider(args.provider)
    print_info(f"[LLMWebValidation] LLM provider validated: {args.provider}")
    
    # Validate LLM model
    args.model = validate_llm_model(args.model)
    print_info(f"[LLMWebValidation] LLM model validated: {args.model}")
    
    # Validate port
    args.port = validate_port(args.port)
    if args.port:
        print_info(f"[LLMWebValidation] Port validated: {args.port}")
    else:
        print_info("[LLMWebValidation] Port will be auto-detected (dynamic allocation)")


def validate_replay_web_arguments(args: argparse.Namespace) -> None:
    """Validate arguments for replay web interface scripts.
    
    Args:
        args: Parsed command line arguments
        
    Educational Value: Shows centralized validation for replay web scripts
    Extension Pattern: Extensions can copy this pattern for replay validation
    """
    # Validate log directory
    args.log_dir = validate_log_directory(args.log_dir)
    print_info(f"[ReplayWebValidation] Log directory validated: {args.log_dir}")
    
    # Validate game number
    args.game = validate_game_number(args.game)
    print_info(f"[ReplayWebValidation] Game number validated: {args.game}")
    
    # Validate port
    args.port = validate_port(args.port)
    if args.port:
        print_info(f"[ReplayWebValidation] Port validated: {args.port}")
    else:
        print_info("[ReplayWebValidation] Port will be auto-detected (dynamic allocation)") 