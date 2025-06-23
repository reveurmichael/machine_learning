"""
Common configuration constants for extensions.

This module provides shared configuration constants that can be used across
all extensions to ensure consistency and avoid hardcoding paths and values.

Design Pattern: Configuration Singleton
- Centralizes all extension-specific configuration
- Provides type-safe constants with clear documentation
- Ensures consistency across all extension versions
- Makes configuration changes easy to manage

Usage:
    from extensions.common.config import EXTENSIONS_LOGS_DIR
    from extensions.common.config import get_extension_log_path
    
    # Use the constant directly
    log_dir = EXTENSIONS_LOGS_DIR / "my-extension"
    
    # Or use the helper function
    log_path = get_extension_log_path("my-extension")
"""

from __future__ import annotations

import pathlib
from typing import Final

# Import path utilities to ensure project root is available
from .path_utils import ensure_project_root_on_path

# Ensure project root is on path for relative path resolution
PROJECT_ROOT = ensure_project_root_on_path()

# ---------------------------------------------------------------------------
# Extension Logging Configuration
# ---------------------------------------------------------------------------

# Base directory for all extension logs (relative to project root)
EXTENSIONS_LOGS_DIR: Final[pathlib.Path] = PROJECT_ROOT / "logs" / "extensions"

# ---------------------------------------------------------------------------
# Extension Naming Conventions
# ---------------------------------------------------------------------------

# Prefix for heuristic extension log folders
HEURISTICS_LOG_PREFIX: Final[str] = "heuristics-"

# Prefix for other extension types (future use)
RL_LOG_PREFIX: Final[str] = "rl-"
SUPERVISED_LOG_PREFIX: Final[str] = "supervised-"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_extension_log_path(extension_name: str) -> pathlib.Path:
    """
    Get the log directory path for a specific extension.
    
    Args:
        extension_name: Name of the extension (e.g., "heuristics-bfs")
        
    Returns:
        Path to the extension's log directory
        
    Example:
        >>> get_extension_log_path("heuristics-bfs")
        Path('/path/to/project/logs/extensions/heuristics-bfs')
    """
    return EXTENSIONS_LOGS_DIR / extension_name


def ensure_extensions_logs_dir() -> pathlib.Path:
    """
    Ensure the extensions logs directory exists.
    
    Creates the directory if it doesn't exist and returns the path.
    
    Returns:
        Path to the extensions logs directory
        
    Example:
        >>> ensure_extensions_logs_dir()
        Path('/path/to/project/logs/extensions')
    """
    EXTENSIONS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return EXTENSIONS_LOGS_DIR


def get_heuristic_log_path(algorithm_name: str) -> pathlib.Path:
    """
    Get the log directory path for a heuristic algorithm.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., "bfs", "astar")
        
    Returns:
        Path to the algorithm's log directory
        
    Example:
        >>> get_heuristic_log_path("bfs")
        Path('/path/to/project/logs/extensions/heuristics-bfs')
    """
    return get_extension_log_path(f"{HEURISTICS_LOG_PREFIX}{algorithm_name}")


# ---------------------------------------------------------------------------
# Export Configuration
# ---------------------------------------------------------------------------

__all__ = [
    "EXTENSIONS_LOGS_DIR",
    "HEURISTICS_LOG_PREFIX",
    "RL_LOG_PREFIX", 
    "SUPERVISED_LOG_PREFIX",
    "get_extension_log_path",
    "ensure_extensions_logs_dir",
    "get_heuristic_log_path",
] 