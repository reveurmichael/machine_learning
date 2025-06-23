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

import sys
from pathlib import Path
from typing import Union

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Core Directory Configuration
# ---------------------------------------------------------------------------

# Root project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main logs directory (ROOT/logs/)
LOGS_ROOT = PROJECT_ROOT / "logs"

# Extensions logs directory (ROOT/logs/extensions/)
EXTENSIONS_LOGS_DIR = LOGS_ROOT / "extensions"

# Dataset storage directory (ROOT/logs/extensions/datasets/)
DATASETS_ROOT = EXTENSIONS_LOGS_DIR / "datasets"


# ---------------------------------------------------------------------------
# Heuristics Configuration
# ---------------------------------------------------------------------------

# Log prefix for heuristic algorithms
HEURISTICS_LOG_PREFIX = "heuristics"

# Default grid size for games
DEFAULT_GRID_SIZE = 10

# Available grid sizes for experimentation
SUPPORTED_GRID_SIZES = [8, 10, 12, 16, 20]


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

def get_dataset_dir(grid_size: int = DEFAULT_GRID_SIZE) -> Path:
    """
    Get dataset directory for a specific grid size.
    
    Args:
        grid_size: Size of the game grid
        
    Returns:
        Path to dataset directory (ROOT/logs/extensions/datasets/grid-size-N/)
    """
    return DATASETS_ROOT / f"grid-size-{grid_size}"


def get_dataset_path(
    data_structure: str,
    data_format: str,
    algorithm: str = "mixed",
    grid_size: int = DEFAULT_GRID_SIZE
) -> Path:
    """
    Get full path for a dataset file.
    
    Args:
        data_structure: Type of data structure ("tabular", "sequential", "graph")
        data_format: File format ("csv", "npz", "parquet")
        algorithm: Algorithm name or "mixed" for multi-algorithm datasets
        grid_size: Size of the game grid
        
    Returns:
        Path to dataset file
    """
    dataset_dir = get_dataset_dir(grid_size)
    filename = f"{data_structure}_{algorithm}_data.{data_format}"
    return dataset_dir / filename


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_extension_log_path(extension_name: str) -> Path:
    """
    Get log directory path for an extension.
    
    Args:
        extension_name: Name of the extension
        
    Returns:
        Path to extension log directory
    """
    return EXTENSIONS_LOGS_DIR / extension_name


def ensure_extensions_logs_dir() -> Path:
    """
    Ensure the extensions logs directory exists.
    
    Returns:
        Path to the extensions logs directory
    """
    EXTENSIONS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return EXTENSIONS_LOGS_DIR


def ensure_datasets_dir(grid_size: int = DEFAULT_GRID_SIZE) -> Path:
    """
    Ensure the datasets directory exists for a specific grid size.
    
    Args:
        grid_size: Size of the game grid
        
    Returns:
        Path to the dataset directory
    """
    dataset_dir = get_dataset_dir(grid_size)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_heuristic_log_path(algorithm: str, timestamp: str = None) -> Path:
    """
    Get log directory path for a heuristic algorithm.
    
    Args:
        algorithm: Name of the heuristic algorithm
        timestamp: Optional timestamp, generated if not provided
        
    Returns:
        Path to heuristic log directory
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_name = f"{HEURISTICS_LOG_PREFIX}-{algorithm.lower()}_{timestamp}"
    return EXTENSIONS_LOGS_DIR / log_name


def validate_grid_size(grid_size: int) -> bool:
    """
    Validate if a grid size is supported.
    
    Args:
        grid_size: Size to validate
        
    Returns:
        True if supported, False otherwise
    """
    return grid_size in SUPPORTED_GRID_SIZES


# ---------------------------------------------------------------------------
# Export Configuration
# ---------------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "LOGS_ROOT", 
    "EXTENSIONS_LOGS_DIR",
    "DATASETS_ROOT",
    "HEURISTICS_LOG_PREFIX",
    "DEFAULT_GRID_SIZE",
    "SUPPORTED_GRID_SIZES",
    "get_dataset_dir",
    "get_dataset_path",
    "get_extension_log_path",
    "ensure_extensions_logs_dir",
    "ensure_datasets_dir",
    "get_heuristic_log_path",
    "validate_grid_size",
] 