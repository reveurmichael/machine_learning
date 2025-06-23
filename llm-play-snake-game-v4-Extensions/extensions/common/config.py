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
    from extensions.common.dataset_directory_manager import DatasetDirectoryManager
    
    # Use the constant directly
    log_dir = EXTENSIONS_LOGS_DIR / "my-extension"
    
    # Or use the helper function from the manager
    log_path = DatasetDirectoryManager.get_extension_log_path("my-extension")
"""

from __future__ import annotations

from pathlib import Path

# ---------------------
# Ensure repository root is discoverable via the canonical utility once – this
# avoids DIY sys.path hacking in every module and keeps *single source of truth*.
# ---------------------

from utils.path_utils import get_project_root  # Absolute import – root package

# Cache the resolved project root for cheaper repeated access
PROJECT_ROOT: Path = get_project_root()

# ---------------------
# Core Directory Configuration
# ---------------------

# Main logs directory (ROOT/logs/)
LOGS_ROOT = PROJECT_ROOT / "logs"

# Extensions logs directory (ROOT/logs/extensions/)
EXTENSIONS_LOGS_DIR = LOGS_ROOT / "extensions"

# Dataset storage directory (ROOT/logs/extensions/datasets/)
DATASETS_ROOT = EXTENSIONS_LOGS_DIR / "datasets"


# ---------------------
# Heuristics Configuration
# ---------------------

# Log prefix for heuristic algorithms
HEURISTICS_LOG_PREFIX = "heuristics"

# Default grid size for games
DEFAULT_GRID_SIZE = 10

# Available grid sizes for experimentation
SUPPORTED_GRID_SIZES = [8, 10, 12, 16, 20]

# ---------------------
# Export Configuration
# ---------------------

__all__ = [
    "PROJECT_ROOT",
    "LOGS_ROOT", 
    "EXTENSIONS_LOGS_DIR",
    "DATASETS_ROOT",
    "HEURISTICS_LOG_PREFIX",
    "DEFAULT_GRID_SIZE",
    "SUPPORTED_GRID_SIZES",
]