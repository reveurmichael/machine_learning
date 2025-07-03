"""
Configuration constants for Snake Game AI extensions.

This module defines shared configuration constants and path patterns
used across all extensions, following SUPREME_RULES from final-decision-10.md.

Design Philosophy:
- Centralized configuration management
- Simple constants (no complex configuration classes)
- Consistent naming patterns across extensions
- Path templates for organized data storage

Reference: docs/extensions-guideline/final-decision-10.md
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from .path_constants import *

# =============================================================================
# Directory Configuration
# =============================================================================

# Base project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOGS_ROOT = PROJECT_ROOT / "logs"
EXTENSIONS_LOGS_DIR = str(LOGS_ROOT / "extensions")
"""Directory for all extension logs: logs/extensions/"""

# Extension-specific prefixes for log directories
HEURISTICS_LOG_PREFIX = "heuristics-"
"""Prefix for heuristics extension log directories: heuristics-{algorithm}_{timestamp}"""

SUPERVISED_LOG_PREFIX = "supervised-"
"""Prefix for supervised learning extension log directories: supervised-{model}_{timestamp}"""

REINFORCEMENT_LOG_PREFIX = "reinforcement-"
"""Prefix for reinforcement learning extension log directories: reinforcement-{agent}_{timestamp}"""

EVOLUTIONARY_LOG_PREFIX = "evolutionary-"
"""Prefix for evolutionary algorithm extension log directories: evolutionary-{algorithm}_{timestamp}"""

# =============================================================================
# Dataset Format Configuration
# =============================================================================

# Supported dataset formats
SUPPORTED_FORMATS = ["csv", "jsonl", "npz"]
"""List of supported dataset formats across all extensions."""

# Default format preferences by extension type
DEFAULT_FORMATS = {
    "heuristics": ["csv", "jsonl"],  # v0.04 supports both
    "supervised": ["csv", "npz"],
    "reinforcement": ["npz"],
    "evolutionary": ["npz"]
}
"""Default dataset formats for each extension type."""

# =============================================================================
# Common Game Configuration
# =============================================================================

DEFAULT_GRID_SIZE = 10
"""Default grid size for Snake game across all extensions."""

MIN_GRID_SIZE = 4
"""Minimum allowed grid size."""

MAX_GRID_SIZE = 50
"""Maximum allowed grid size."""

DEFAULT_MAX_STEPS = 1000
"""Default maximum steps per game."""

DEFAULT_MAX_GAMES = 10
"""Default maximum number of games per session."""

# =============================================================================
# Export Configuration for Common Package
# =============================================================================

__all__ = [
    # Path constants
    "PROJECT_ROOT",
    "LOGS_ROOT", 
    "EXTENSIONS_LOGS_DIR",
    
    # Extension prefixes
    "HEURISTICS_LOG_PREFIX",
    "SUPERVISED_LOG_PREFIX",
    "REINFORCEMENT_LOG_PREFIX", 
    "EVOLUTIONARY_LOG_PREFIX",
    
    # Format configuration
    "SUPPORTED_FORMATS",
    "DEFAULT_FORMATS",
    
    # Game configuration
    "DEFAULT_GRID_SIZE",
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_GAMES",
    
    # Path constants from path_constants.py
    "ROOT_DIR_NAME",
    "EXTENSIONS_DIR_NAME",
    "DATASETS_DIR_NAME",
    "MODELS_DIR_NAME",
    "DATASET_PATH_TEMPLATE",
    "MODEL_PATH_TEMPLATE",
    "DATASET_FILE_PATTERNS",
    "MODEL_FILE_PATTERNS",
    "DEFAULT_PATHS"
] 