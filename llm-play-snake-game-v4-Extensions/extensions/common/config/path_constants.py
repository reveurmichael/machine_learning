"""
Path Constants and Templates for Snake Game AI Extensions.

This module defines standardized path patterns and directory structure
templates used across all extensions.

Design Pattern: Template Method Pattern
- Standardized path construction patterns
- Configurable path templates
- Consistent directory hierarchies

Educational Value:
Demonstrates how to create maintainable path management systems with
proper separation of concerns and reusable templates.
"""

from typing import Dict, List

# =============================================================================
# Base Path Templates
# =============================================================================

# Core directory structure
ROOT_DIR_NAME: str = "logs"
"""Root directory for all generated files."""

EXTENSIONS_DIR_NAME: str = "extensions" 
"""Subdirectory under logs for extension-specific files."""

DATASETS_DIR_NAME: str = "datasets"
"""Subdirectory for dataset storage."""

MODELS_DIR_NAME: str = "models"
"""Subdirectory for trained model storage."""

# =============================================================================
# Path Template Patterns
# =============================================================================

DATASET_PATH_TEMPLATE: str = "{root_dir}/{extensions_dir}/{datasets_dir}/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}"
"""Template for dataset directory paths.
Example: logs/extensions/datasets/grid-size-10/heuristics_v0.04_20240101_120000"""

MODEL_PATH_TEMPLATE: str = "{root_dir}/{extensions_dir}/{models_dir}/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}"
"""Template for model directory paths.
Example: logs/extensions/models/grid-size-10/supervised_v0.02_20240101_120000"""


CHECKPOINT_PATH_TEMPLATE: str = "{model_dir}/checkpoints/{algorithm}/epoch_{epoch:03d}_{metric}_{value:.4f}.pth"
"""Template for model checkpoint paths."""

EXPORT_PATH_TEMPLATE: str = "{model_dir}/exports/{algorithm}/{format}/{timestamp}"
"""Template for model export paths."""

# =============================================================================
# Grid-Size Directory Patterns
# =============================================================================

GRID_SIZE_DIR_PATTERN: str = "grid-size-{grid_size}"
"""Pattern for grid-size specific directories."""

DEFAULT_GRID_SIZE: int = 10
"""Default grid size for new experiments."""

# =============================================================================
# Extension Version Patterns
# =============================================================================

VERSION_PATTERN: str = "v{major}.{minor:02d}"
"""Pattern for version formatting (e.g., v0.03)."""

EXTENSION_DIR_PATTERN: str = "{extension_type}-{version}"
"""Pattern for extension directory names (e.g., heuristics-v0.03)."""

# Educational Note (final-decision-10.md Guideline 3):
# We should be able to add new extensions easily and try out new ideas.
# Therefore, we don't restrict specific extensions or versions - 
# this encourages experimentation and flexibility in an educational project.

# =============================================================================
# Timestamp Patterns
# =============================================================================

TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
"""Standard timestamp format for directories and files."""

TIMESTAMP_PATTERN: str = r"\d{8}_\d{6}"
"""Regex pattern for timestamp validation."""

# =============================================================================
# File Naming Patterns
# =============================================================================

DATASET_FILE_PATTERNS: Dict[str, str] = {
    "csv": "{algorithm}_dataset.csv",
    "jsonl": "{algorithm}_reasoning.jsonl", 
    "npz_sequential": "{algorithm}_sequences.npz",
    "npz_spatial": "{algorithm}_spatial.npz",
    "npz_raw": "{algorithm}_raw.npz",
    "npz_evolutionary": "{algorithm}_evolutionary.npz"
}
"""File naming patterns for different dataset formats."""

MODEL_FILE_PATTERNS: Dict[str, str] = {
    "pytorch": "{algorithm}_model.pth",
    "onnx": "{algorithm}_model.onnx",
    "sklearn": "{algorithm}_model.pkl",
    "joblib": "{algorithm}_model.joblib"
}
"""File naming patterns for different model formats."""

# =============================================================================
# Default Path Configurations
# =============================================================================

DEFAULT_PATHS: Dict[str, str] = {
    "root": ROOT_DIR_NAME,
    "extensions": f"{ROOT_DIR_NAME}/{EXTENSIONS_DIR_NAME}",
    "datasets": f"{ROOT_DIR_NAME}/{EXTENSIONS_DIR_NAME}/{DATASETS_DIR_NAME}",
    "models": f"{ROOT_DIR_NAME}/{EXTENSIONS_DIR_NAME}/{MODELS_DIR_NAME}"
}
"""Default path configurations."""
