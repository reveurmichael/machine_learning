"""
Basic Dataset Format Specifications for Snake Game Extensions.

This module defines simple format specifications that can be used
across different extensions without being overly restrictive.

Design Philosophy: Keep it Simple and Generic
- Basic format definitions only
- No complex validation logic
- Extensions can add their own specific requirements
"""

from typing import List, Set, Dict, Any

# =============================================================================
# Basic CSV Format (16-Feature Tabular) - Commonly Used
# =============================================================================

CSV_BASIC_COLUMNS: List[str] = [
    # Metadata
    "game_id", "step_in_game",
    
    # Basic position features
    "head_x", "head_y", "apple_x", "apple_y",
    
    # Game state
    "snake_length",
    
    # Direction features (binary 0/1)
    "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
    
    # Safety features (binary 0/1)
    "danger_straight", "danger_left", "danger_right",
    
    # Space features (counts)
    "free_space_up", "free_space_down", "free_space_left", "free_space_right",
    
    # Target
    "target_move"
]
"""Basic CSV column specification - extensions can modify as needed."""

CSV_VALID_MOVES: Set[str] = {"UP", "DOWN", "LEFT", "RIGHT"}
"""Standard move directions for snake game."""

# =============================================================================
# Basic JSONL Format - For Language Models
# =============================================================================

JSONL_BASIC_KEYS: Set[str] = {"prompt", "completion"}
"""Basic required keys for JSONL records."""

JSONL_COMMON_OPTIONAL_KEYS: Set[str] = {
    "game_id", "step_in_game", "metadata"
}
"""Common optional keys that extensions might use."""

# =============================================================================
# File Extensions by Type
# =============================================================================

COMMON_DATASET_EXTENSIONS: Dict[str, Set[str]] = {
    "tabular": {".csv"},
    "language": {".jsonl", ".json"},
    "arrays": {".npz", ".npy"},
}
"""Common file extensions for different data types."""

# =============================================================================
# Grid Size Defaults
# =============================================================================

DEFAULT_GRID_SIZE: int = 10
"""Default grid size for snake game."""

COMMON_GRID_SIZES: Set[int] = {8, 10, 12, 16, 20}
"""Commonly used grid sizes across extensions."""

# -----------------------------------------------------------------------------
# Aliases / Extended Names (kept for backward-compatibility inside common utils)
# -----------------------------------------------------------------------------
# NOTE:
# A handful of helper utilities inside `extensions/common/utils/` were written
# before the great simplification effort and are still expecting more verbose
# constant names.  In order to keep the public interface of this *very* small
# module stable – and without introducing another indirection layer – we simply
# provide *aliases* that point to the already defined, single-source-of-truth
# lists above.  These aliases **MUST NOT** be used for new code; always prefer
# the `CSV_BASIC_*` / `JSONL_BASIC_*` style names instead.  They live here
# purely to avoid breaking existing utilities while we refactor progressively.

CSV_COLUMN_NAMES = CSV_BASIC_COLUMNS  # Historical alias; identical content

# A more granular split of the 16-feature schema that some utilities rely on.
# The lists are derived programmatically to avoid duplication and to ensure the
# single source of truth principle.
CSV_METADATA_COLUMNS = [
    "game_id",
    "step_in_game",
]

CSV_TARGET_COLUMN = "target_move"

# All positions features (absolute board coordinates)
CSV_POSITION_FEATURES = [
    "head_x", "head_y", "apple_x", "apple_y",
]

# Binary indicator features
CSV_BINARY_FEATURES = [
    "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
    "danger_straight", "danger_left", "danger_right",
]

# Count-based features
CSV_COUNT_FEATURES = [
    "free_space_up", "free_space_down", "free_space_left", "free_space_right",
]

# Feature names exclude metadata + target column
CSV_FEATURE_NAMES = CSV_POSITION_FEATURES + [
    "snake_length",
] + CSV_BINARY_FEATURES + CSV_COUNT_FEATURES

# For convenience in a couple of spots we expose the full expected column set
CSV_EXPECTED_COLUMNS = CSV_METADATA_COLUMNS + CSV_FEATURE_NAMES + [CSV_TARGET_COLUMN]

# Keep backward compatibility with older variable name (plural vs. singular)
CSV_EXPECTED_COLUMNS_SET = set(CSV_EXPECTED_COLUMNS)

