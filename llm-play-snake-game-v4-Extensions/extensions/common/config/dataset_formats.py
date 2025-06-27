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

