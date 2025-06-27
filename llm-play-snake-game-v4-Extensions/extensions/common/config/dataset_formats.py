"""
Dataset Format Specifications for Snake Game AI Extensions.

This module defines the structure and format requirements for different
dataset types used across extensions.

Design Pattern: Specification Pattern
- Clear format definitions
- Validation-ready specifications  
- Type-safe format descriptions

Educational Value:
Demonstrates how to formally specify data formats with proper validation
and documentation for different use cases.
"""

from typing import List, Dict, Any, Set, Tuple

# =============================================================================
# CSV Format Specification (16-Feature Tabular)
# =============================================================================

CSV_COLUMN_NAMES: List[str] = [
    # Metadata columns
    "game_id",
    "step_in_game",
    
    # Position features
    "head_x",
    "head_y", 
    "apple_x",
    "apple_y",
    
    # Game state features
    "snake_length",
    
    # Apple direction features (binary flags)
    "apple_dir_up",
    "apple_dir_down", 
    "apple_dir_left",
    "apple_dir_right",
    
    # Danger detection features (binary flags)
    "danger_straight",
    "danger_left",
    "danger_right",
    
    # Free space features (counts)
    "free_space_up",
    "free_space_down",
    "free_space_left", 
    "free_space_right",
    
    # Target column
    "target_move"
]
"""Complete CSV column specification for 16-feature tabular format."""

CSV_FEATURE_NAMES: List[str] = [
    "head_x", "head_y", "apple_x", "apple_y",
    "snake_length",
    "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
    "danger_straight", "danger_left", "danger_right", 
    "free_space_up", "free_space_down", "free_space_left", "free_space_right"
]
"""Feature columns only (excludes metadata and target)."""

CSV_METADATA_COLUMNS: List[str] = ["game_id", "step_in_game"]
"""Metadata columns for tracking and identification."""

CSV_TARGET_COLUMN: str = "target_move"
"""Target column name."""

CSV_EXPECTED_COLUMNS: int = 19
"""Expected total number of columns (2 metadata + 16 features + 1 target)."""

CSV_BINARY_FEATURES: Set[str] = {
    "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
    "danger_straight", "danger_left", "danger_right"
}
"""Features that should contain only 0/1 values."""

CSV_COUNT_FEATURES: Set[str] = {
    "free_space_up", "free_space_down", "free_space_left", "free_space_right"
}
"""Features that contain count values (non-negative integers)."""

CSV_POSITION_FEATURES: Set[str] = {
    "head_x", "head_y", "apple_x", "apple_y"
}
"""Features that contain position coordinates."""

CSV_VALID_MOVES: Set[str] = {"UP", "DOWN", "LEFT", "RIGHT"}
"""Valid values for target_move column."""

# =============================================================================
# JSONL Format Specification (Language-Rich)
# =============================================================================

JSONL_REQUIRED_KEYS: Set[str] = {"prompt", "completion"}
"""Required keys for each JSONL record."""

JSONL_OPTIONAL_KEYS: Set[str] = {
    "game_id", "step_in_game", "algorithm", "reasoning", 
    "confidence", "metadata"
}
"""Optional keys for enhanced JSONL records."""

JSONL_PROMPT_TEMPLATES: Dict[str, str] = {
    "basic": "Snake at ({head_x},{head_y}), apple at ({apple_x},{apple_y}). What move?",
    "detailed": "Snake head at ({head_x},{head_y}), apple at ({apple_x},{apple_y}). Snake length: {snake_length}. Dangers: {dangers}. What's the best move?",
    "reasoning": "Given the game state:\n- Snake head: ({head_x},{head_y})\n- Apple: ({apple_x},{apple_y})\n- Snake length: {snake_length}\n- Immediate dangers: {dangers}\n\nAnalyze the situation and choose the optimal move."
}
"""Template formats for JSONL prompts."""

JSONL_COMPLETION_FORMATS: List[str] = [
    "move_only",      # Just "UP", "DOWN", etc.
    "move_with_reason", # "Move RIGHT because..."
    "full_analysis"    # Detailed reasoning + move
]
"""Different completion format styles."""

# =============================================================================
# NPZ Format Specifications
# =============================================================================

# Sequential NPZ (for RNN/LSTM and RL)
NPZ_SEQUENTIAL_ARRAYS: Dict[str, Tuple[str, ...]] = {
    "states": ("timesteps", "features"),
    "actions": ("timesteps",),
    "rewards": ("timesteps",),
    "values": ("timesteps",),
    "next_states": ("timesteps", "features")
}
"""Array specifications for sequential NPZ format."""

# Spatial NPZ (for CNN)
NPZ_SPATIAL_ARRAYS: Dict[str, Tuple[str, ...]] = {
    "board_states": ("samples", "height", "width", "channels"),
    "actions": ("samples",),
    "metadata": ("samples", "metadata_features")
}
"""Array specifications for spatial NPZ format."""

# Raw Arrays NPZ (for evolutionary algorithms)
NPZ_RAW_ARRAYS: Dict[str, Tuple[str, ...]] = {
    "population": ("population_size", "individual_length"),
    "fitness_scores": ("population_size", "num_objectives"),
    "generation_history": ("num_generations", "population_size", "individual_length"),
    "metadata": ("num_generations", "metadata_features")
}
"""Array specifications for raw NPZ format (evolutionary algorithms)."""

# Evolutionary-specific NPZ extensions
NPZ_EVOLUTIONARY_ARRAYS: Dict[str, Tuple[str, ...]] = {
    "crossover_points": ("num_crossovers", "parent_indices"),
    "mutation_mask": ("population_size", "individual_length"),
    "selection_pressure": ("num_generations",),
    "fitness_landscape": ("grid_size", "grid_size", "num_objectives"),
    "pareto_front": ("pareto_size", "num_objectives")
}
"""Extended arrays for evolutionary algorithm NPZ format."""

NPZ_ARRAY_NAMES: Dict[str, List[str]] = {
    "sequential": list(NPZ_SEQUENTIAL_ARRAYS.keys()),
    "spatial": list(NPZ_SPATIAL_ARRAYS.keys()),
    "raw": list(NPZ_RAW_ARRAYS.keys()),
    "evolutionary": list(NPZ_RAW_ARRAYS.keys()) + list(NPZ_EVOLUTIONARY_ARRAYS.keys())
}
"""Array names for different NPZ format types."""

# =============================================================================
# Format Validation Specifications
# =============================================================================

SUPPORTED_FILE_EXTENSIONS: Dict[str, Set[str]] = {
    "csv": {".csv"},
    "jsonl": {".jsonl", ".json"},
    "npz": {".npz"},
    "model": {".pth", ".pkl", ".joblib", ".onnx"},
}
"""Valid file extensions for each format type."""

GRID_SIZE_VALIDATION: Dict[str, Tuple[int, int]] = {
    "min_max": (5, 50),  # Minimum 5x5, maximum 50x50
    "recommended": (8, 20),  # Recommended range
    "default": (10, 10)  # Default size
}
"""Grid size validation ranges."""

FORMAT_COMPATIBILITY: Dict[str, Dict[str, bool]] = {
    "csv": {
        "tree_models": True,
        "simple_mlp": True,
        "cnn": False,
        "rnn": False,
        "evolutionary": False
    },
    "npz_sequential": {
        "tree_models": False,
        "simple_mlp": False,
        "cnn": False,
        "rnn": True,
        "evolutionary": False
    },
    "npz_spatial": {
        "tree_models": False,
        "simple_mlp": False,
        "cnn": True,
        "rnn": False,
        "evolutionary": False
    },
    "npz_raw": {
        "tree_models": False,
        "simple_mlp": False,
        "cnn": False,
        "rnn": False,
        "evolutionary": True
    },
    "jsonl": {
        "llm_finetuning": True,
        "prompt_engineering": True,
        "reasoning_analysis": True
    }
}
"""Format compatibility matrix for different use cases."""
