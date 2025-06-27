"""
Validation Rules and Thresholds for Snake Game AI Extensions.

This module defines validation rules, thresholds, and supported configurations
used across all extensions for ensuring data quality and configuration compliance.

Design Pattern: Validator Pattern
- Centralized validation logic
- Configurable validation thresholds
- Type-safe validation rules

Educational Value:
Demonstrates how to implement robust validation systems with clear
separation between validation rules and business logic.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import re

# =============================================================================
# Grid Size Validation
# =============================================================================

MIN_GRID_SIZE: int = 5
"""Minimum supported grid size."""

MAX_GRID_SIZE: int = 50
"""Maximum supported grid size."""

RECOMMENDED_GRID_SIZES: Set[int] = {8, 10, 12, 16, 20}
"""Recommended grid sizes for optimal performance."""

DEFAULT_GRID_SIZE: int = 10
"""Default grid size for new experiments."""

# =============================================================================
# Algorithm Support (Flexible for Educational Project)
# =============================================================================

# Educational Note (SUPREME_RULE NO.3):
# We should be able to add new extensions easily and try out new ideas.
# Therefore, we don't restrict specific algorithms - extensions can implement
# any algorithm following the naming conventions (agent_algorithm.py pattern)

# =============================================================================
# File Extension Validation
# =============================================================================

# TODO: make sure this follow the rules set in the folder "./docs/extensions-guideline/"
REQUIRED_FILE_EXTENSIONS: Dict[str, Set[str]] = {
    "dataset_csv": {".csv"},
    "dataset_jsonl": {".jsonl"},
    "dataset_npz": {".npz"},
    "model_pytorch": {".pth", ".pt"},
    "model_sklearn": {".pkl", ".joblib"},
    "model_onnx": {".onnx"},
}
"""Required file extensions for different file types."""

# =============================================================================
# Import Validation Rules
# =============================================================================

FORBIDDEN_IMPORT_PATTERNS: Dict[str, List[str]] = {
    "cross_extension": [
        r"from\s+extensions\.heuristics_v\d+_\d+",
        r"from\s+extensions\.supervised_v\d+_\d+",
        r"from\s+extensions\.reinforcement_v\d+_\d+",
        r"import\s+.*heuristics_v\d+_\d+",
        r"import\s+.*supervised_v\d+_\d+",
        r"import\s+.*reinforcement_v\d+_\d+"
    ],
    "llm_constants": [
        r"from\s+config\.llm_constants",
        r"from\s+config\.prompt_templates"
    ]
}
"""Forbidden import patterns for different extension types."""

# LLM constants whitelist (only these extension prefixes can import LLM constants)
LLM_WHITELIST_EXTENSIONS: List[str] = [
    "agentic-llms", "llm", "vision-language-model"
]
"""Extension prefixes allowed to import LLM constants.

Educational Note:
According to config.md guidelines, only extensions whose folder names START WITH
these prefixes may import from config.llm_constants or config.prompt_templates.
This prevents LLM-specific configuration pollution in general-purpose extensions.

Examples of allowed extensions:
- agentic-llms-v0.02
- llm-finetune-v0.03
- vision-language-model-v0.01

Examples of forbidden extensions:
- heuristics-v0.03 (general purpose)
- supervised-v0.02 (general purpose)
- reinforcement-v0.02 (general purpose)
"""

# =============================================================================
# Data Format Validation
# =============================================================================

CSV_VALIDATION_RULES: Dict[str, Any] = {
    "required_columns": 19,  # 2 metadata + 16 features + 1 target
    "feature_columns": 16,
    "binary_features": {
        "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
        "danger_straight", "danger_left", "danger_right"
    },
    "count_features": {
        "free_space_up", "free_space_down", "free_space_left", "free_space_right"
    },
    "position_features": {
        "head_x", "head_y", "apple_x", "apple_y"
    },
    "valid_moves": {"UP", "DOWN", "LEFT", "RIGHT"}
}
"""CSV format validation rules."""

JSONL_VALIDATION_RULES: Dict[str, Any] = {
    "required_keys": {"prompt", "completion"},
    "optional_keys": {
        "game_id", "step_in_game", "algorithm", "reasoning", 
        "confidence", "metadata"
    },
    "max_prompt_length": 2048,
    "max_completion_length": 1024
}
"""JSONL format validation rules."""

NPZ_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    "sequential": {
        "required_arrays": {"states", "actions"},
        "optional_arrays": {"rewards", "values", "next_states"},
        "min_timesteps": 10
    },
    "spatial": {
        "required_arrays": {"board_states", "actions"},
        "optional_arrays": {"metadata"},
        "required_dimensions": 4  # (samples, height, width, channels)
    },
    "evolutionary": {
        "required_arrays": {"population", "fitness_scores"},
        "optional_arrays": {"generation_history", "metadata"},
        "min_population_size": 10
    }
}
"""NPZ format validation rules for different types."""


