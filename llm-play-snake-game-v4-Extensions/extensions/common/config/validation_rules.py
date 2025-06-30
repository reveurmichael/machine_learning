"""
Validation Rules Configuration for Snake Game AI Extensions.

This module defines validation rules and constraints used across
all extensions for consistent data validation.

Following final-decision-10.md SUPREME_RULES:
- Simple, OOP-based utilities with inheritance support
- Simple logging with print() statements only
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Supports experimentation and flexibility

Design Philosophy:
- Lightweight validation rules that can be extended
- Grid-size agnostic validation constraints
- Simple error handling and reporting
"""

from typing import Dict, Any

# =============================================================================
# CSV Validation Rules
# =============================================================================

CSV_VALIDATION_RULES: Dict[str, Any] = {
    # Column validation
    "required_columns": [
        "game_id", "step_in_game", "head_x", "head_y", "apple_x", "apple_y",
        "snake_length", "apple_dir_up", "apple_dir_down", "apple_dir_left", 
        "apple_dir_right", "danger_straight", "danger_left", "danger_right",
        "free_space_up", "free_space_down", "free_space_left", "free_space_right",
        "target_move"
    ],
    
    # Data type validation
    "integer_columns": [
        "game_id", "step_in_game", "head_x", "head_y", "apple_x", "apple_y",
        "snake_length", "free_space_up", "free_space_down", "free_space_left", 
        "free_space_right"
    ],
    
    "binary_columns": [
        "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
        "danger_straight", "danger_left", "danger_right"
    ],
    
    # Value constraints
    "valid_moves": {"UP", "DOWN", "LEFT", "RIGHT"},
    "binary_values": {0, 1},
    
    # Range validation (grid-size agnostic minimums)
    "min_values": {
        "snake_length": 1,
        "free_space_up": 0,
        "free_space_down": 0, 
        "free_space_left": 0,
        "free_space_right": 0
    },
    
    # Logical consistency rules
    "consistency_rules": [
        "snake_length >= 1",
        "apple_dir_up + apple_dir_down + apple_dir_left + apple_dir_right >= 1"
    ]
}

# =============================================================================
# JSONL Validation Rules  
# =============================================================================

JSONL_VALIDATION_RULES: Dict[str, Any] = {
    "required_keys": {"prompt", "completion"},
    "optional_keys": {"game_id", "step_in_game", "metadata"},
    "max_prompt_length": 1000000,
    "max_completion_length": 100000,
    "min_prompt_length": 1,
    "min_completion_length": 1
}

# =============================================================================
# NPZ Validation Rules
# =============================================================================

NPZ_VALIDATION_RULES: Dict[str, Any] = {
    "required_arrays": {"data", "targets"},
    "optional_arrays": {"metadata", "timestamps"},
    "max_array_size": 1000000,  # 1M elements max
    "supported_dtypes": {"float32", "float64", "int32", "int64", "bool"}
}

# =============================================================================
# Grid Size Validation Rules
# =============================================================================

GRID_SIZE_RULES: Dict[str, Any] = {
    "min_grid_size": 5,
    "max_grid_size": 50,
    "common_grid_sizes": {8, 10, 12, 16, 20},
    "default_grid_size": 10
}

# =============================================================================
# Extension Validation Rules
# =============================================================================

EXTENSION_VALIDATION_RULES: Dict[str, Any] = {
    "required_files": ["__init__.py", "game_logic.py", "game_manager.py"],
    "version_pattern": r"v\d+\.\d{2}",
    "name_pattern": r"^[a-z]+(-[a-z]+)*-v\d+\.\d{2}$"
}

# =============================================================================
# Simple Validation Functions (SUPREME_RULES Compliant)
# =============================================================================

def validate_grid_size(grid_size: int) -> bool:
    """Simple grid size validation following SUPREME_RULES."""
    min_size = GRID_SIZE_RULES["min_grid_size"]
    max_size = GRID_SIZE_RULES["max_grid_size"]
    
    if grid_size < min_size or grid_size > max_size:
        print(f"[ValidationRules] Invalid grid size: {grid_size} (must be {min_size}-{max_size})")  # Simple logging
        return False
    
    print(f"[ValidationRules] Grid size {grid_size} is valid")  # Simple logging  
    return True

def validate_move(move: str) -> bool:
    """Simple move validation following SUPREME_RULES."""
    valid_moves = CSV_VALIDATION_RULES["valid_moves"]
    
    if move not in valid_moves:
        print(f"[ValidationRules] Invalid move: {move} (must be one of {valid_moves})")  # Simple logging
        return False
    
    print(f"[ValidationRules] Move {move} is valid")  # Simple logging
    return True

def get_validation_rules(data_type: str) -> Dict[str, Any]:
    """Get validation rules for specific data type."""
    rules_map = {
        "csv": CSV_VALIDATION_RULES,
        "jsonl": JSONL_VALIDATION_RULES, 
        "npz": NPZ_VALIDATION_RULES,
        "grid": GRID_SIZE_RULES,
        "extension": EXTENSION_VALIDATION_RULES
    }
    
    rules = rules_map.get(data_type.lower(), {})
    print(f"[ValidationRules] Retrieved {data_type} validation rules")  # Simple logging
    return rules 