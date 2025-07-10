"""
Validation Rules for Snake Game AI Extensions.

This module defines validation rules and constraints used across
all extensions to ensure data quality and consistency.

Design Philosophy:
- Simple, practical validation rules
- Clear error messages for educational value
- Extensible validation patterns
- Performance-conscious validation

Reference: docs/extensions-guideline/final-decision.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from typing import Dict, List, Any, Tuple, Optional
import re

# =============================================================================
# Grid Size Validation
# =============================================================================

MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 50

def validate_grid_size(grid_size: int) -> Tuple[bool, str]:
    """Validate grid size parameter."""
    if not isinstance(grid_size, int):
        return False, f"Grid size must be integer, got {type(grid_size)}"
    
    if grid_size < MIN_GRID_SIZE:
        return False, f"Grid size must be at least {MIN_GRID_SIZE}, got {grid_size}"
    
    if grid_size > MAX_GRID_SIZE:
        return False, f"Grid size must be at most {MAX_GRID_SIZE}, got {grid_size}"
    
    return True, "Valid grid size"

# =============================================================================
# Move Validation
# =============================================================================

VALID_MOVES = {"UP", "DOWN", "LEFT", "RIGHT"}

def validate_move(move: str) -> Tuple[bool, str]:
    """Validate move direction."""
    if not isinstance(move, str):
        return False, f"Move must be string, got {type(move)}"
    
    if move not in VALID_MOVES:
        return False, f"Invalid move '{move}', must be one of {VALID_MOVES}"
    
    return True, "Valid move"

# =============================================================================
# File Path Validation
# =============================================================================

VALID_EXTENSIONS = {".csv", ".jsonl", ".npz", ".json", ".pth", ".pkl"}

def validate_file_extension(filepath: str, expected_extensions: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate file extension."""
    import os
    
    if not isinstance(filepath, str):
        return False, f"Filepath must be string, got {type(filepath)}"
    
    _, ext = os.path.splitext(filepath)
    
    if expected_extensions is None:
        expected_extensions = list(VALID_EXTENSIONS)
    
    if ext.lower() not in [e.lower() for e in expected_extensions]:
        return False, f"Invalid extension '{ext}', expected one of {expected_extensions}"
    
    return True, "Valid file extension"

# =============================================================================
# Timestamp Validation
# =============================================================================

TIMESTAMP_PATTERN = re.compile(r'^\d{8}_\d{6}$')

def validate_timestamp(timestamp: str) -> Tuple[bool, str]:
    """Validate timestamp format (YYYYMMDD_HHMMSS)."""
    if not isinstance(timestamp, str):
        return False, f"Timestamp must be string, got {type(timestamp)}"
    
    if not TIMESTAMP_PATTERN.match(timestamp):
        return False, f"Invalid timestamp format '{timestamp}', expected YYYYMMDD_HHMMSS"
    
    return True, "Valid timestamp"

# =============================================================================
# Dataset Format Validation
# =============================================================================

def validate_csv_columns(columns: List[str], expected_columns: List[str]) -> Tuple[bool, str]:
    """Validate CSV column names."""
    missing = set(expected_columns) - set(columns)
    extra = set(columns) - set(expected_columns)
    
    if missing:
        return False, f"Missing required columns: {sorted(missing)}"
    
    if extra:
        return False, f"Unexpected columns: {sorted(extra)}"
    
    return True, "Valid CSV columns"

def validate_jsonl_record(record: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
    """Validate JSONL record structure."""
    if not isinstance(record, dict):
        return False, f"JSONL record must be dict, got {type(record)}"
    
    missing = set(required_fields) - set(record.keys())
    
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    return True, "Valid JSONL record"

# =============================================================================
# Game State Validation
# =============================================================================

def validate_position(x: int, y: int, grid_size: int) -> Tuple[bool, str]:
    """Validate position coordinates."""
    if not isinstance(x, int) or not isinstance(y, int):
        return False, f"Coordinates must be integers, got x={type(x)}, y={type(y)}"
    
    if x < 0 or x >= grid_size:
        return False, f"X coordinate {x} out of bounds [0, {grid_size-1}]"
    
    if y < 0 or y >= grid_size:
        return False, f"Y coordinate {y} out of bounds [0, {grid_size-1}]"
    
    return True, "Valid position"

def validate_snake_length(length: int, grid_size: int) -> Tuple[bool, str]:
    """Validate snake length."""
    if not isinstance(length, int):
        return False, f"Snake length must be integer, got {type(length)}"
    
    if length < 1:
        return False, f"Snake length must be at least 1, got {length}"
    
    max_length = grid_size * grid_size
    if length > max_length:
        return False, f"Snake length {length} exceeds grid capacity {max_length}"
    
    return True, "Valid snake length"

# =============================================================================
# Binary Feature Validation
# =============================================================================

def validate_binary_feature(value: Any, feature_name: str) -> Tuple[bool, str]:
    """Validate binary feature (0 or 1)."""
    if not isinstance(value, (int, float)):
        return False, f"{feature_name} must be numeric, got {type(value)}"
    
    if value not in {0, 1}:
        return False, f"{feature_name} must be 0 or 1, got {value}"
    
    return True, f"Valid {feature_name}"

# =============================================================================
# Count Feature Validation
# =============================================================================

def validate_count_feature(value: Any, feature_name: str, max_count: int) -> Tuple[bool, str]:
    """Validate count feature (non-negative integer)."""
    if not isinstance(value, (int, float)):
        return False, f"{feature_name} must be numeric, got {type(value)}"
    
    if value < 0:
        return False, f"{feature_name} must be non-negative, got {value}"
    
    if value > max_count:
        return False, f"{feature_name} {value} exceeds maximum {max_count}"
    
    return True, f"Valid {feature_name}"

# =============================================================================
# Batch Validation Functions
# =============================================================================

def validate_multiple(validators: List[Tuple[callable, tuple]]) -> Tuple[bool, List[str]]:
    """Run multiple validations and collect all error messages."""
    errors = []
    
    for validator, args in validators:
        is_valid, message = validator(*args)
        if not is_valid:
            errors.append(message)
    
    return len(errors) == 0, errors

# =============================================================================
# Export Configuration
# =============================================================================

__all__ = [
    # Grid size
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE", 
    "validate_grid_size",
    
    # Moves
    "VALID_MOVES",
    "validate_move",
    
    # Files
    "VALID_EXTENSIONS",
    "validate_file_extension",
    
    # Timestamps
    "TIMESTAMP_PATTERN",
    "validate_timestamp",
    
    # Dataset formats
    "validate_csv_columns",
    "validate_jsonl_record",
    
    # Game state
    "validate_position",
    "validate_snake_length",
    
    # Features
    "validate_binary_feature",
    "validate_count_feature",
    
    # Batch validation
    "validate_multiple",
] 