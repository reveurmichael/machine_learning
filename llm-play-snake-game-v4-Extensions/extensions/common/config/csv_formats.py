"""
CSV Format Specifications for Snake Game AI Extensions

This module defines the standardized CSV format specifications used
across all extensions, following forward-looking architecture principles.

Design Philosophy:
- Forward-looking: No legacy compatibility, clean and self-contained
- Grid-size agnostic: Works with any board size (8x8, 10x10, 12x12, etc.)
- Single responsibility: Only CSV format definitions
- Educational clarity: Clear, documented constants

Reference: docs/extensions-guideline/forward-looking-architecture.md
"""

from typing import List, Dict, Any

# =============================================================================
# CSV Schema Definition (16-Feature Tabular)
# =============================================================================

# Core feature columns (16 features)
CSV_FEATURE_COLUMNS: List[str] = [
    # Position features (4)
    'head_x', 'head_y', 'apple_x', 'apple_y',
    
    # Game state features (1)
    'snake_length',
    
    # Apple direction features (4) - binary flags
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
    
    # Danger detection features (3) - collision risk
    'danger_straight', 'danger_left', 'danger_right',
    
    # Free space features (4) - available moves
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'
]

# Metadata columns (2 columns)
CSV_METADATA_COLUMNS: List[str] = [
    'game_id',        # Unique game session identifier
    'step_in_game'    # Step number within the game
]

# Target column (1 column)
CSV_TARGET_COLUMN: str = 'target_move'  # The move taken (UP, DOWN, LEFT, RIGHT)

# Complete column set (2 metadata + 16 features + 1 target = 19 columns)
CSV_ALL_COLUMNS: List[str] = CSV_METADATA_COLUMNS + CSV_FEATURE_COLUMNS + [CSV_TARGET_COLUMN]

# Feature categorization for validation and processing
CSV_POSITION_FEATURES: List[str] = ['head_x', 'head_y', 'apple_x', 'apple_y']
CSV_BINARY_FEATURES: List[str] = [
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
    'danger_straight', 'danger_left', 'danger_right'
]
CSV_COUNT_FEATURES: List[str] = [
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'
]

# Column data types for validation
CSV_COLUMN_TYPES: Dict[str, str] = {
    # Metadata - integers
    'game_id': 'int64',
    'step_in_game': 'int64',
    
    # Position features - integers (coordinates)
    'head_x': 'int64',
    'head_y': 'int64', 
    'apple_x': 'int64',
    'apple_y': 'int64',
    
    # Game state - integer
    'snake_length': 'int64',
    
    # Binary features - integers (0 or 1)
    'apple_dir_up': 'int64',
    'apple_dir_down': 'int64',
    'apple_dir_left': 'int64',
    'apple_dir_right': 'int64',
    'danger_straight': 'int64',
    'danger_left': 'int64',
    'danger_right': 'int64',
    
    # Count features - integers
    'free_space_up': 'int64',
    'free_space_down': 'int64',
    'free_space_left': 'int64',
    'free_space_right': 'int64',
    
    # Target - string/categorical
    'target_move': 'string'
}

# =============================================================================
# Validation Rules
# =============================================================================

# Grid size validation
MIN_GRID_SIZE: int = 4
MAX_GRID_SIZE: int = 50
SUPPORTED_GRID_SIZES: List[int] = list(range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))

# Move validation
VALID_MOVES: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
VALID_MOVE_SET: set = set(VALID_MOVES)

# Feature value ranges for validation
FEATURE_RANGES: Dict[str, Dict[str, int]] = {
    'head_x': {'min': 0, 'max': MAX_GRID_SIZE - 1},
    'head_y': {'min': 0, 'max': MAX_GRID_SIZE - 1},
    'apple_x': {'min': 0, 'max': MAX_GRID_SIZE - 1},
    'apple_y': {'min': 0, 'max': MAX_GRID_SIZE - 1},
    'snake_length': {'min': 1, 'max': MAX_GRID_SIZE * MAX_GRID_SIZE},
    'apple_dir_up': {'min': 0, 'max': 1},
    'apple_dir_down': {'min': 0, 'max': 1},
    'apple_dir_left': {'min': 0, 'max': 1},
    'apple_dir_right': {'min': 0, 'max': 1},
    'danger_straight': {'min': 0, 'max': 1},
    'danger_left': {'min': 0, 'max': 1},
    'danger_right': {'min': 0, 'max': 1},
    'free_space_up': {'min': 0, 'max': MAX_GRID_SIZE * MAX_GRID_SIZE},
    'free_space_down': {'min': 0, 'max': MAX_GRID_SIZE * MAX_GRID_SIZE},
    'free_space_left': {'min': 0, 'max': MAX_GRID_SIZE * MAX_GRID_SIZE},
    'free_space_right': {'min': 0, 'max': MAX_GRID_SIZE * MAX_GRID_SIZE}
}

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core CSV schema
    "CSV_FEATURE_COLUMNS",
    "CSV_METADATA_COLUMNS", 
    "CSV_TARGET_COLUMN",
    "CSV_ALL_COLUMNS",
    "CSV_POSITION_FEATURES",
    "CSV_BINARY_FEATURES", 
    "CSV_COUNT_FEATURES",
    "CSV_COLUMN_TYPES",
    
    # Validation
    "VALID_MOVES",
    "VALID_MOVE_SET",
    "FEATURE_RANGES",
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE",
    "SUPPORTED_GRID_SIZES",
] 