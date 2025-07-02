"""
Dataset Format Specifications for Snake Game AI Extensions.

This module defines the standardized data format specifications used
across all extensions, following the data format decision guide.

Design Philosophy:
- Consistent schema across all extensions
- Grid-size agnostic feature design  
- Format-specific optimizations for different use cases
- Educational clarity and documentation

Reference: docs/extensions-guideline/data-format-decision-guide.md
"""

from typing import List, Dict, Any

# =============================================================================
# CSV Format Specifications (16-Feature Tabular)
# =============================================================================

# Fixed 16-feature schema that works for any grid size
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

# Metadata columns for tracking and analysis
CSV_METADATA_COLUMNS: List[str] = [
    'game_id',        # Unique game session identifier
    'step_in_game'    # Step number within the game
]

# Target column for supervised learning
CSV_TARGET_COLUMNS: List[str] = [
    'target_move'     # The move taken (UP, DOWN, LEFT, RIGHT)
]

# Complete column set (2 metadata + 16 features + 1 target = 19 columns)
CSV_BASIC_COLUMNS: List[str] = CSV_METADATA_COLUMNS + CSV_FEATURE_COLUMNS + CSV_TARGET_COLUMNS

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
# JSONL Format Specifications (Language-Rich)
# =============================================================================

# Required fields for JSONL format (heuristics-v0.04 specialty)
JSONL_REQUIRED_FIELDS: List[str] = [
    'prompt',       # Game state description for LLM
    'completion'    # Natural language explanation/reasoning
]

# Optional fields for JSONL format
JSONL_OPTIONAL_FIELDS: List[str] = [
    'game_id',          # Link to original game session
    'step_in_game',     # Step number for sequence tracking
    'algorithm',        # Which heuristic algorithm generated this
    'move',             # The actual move taken
    'confidence',       # Algorithm confidence (if available)
    'metadata'          # Additional context information
]

# JSONL schema validation
JSONL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": JSONL_REQUIRED_FIELDS,
    "properties": {
        "prompt": {"type": "string", "minLength": 10},
        "completion": {"type": "string", "minLength": 20},
        "game_id": {"type": "integer", "minimum": 1},
        "step_in_game": {"type": "integer", "minimum": 0},
        "algorithm": {"type": "string"},
        "move": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "metadata": {"type": "object"}
    }
}

# =============================================================================
# NPZ Format Specifications
# =============================================================================

# Sequential NPZ arrays (for RNN/RL)
NPZ_SEQUENTIAL_ARRAYS: List[str] = [
    'states',        # Shape: (timesteps, features)
    'actions',       # Shape: (timesteps,)
    'rewards',       # Shape: (timesteps,)
    'next_states',   # Shape: (timesteps, features)
    'dones'          # Shape: (timesteps,) - episode termination flags
]

# Spatial NPZ arrays (for CNN)
NPZ_SPATIAL_ARRAYS: List[str] = [
    'boards',        # Shape: (samples, height, width, channels)
    'targets',       # Shape: (samples,) - move indices
    'metadata'       # Shape: (samples, metadata_dim)
]

# Raw NPZ arrays (for evolutionary algorithms)
NPZ_RAW_ARRAYS: List[str] = [
    'population',           # Shape: (population_size, individual_length)
    'fitness_scores',       # Shape: (population_size, num_objectives)
    'generation_history',   # Shape: (num_generations, population_size, individual_length)
    'selection_pressure',   # Shape: (num_generations,)
    'diversity_metrics'     # Shape: (num_generations,)
]

# =============================================================================
# Format Selection Guidelines
# =============================================================================

FORMAT_USE_CASES: Dict[str, Dict[str, Any]] = {
    "csv": {
        "best_for": ["Tree models", "Simple MLPs", "Traditional ML"],
        "algorithms": ["XGBoost", "LightGBM", "Random Forest", "SVM"],
        "grid_size_support": "Universal",
        "pros": ["Fast loading", "Human readable", "Small size"],
        "cons": ["Limited to tabular data", "No sequence information"]
    },
    "jsonl": {
        "best_for": ["LLM fine-tuning", "Language models"],
        "algorithms": ["GPT", "Claude", "LLaMA", "T5"],
        "grid_size_support": "Universal", 
        "pros": ["Rich explanations", "Human readable", "Flexible"],
        "cons": ["Large file size", "Requires processing"]
    },
    "npz": {
        "best_for": ["Deep learning", "Sequential models", "Spatial models"],
        "algorithms": ["CNN", "RNN", "LSTM", "RL agents"],
        "grid_size_support": "Universal",
        "pros": ["Efficient storage", "Native numpy", "Multiple arrays"],
        "cons": ["Binary format", "Requires numpy"]
    }
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
# File Extensions by Type (for compatibility with existing utils)
# =============================================================================

COMMON_DATASET_EXTENSIONS = {
    "tabular": {".csv"},
    "language": {".jsonl", ".json"},
    "arrays": {".npz", ".npy"},
}
"""Common file extensions for different data types."""

# =============================================================================
# Legacy Compatibility Aliases (for existing utils)
# =============================================================================

# Column aliases for backward compatibility
CSV_COLUMN_NAMES = CSV_BASIC_COLUMNS
CSV_FEATURE_NAMES = CSV_FEATURE_COLUMNS
CSV_TARGET_COLUMN = CSV_TARGET_COLUMNS[0]  # Extract single string from list
CSV_EXPECTED_COLUMNS = CSV_BASIC_COLUMNS

# Feature type aliases
CSV_POSITION_FEATURES = ['head_x', 'head_y', 'apple_x', 'apple_y']
CSV_BINARY_FEATURES = [
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
    'danger_straight', 'danger_left', 'danger_right'
]
CSV_COUNT_FEATURES = [
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'
]

# Move validation alias
CSV_VALID_MOVES = VALID_MOVE_SET

# JSONL aliases
JSONL_BASIC_KEYS = set(JSONL_REQUIRED_FIELDS)

# =============================================================================
# Export Configuration
# =============================================================================

__all__ = [
    # CSV format
    "CSV_FEATURE_COLUMNS",
    "CSV_METADATA_COLUMNS", 
    "CSV_TARGET_COLUMNS",
    "CSV_BASIC_COLUMNS",
    "CSV_COLUMN_TYPES",
    
    # JSONL format
    "JSONL_REQUIRED_FIELDS",
    "JSONL_OPTIONAL_FIELDS",
    "JSONL_SCHEMA",
    
    # NPZ format
    "NPZ_SEQUENTIAL_ARRAYS",
    "NPZ_SPATIAL_ARRAYS",
    "NPZ_RAW_ARRAYS",
    
    # Guidelines
    "FORMAT_USE_CASES",
    
    # Validation
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE",
    "SUPPORTED_GRID_SIZES",
    "VALID_MOVES",
    "VALID_MOVE_SET",
    "FEATURE_RANGES",
    
    # File extensions (compatibility)
    "COMMON_DATASET_EXTENSIONS",
    
    # Legacy compatibility aliases
    "CSV_COLUMN_NAMES",
    "CSV_FEATURE_NAMES", 
    "CSV_TARGET_COLUMN",
    "CSV_EXPECTED_COLUMNS",
    "CSV_POSITION_FEATURES",
    "CSV_BINARY_FEATURES",
    "CSV_COUNT_FEATURES",
    "CSV_VALID_MOVES",
    "JSONL_BASIC_KEYS"
]

