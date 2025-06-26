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

REQUIRED_FILE_EXTENSIONS: Dict[str, Set[str]] = {
    "dataset_csv": {".csv"},
    "dataset_jsonl": {".jsonl", ".json"},
    "dataset_npz": {".npz"},
    "model_pytorch": {".pth", ".pt"},
    "model_sklearn": {".pkl", ".joblib"},
    "model_onnx": {".onnx"},
    "config": {".json", ".yaml", ".yml"},
    "log": {".log", ".txt", ".json"}
}
"""Required file extensions for different file types."""

# =============================================================================
# Version Validation (Flexible for Educational Project)
# =============================================================================

VERSION_PATTERN: str = r"^v?\d+\.\d{2}$"
"""Regex pattern for version validation.

Educational Note (SUPREME_RULE NO.3):
We should be able to add new extensions easily and try out new ideas.
Therefore, we don't restrict specific extension versions - any version
following the pattern is valid to encourage experimentation.
"""

# =============================================================================
# Dataset Quality Validation
# =============================================================================

DATASET_QUALITY_RULES: Dict[str, Any] = {
    "min_samples": {
        "csv": 100,
        "jsonl": 100,
        "npz": 100
    },
    "max_samples": {
        "csv": 1000000,
        "jsonl": 100000,
        "npz": 10000000
    },
    "min_classes": 2,
    "max_class_imbalance": 10.0,  # ratio
    "max_missing_ratio": 0.1,
    "max_duplicate_ratio": 0.05,
    "min_feature_variance": 1e-6
}
"""Dataset quality validation rules."""

# =============================================================================
# Configuration Validation
# =============================================================================

HYPERPARAMETER_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "neural_networks": {
        "learning_rate": (1e-6, 1e-1),
        "batch_size": (1, 1024),
        "epochs": (1, 10000),
        "dropout_rate": (0.0, 0.9),
        "weight_decay": (0.0, 1.0)
    },
    "tree_models": {
        "n_estimators": (1, 10000),
        "max_depth": (1, 50),
        "min_samples_split": (2, 1000),
        "min_samples_leaf": (1, 500)
    },
    "reinforcement_learning": {
        "epsilon": (0.0, 1.0),
        "gamma": (0.0, 1.0),
        "learning_rate": (1e-6, 1e-1),
        "buffer_size": (1000, 1000000)
    }
}
"""Valid ranges for hyperparameters."""

# =============================================================================
# Path Validation Rules
# =============================================================================

PATH_VALIDATION_RULES: Dict[str, str] = {
    "extension_type": r"^[a-z][a-z0-9_]*[a-z0-9]$",
    "algorithm": r"^[a-z][a-z0-9_]*[a-z0-9]$",
    "version": r"^\d+\.\d{2}$",
    "timestamp": r"^\d{8}_\d{6}$",
    "grid_size": r"^\d+$"
}
"""Regex patterns for path component validation."""

FORBIDDEN_PATH_CHARACTERS: str = r'[<>:"/\\|?*\x00-\x1f]'
"""Characters forbidden in path components."""

MAX_PATH_LENGTH: int = 260
"""Maximum total path length."""

MAX_COMPONENT_LENGTH: int = 255
"""Maximum length for individual path components."""

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

# =============================================================================
# Performance Validation Thresholds
# =============================================================================

PERFORMANCE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "accuracy": {
        "minimum": 0.5,      # Better than random
        "good": 0.7,         # Decent performance
        "excellent": 0.9     # Excellent performance
    },
    "loss": {
        "maximum": 10.0,     # Reasonable loss ceiling
        "good": 1.0,         # Good loss level
        "excellent": 0.1     # Excellent loss level
    },
    "training_time": {
        "reasonable_hours": 24,   # Maximum reasonable training time
        "fast_hours": 1,          # Fast training threshold
        "very_fast_minutes": 10   # Very fast training threshold
    }
}
"""Performance validation thresholds."""

# =============================================================================
# Resource Usage Validation
# =============================================================================

RESOURCE_LIMITS: Dict[str, Dict[str, Any]] = {
    "memory": {
        "dataset_mb": 1000,      # 1GB dataset limit
        "model_mb": 500,         # 500MB model limit
        "training_mb": 4000      # 4GB training memory limit
    },
    "disk": {
        "experiment_gb": 10,     # 10GB per experiment
        "total_gb": 100         # 100GB total storage
    },
    "time": {
        "max_training_hours": 48,    # 48 hour training limit
        "max_evaluation_hours": 4    # 4 hour evaluation limit
    }
}
"""Resource usage validation limits."""

# =============================================================================
# Naming Convention Validation
# =============================================================================

NAMING_PATTERNS: Dict[str, str] = {
    "agent_file": r"^agent_[a-z][a-z0-9_]*\.py$",
    "class_name": r"^[A-Z][A-Za-z0-9]*Agent$",
    "function_name": r"^[a-z][a-z0-9_]*[a-z0-9]$",
    "constant_name": r"^[A-Z][A-Z0-9_]*[A-Z0-9]$",
    "variable_name": r"^[a-z][a-z0-9_]*[a-z0-9]$"
}
"""Naming convention validation patterns."""

# =============================================================================
# Configuration Compliance Rules
# =============================================================================

COMPLIANCE_RULES: Dict[str, Dict[str, Any]] = {
    "directory_structure": {
        "required_dirs": ["agents", "scripts"],  # v0.02+
        "optional_dirs": ["dashboard", "utils", "config"],
        "forbidden_dirs": ["legacy", "deprecated"]
    },
    "file_organization": {
        "agents_in_subdir": True,     # v0.02+ requirement
        "max_files_per_dir": 20,
        "min_documentation_ratio": 0.3  # 30% of lines should be comments/docs
    },
    "import_compliance": {
        "no_relative_imports": True,
        "no_circular_imports": True,
        "use_type_hints": True
    }
}
"""Extension compliance validation rules.""" 