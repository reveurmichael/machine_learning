"""
Extension-specific configuration constants.

This package contains configuration constants shared across multiple extensions
but not used by the universal ROOT/config/ constants.

Design Philosophy:
- Single Source of Truth: Shared constants prevent duplication
- Clear Boundaries: Extension-specific vs universal constants
- Type Safety: All constants are properly typed
- Educational Value: Clear separation of concerns

Modules:
- ml_constants: Machine learning hyperparameters and thresholds
- training_defaults: Default training configurations  
- dataset_formats: Data format specifications
- path_constants: Directory path templates
- validation_rules: Validation thresholds and rules
- model_registry: Model type definitions and metadata

Usage:
    from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
    from extensions.common.config.training_defaults import EARLY_STOPPING_PATIENCE
    from extensions.common.config.dataset_formats import CSV_COLUMN_NAMES
"""

from .ml_constants import *
from .training_defaults import *
from .dataset_formats import *
from .path_constants import *
from .validation_rules import *
from .model_registry import *

__all__ = [
    # ML Constants
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE", 
    "DEFAULT_EPOCHS",
    "SUPPORTED_OPTIMIZERS",
    
    # Training Defaults
    "EARLY_STOPPING_PATIENCE",
    "VALIDATION_SPLIT_RATIO",
    "TEST_SPLIT_RATIO",
    "RANDOM_SEED",
    
    # Dataset Formats
    "CSV_COLUMN_NAMES",
    "CSV_FEATURE_NAMES", 
    "JSONL_REQUIRED_KEYS",
    "NPZ_ARRAY_NAMES",
    
    # Path Constants
    "DATASET_PATH_TEMPLATE",
    "MODEL_PATH_TEMPLATE",
    "LOG_PATH_TEMPLATE",
    
    # Validation Rules
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE", 
    "REQUIRED_FILE_EXTENSIONS",
    
    # Model Registry
    "SUPPORTED_MODEL_TYPES",
    "MODEL_CONFIG_SCHEMAS",
    "MODEL_EXPORT_FORMATS",
] 