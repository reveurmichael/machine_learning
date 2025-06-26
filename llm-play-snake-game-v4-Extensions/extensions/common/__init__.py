"""
Extensions Common Utilities Package

This package provides shared utilities for all Snake Game AI extensions, following the
principle that each extension + common folder = standalone unit.

Design Philosophy:
- Single source of truth for shared functionality
- Grid-size agnostic implementations
- Educational value with comprehensive documentation
- Support for multiple data formats and validation
- Factory patterns for creating appropriate components

Key Components:
- Configuration management (config/)
- Data validation utilities (validation/)
- Dataset loading and processing
- Path management utilities
- Factory pattern implementations
- CSV schema for supervised learning

Usage:
    from extensions.common import get_dataset_path, validate_dataset_format
    from extensions.common.config import DEFAULT_LEARNING_RATE
    from extensions.common.validation import DatasetValidator
"""

# Core path utilities - fundamental for all extensions
from .path_utils import (
    ensure_project_root,
    get_extension_path,
    get_dataset_path,
    get_model_path,
    validate_path_structure,
    setup_extension_environment
)

# Configuration utilities
from .config import (
    # ML constants
    DEFAULT_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    DEFAULT_BATCH_SIZE,
    
    # Training defaults
    TRAIN_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
    
    # Dataset formats
    CSV_FEATURE_SCHEMA,
    JSONL_SCHEMA,
    NPZ_SEQUENTIAL_SCHEMA,
    
    # Path constants
    DATASET_PATH_TEMPLATE,
    MODEL_PATH_TEMPLATE,
    
    # Validation rules
    MIN_GRID_SIZE,
    MAX_GRID_SIZE,
    
    # Model registry
    ModelType,
    MODEL_CAPABILITIES,
    MODEL_METADATA
)

# Data processing utilities
from .csv_schema import (
    generate_csv_schema,
    create_csv_row,
    TabularFeatureExtractor,
    validate_csv_schema,
    FEATURE_COUNT,
    COLUMN_NAMES
)

from .dataset_loader import (
    DatasetLoader,
    load_dataset_for_training,
    prepare_features_and_targets,
    split_dataset,
    validate_dataset_compatibility
)

# Factory utilities
from .factory_utils import (
    BaseFactory,
    AgentFactory,
    ModelFactory,
    ValidatorFactory,
    create_appropriate_validator,
    register_factory_type
)

# Validation utilities - comprehensive validation system
from .validation import (
    # Main validation functions
    validate_dataset_format,
    validate_model_output,
    validate_path_structure as validate_paths,
    validate_config_access,
    validate_extension_compliance,
    
    # Validation types
    ValidationResult,
    ValidationLevel,
    ValidationReport,
    
    # Specific validators
    DatasetValidator,
    ModelValidator,
    PathValidator,
    ConfigValidator,
    ExtensionValidator
)

# Extension management utilities
from .extension_utils import (
    ExtensionType,
    ExtensionConfig,
    ExtensionEnvironment,
    create_extension_environment,
    get_extension_info,
    list_available_extensions,
    validate_extension_standalone
)

# Testing utilities  
from .test_utils import (
    TestRunner,
    TestCase,
    create_test_environment,
    cleanup_test_environment,
    run_common_utilities_tests
)

# Version information
__version__ = "1.0.0"
__author__ = "Snake Game AI Project"
__description__ = "Common utilities for Snake Game AI extensions"

# Export categories for convenience
__all__ = [
    # Path management
    'ensure_project_root',
    'get_extension_path', 
    'get_dataset_path',
    'get_model_path',
    'validate_path_structure',
    'setup_extension_environment',
    
    # Configuration constants
    'DEFAULT_LEARNING_RATE',
    'EARLY_STOPPING_PATIENCE',
    'DEFAULT_BATCH_SIZE',
    'TRAIN_SPLIT_RATIO',
    'VALIDATION_SPLIT_RATIO',
    'TEST_SPLIT_RATIO',
    'CSV_FEATURE_SCHEMA',
    'JSONL_SCHEMA',
    'NPZ_SEQUENTIAL_SCHEMA',
    'DATASET_PATH_TEMPLATE',
    'MODEL_PATH_TEMPLATE',
    'MIN_GRID_SIZE',
    'MAX_GRID_SIZE',
    'ModelType',
    'MODEL_CAPABILITIES',
    'MODEL_METADATA',
    
    # Data processing
    'generate_csv_schema',
    'create_csv_row',
    'TabularFeatureExtractor',
    'validate_csv_schema',
    'FEATURE_COUNT',
    'COLUMN_NAMES',
    'DatasetLoader',
    'load_dataset_for_training',
    'prepare_features_and_targets',
    'split_dataset',
    'validate_dataset_compatibility',
    
    # Factory patterns
    'BaseFactory',
    'AgentFactory',
    'ModelFactory',
    'ValidatorFactory',
    'create_appropriate_validator',
    'register_factory_type',
    
    # Validation system
    'validate_dataset_format',
    'validate_model_output',
    'validate_paths',
    'validate_config_access',
    'validate_extension_compliance',
    'ValidationResult',
    'ValidationLevel',
    'ValidationReport',
    'DatasetValidator',
    'ModelValidator',
    'PathValidator',
    'ConfigValidator',
    'ExtensionValidator',
    
    # Extension management
    'ExtensionType',
    'ExtensionConfig',
    'ExtensionEnvironment',
    'create_extension_environment',
    'get_extension_info',
    'list_available_extensions',
    'validate_extension_standalone',
    
    # Testing utilities
    'TestRunner',
    'TestCase',
    'create_test_environment',
    'cleanup_test_environment',
    'run_common_utilities_tests',
] 