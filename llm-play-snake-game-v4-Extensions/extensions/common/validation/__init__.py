"""
Validation Package for Snake Game AI Extensions.

This package provides comprehensive validation utilities for configurations,
datasets, models, and path structures across all extension types.

Design Patterns:
- Strategy Pattern: Different validation strategies for different data types
- Chain of Responsibility: Chained validators for complex validation logic
- Command Pattern: Encapsulated validation commands
- Composite Pattern: Hierarchical validation rules

Educational Value:
Demonstrates how to build robust validation systems that provide clear
error messages and maintain data quality across a complex ML pipeline.

Package Structure:
- dataset_validator: Dataset quality and format validation
- model_validator: Model format and performance validation
- path_validator: Path structure and naming convention validation
- config_validator: Configuration compliance validation

Usage:
    from extensions.common.validation import (
        validate_dataset_format,
        validate_model_output,
        validate_path_structure,
        validate_config_access
    )
"""

# Core validation classes and utilities
from .dataset_validator import (
    DatasetValidator,
    DataQualityValidator,
    validate_dataset_format,
    validate_dataset_quality
)

from .model_validator import (
    ModelValidator,
    PerformanceValidator,
    validate_model_output,
    validate_performance_thresholds
)

from .path_validator import (
    PathValidator,
    NamingConventionValidator,
    validate_path_structure,
    validate_extension_naming
)

from .config_validator import (
    ConfigValidator,
    ImportValidator,
    validate_config_access,
    validate_import_restrictions
)

# High-level validation functions
from .extension_validator import (
    ExtensionValidator,
    ValidationReport,
    ValidationResult,
    ValidationLevel,
    validate_extension_compliance
)

__all__ = [
    # Dataset validation
    "DatasetValidator",
    "DataQualityValidator", 
    "validate_dataset_format",
    "validate_dataset_quality",
    
    # Model validation
    "ModelValidator",
    "PerformanceValidator",
    "validate_model_output",
    "validate_performance_thresholds",
    
    # Path validation
    "PathValidator", 
    "NamingConventionValidator",
    "validate_path_structure",
    "validate_extension_naming",
    
    # Configuration validation
    "ConfigValidator",
    "ImportValidator", 
    "validate_config_access",
    "validate_import_restrictions",
    
    # Comprehensive validation
    "ExtensionValidator",
    "ValidationReport",
    "ValidationResult", 
    "ValidationLevel",
    "validate_extension_compliance",
]

# Version info
__version__ = "0.1.0"
__author__ = "Snake Game AI Project" 