"""
Dataset Validation Utilities for Snake Game AI Extensions.

This module provides comprehensive dataset validation for different formats
(CSV, JSONL, NPZ) used across all extension types.

Design Patterns:
- Strategy Pattern: Different validation strategies for different formats
- Template Method Pattern: Common validation workflow with format-specific details
- Chain of Responsibility: Sequential validation checks

Educational Value:
Demonstrates how to build robust data validation systems that ensure
data quality and format compliance across different machine learning pipelines.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# Import configuration constants
from ..config.dataset_formats import (
    CSV_COLUMN_NAMES, JSONL_REQUIRED_FIELDS, NPZ_ARRAY_NAMES
)
from ..config.validation_rules import DATA_QUALITY_THRESHOLDS

# Validation result classes
from .validation_types import ValidationResult, ValidationLevel

# =============================================================================
# Base Dataset Validator
# =============================================================================

class BaseDatasetValidator(ABC):
    """
    Abstract base class for dataset validators.
    
    Design Pattern: Template Method Pattern
    Purpose: Define common validation workflow with format-specific customization
    
    Educational Note:
    This demonstrates how to use the Template Method pattern to provide
    consistent validation behavior while allowing format-specific validation logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"DatasetValidator.{name}")
    
    def validate(self, data_path: Path) -> ValidationResult:
        """
        Template method for dataset validation.
        
        This method defines the common workflow:
        1. Validate file exists and is readable
        2. Load and parse data
        3. Validate format structure
        4. Check data quality
        5. Return comprehensive result
        """
        try:
            # Step 1: File validation
            file_result = self._validate_file(data_path)
            if not file_result.is_valid:
                return file_result
            
            # Step 2: Load data
            data = self._load_data(data_path)
            
            # Step 3: Format validation
            format_result = self._validate_format(data)
            if not format_result.is_valid:
                return format_result
            
            # Step 4: Quality validation
            quality_result = self._validate_quality(data)
            if not quality_result.is_valid:
                return quality_result
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Dataset validation passed for {self.name}",
                details={"file_path": str(data_path), "data_size": self._get_data_size(data)}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Dataset validation failed: {str(e)}",
                details={"file_path": str(data_path), "error": str(e)}
            )
    
    def _validate_file(self, data_path: Path) -> ValidationResult:
        """Validate that file exists and is accessible."""
        if not data_path.exists():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Dataset file not found: {data_path}",
                suggestion="Check file path and ensure file exists"
            )
        
        if not data_path.is_file():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Path is not a file: {data_path}",
                suggestion="Provide path to a file, not a directory"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="File validation passed"
        )
    
    @abstractmethod
    def _load_data(self, data_path: Path) -> Any:
        """Load data from file (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_format(self, data: Any) -> ValidationResult:
        """Validate data format (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_quality(self, data: Any) -> ValidationResult:
        """Validate data quality (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _get_data_size(self, data: Any) -> int:
        """Get size of dataset (format-specific implementation)."""
        pass

# =============================================================================
# CSV Dataset Validator
# =============================================================================

class CSVDatasetValidator(BaseDatasetValidator):
    """
    Validator for CSV datasets with 16-feature tabular format.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle CSV-specific validation logic
    
    Educational Note:
    Shows how to validate tabular data while maintaining grid-size agnostic design.
    """
    
    def __init__(self):
        super().__init__("CSV")
    
    def _load_data(self, data_path: Path) -> pd.DataFrame:
        """Load CSV data using pandas."""
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def _validate_format(self, data: pd.DataFrame) -> ValidationResult:
        """Validate CSV format structure."""
        # Check required columns
        missing_cols = set(CSV_COLUMN_NAMES) - set(data.columns)
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing required columns: {missing_cols}",
                details={"expected_columns": CSV_COLUMN_NAMES, "found_columns": list(data.columns)},
                suggestion="Ensure dataset includes all required columns"
            )
        
        # Check data types
        type_errors = []
        feature_columns = [col for col in CSV_COLUMN_NAMES if col not in ["game_id", "step_in_game", "target_move"]]
        
        for col in feature_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    type_errors.append(f"Column {col} should be numeric")
        
        if type_errors:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Data type errors: {'; '.join(type_errors)}",
                suggestion="Convert feature columns to numeric types"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="CSV format validation passed"
        )
    
    def _validate_quality(self, data: pd.DataFrame) -> ValidationResult:
        """Validate CSV data quality."""
        issues = []
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        critical_missing = missing_counts[missing_counts > 0]
        if not critical_missing.empty:
            max_missing_pct = DATA_QUALITY_THRESHOLDS.get("max_missing_percentage", 0.1)
            for col, count in critical_missing.items():
                pct = count / len(data)
                if pct > max_missing_pct:
                    issues.append(f"Column {col} has {pct:.1%} missing values (threshold: {max_missing_pct:.1%})")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        max_duplicates_pct = DATA_QUALITY_THRESHOLDS.get("max_duplicate_percentage", 0.05)
        if duplicates / len(data) > max_duplicates_pct:
            issues.append(f"Dataset has {duplicates / len(data):.1%} duplicates (threshold: {max_duplicates_pct:.1%})")
        
        # Check minimum size
        min_samples = DATA_QUALITY_THRESHOLDS.get("min_samples", 100)
        if len(data) < min_samples:
            issues.append(f"Dataset too small: {len(data)} samples (minimum: {min_samples})")
        
        # Check feature distributions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() == 0:
                issues.append(f"Column {col} has zero variance")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Data quality issues found: {'; '.join(issues)}",
                details={"issues": issues},
                suggestion="Consider data cleaning or augmentation"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"CSV quality validation passed ({len(data)} samples)"
        )
    
    def _get_data_size(self, data: pd.DataFrame) -> int:
        """Get number of rows in DataFrame."""
        return len(data)

# =============================================================================
# JSONL Dataset Validator
# =============================================================================

class JSONLDatasetValidator(BaseDatasetValidator):
    """
    Validator for JSONL datasets with prompt-completion pairs.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle JSONL-specific validation for LLM fine-tuning
    
    Educational Note:
    Shows how to validate sequential JSON data while maintaining
    memory efficiency for large datasets.
    """
    
    def __init__(self):
        super().__init__("JSONL")
    
    def _load_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL data line by line."""
        data = []
        try:
            with open(data_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            return data
        except Exception as e:
            raise ValueError(f"Failed to load JSONL file: {e}")
    
    def _validate_format(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate JSONL format structure."""
        if not data:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="JSONL file is empty",
                suggestion="Ensure file contains valid JSON records"
            )
        
        # Check required fields in sample records
        sample_size = min(10, len(data))
        missing_fields_issues = []
        
        for i, record in enumerate(data[:sample_size]):
            missing_fields = set(JSONL_REQUIRED_FIELDS) - set(record.keys())
            if missing_fields:
                missing_fields_issues.append(f"Record {i}: missing {missing_fields}")
        
        if missing_fields_issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing required fields: {'; '.join(missing_fields_issues[:3])}...",
                details={"required_fields": JSONL_REQUIRED_FIELDS, "issues": missing_fields_issues},
                suggestion="Ensure all records have required fields"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="JSONL format validation passed"
        )
    
    def _validate_quality(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate JSONL data quality."""
        issues = []
        
        # Check field consistency
        first_keys = set(data[0].keys())
        inconsistent_records = []
        for i, record in enumerate(data[1:], 1):
            if set(record.keys()) != first_keys:
                inconsistent_records.append(i)
                if len(inconsistent_records) > 10:  # Limit reporting
                    break
        
        if inconsistent_records:
            issues.append(f"Inconsistent field structure in records: {inconsistent_records[:10]}")
        
        # Check for empty content
        empty_content = 0
        for record in data:
            if "prompt" in record and not record["prompt"].strip():
                empty_content += 1
            if "completion" in record and not record["completion"].strip():
                empty_content += 1
        
        if empty_content > 0:
            issues.append(f"{empty_content} records have empty prompt or completion")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"JSONL quality issues: {'; '.join(issues)}",
                details={"issues": issues},
                suggestion="Clean data and ensure consistent record structure"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"JSONL quality validation passed ({len(data)} records)"
        )
    
    def _get_data_size(self, data: List[Dict[str, Any]]) -> int:
        """Get number of records."""
        return len(data)

# =============================================================================
# NPZ Dataset Validator
# =============================================================================

class NPZDatasetValidator(BaseDatasetValidator):
    """
    Validator for NPZ datasets with numerical arrays.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle NPZ-specific validation for neural network training
    
    Educational Note:
    Demonstrates efficient validation of numerical data for deep learning
    while maintaining format flexibility.
    """
    
    def __init__(self):
        super().__init__("NPZ")
    
    def _load_data(self, data_path: Path) -> Dict[str, np.ndarray]:
        """Load NPZ data as dictionary of arrays."""
        try:
            return dict(np.load(data_path))
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {e}")
    
    def _validate_format(self, data: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate NPZ format structure."""
        # Check required arrays
        missing_arrays = set(NPZ_ARRAY_NAMES) - set(data.keys())
        if missing_arrays:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing required arrays: {missing_arrays}",
                details={"expected_arrays": NPZ_ARRAY_NAMES, "found_arrays": list(data.keys())},
                suggestion="Ensure NPZ file contains all required arrays"
            )
        
        # Check array shapes are consistent
        if "features" in data and "targets" in data:
            if data["features"].shape[0] != data["targets"].shape[0]:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Features and targets must have same number of samples",
                    details={
                        "features_shape": data["features"].shape,
                        "targets_shape": data["targets"].shape
                    },
                    suggestion="Ensure features and targets arrays have matching first dimension"
                )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="NPZ format validation passed"
        )
    
    def _validate_quality(self, data: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate NPZ data quality."""
        issues = []
        
        for array_name, array_data in data.items():
            # Check for NaN values
            if np.isnan(array_data).any():
                nan_count = np.isnan(array_data).sum()
                issues.append(f"Array {array_name} contains {nan_count} NaN values")
            
            # Check for infinite values
            if np.isinf(array_data).any():
                inf_count = np.isinf(array_data).sum()
                issues.append(f"Array {array_name} contains {inf_count} infinite values")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"NPZ quality issues: {'; '.join(issues)}",
                details={"issues": issues},
                suggestion="Handle NaN and infinite values before training"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="NPZ quality validation passed"
        )
    
    def _get_data_size(self, data: Dict[str, np.ndarray]) -> int:
        """Get number of samples in arrays."""
        if "features" in data:
            return data["features"].shape[0]
        elif "states" in data:
            return data["states"].shape[0]
        else:
            return 0

# =============================================================================
# Factory and High-Level Functions
# =============================================================================

class DatasetValidatorFactory:
    """
    Factory for creating dataset validators.
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate validator based on file format
    
    Educational Note:
    Demonstrates how to use the Factory pattern to encapsulate
    validator creation logic based on file extension.
    """
    
    _validators = {
        ".csv": CSVDatasetValidator,
        ".jsonl": JSONLDatasetValidator,
        ".npz": NPZDatasetValidator,
    }
    
    @classmethod
    def create_validator(cls, file_path: Path) -> BaseDatasetValidator:
        """Create appropriate validator based on file extension."""
        extension = file_path.suffix.lower()
        validator_class = cls._validators.get(extension)
        
        if validator_class is None:
            supported_formats = list(cls._validators.keys())
            raise ValueError(f"Unsupported format: {extension}. "
                           f"Supported formats: {supported_formats}")
        
        return validator_class()

# High-level validation functions
def validate_dataset_format(file_path: Union[str, Path]) -> ValidationResult:
    """
    High-level function to validate dataset format.
    
    Args:
        file_path: Path to dataset file
    
    Returns:
        ValidationResult with validation outcome
    
    Educational Note:
    This function demonstrates how to provide a simple interface
    that hides the complexity of the factory and strategy patterns.
    """
    file_path = Path(file_path)
    
    try:
        validator = DatasetValidatorFactory.create_validator(file_path)
        return validator.validate(file_path)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.CRITICAL,
            message=f"Dataset validation failed: {str(e)}",
            details={"file_path": str(file_path), "error": str(e)}
        )

def validate_dataset_quality(data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> ValidationResult:
    """
    Validate data quality for different data types.
    
    Args:
        data: Dataset to validate
    
    Returns:
        ValidationResult with quality assessment
    """
    if isinstance(data, pd.DataFrame):
        validator = CSVDatasetValidator()
        return validator._validate_quality(data)
    elif isinstance(data, list):
        validator = JSONLDatasetValidator()
        return validator._validate_quality(data)
    elif isinstance(data, dict):
        validator = NPZDatasetValidator()
        return validator._validate_quality(data)
    else:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Unsupported data type for quality validation: {type(data)}"
        )

# Convenience class exports
DatasetValidator = BaseDatasetValidator
DataQualityValidator = CSVDatasetValidator  # Alias for backward compatibility 