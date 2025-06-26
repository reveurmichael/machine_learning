"""
Path Validation Utilities for Snake Game AI Extensions.

This module provides comprehensive path validation for directory structures,
naming conventions, and compliance with extension guidelines.

Design Patterns:
- Strategy Pattern: Different validation strategies for different path types
- Chain of Responsibility: Sequential path validation checks
- Command Pattern: Encapsulated validation commands

Educational Value:
Demonstrates how to build robust path validation systems that ensure
consistency and compliance across a complex project structure.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Pattern
from abc import ABC, abstractmethod
from pathlib import Path
import re
import logging

# Import configuration constants
from ..config.path_constants import (
    DATASET_PATH_PATTERN, MODEL_PATH_PATTERN, EXTENSION_PATH_PATTERN,
    REQUIRED_DIRECTORIES, FORBIDDEN_PATTERNS
)
from ..config.validation_rules import NAMING_CONVENTIONS

# Validation result classes
from .validation_types import ValidationResult, ValidationLevel

# =============================================================================
# Base Path Validator
# =============================================================================

class BasePathValidator(ABC):
    """
    Abstract base class for path validators.
    
    Design Pattern: Template Method Pattern
    Purpose: Define common validation workflow with path-specific customization
    
    Educational Note:
    This demonstrates how to use the Template Method pattern to provide
    consistent validation behavior while allowing path-specific validation logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"PathValidator.{name}")
    
    def validate(self, path: Path, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Template method for path validation.
        
        This method defines the common workflow:
        1. Validate path format and structure
        2. Check naming conventions
        3. Verify directory compliance
        4. Check for forbidden patterns
        5. Return comprehensive result
        """
        try:
            context = context or {}
            
            # Step 1: Format validation
            format_result = self._validate_format(path, context)
            if not format_result.is_valid:
                return format_result
            
            # Step 2: Naming convention validation
            naming_result = self._validate_naming(path, context)
            if not naming_result.is_valid:
                return naming_result
            
            # Step 3: Directory structure validation
            structure_result = self._validate_structure(path, context)
            if not structure_result.is_valid:
                return structure_result
            
            # Step 4: Forbidden patterns check
            forbidden_result = self._check_forbidden_patterns(path)
            if not forbidden_result.is_valid:
                return forbidden_result
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Path validation passed for {self.name}",
                details={"path": str(path), "validator": self.name}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Path validation failed: {str(e)}",
                details={"path": str(path), "error": str(e)}
            )
    
    @abstractmethod
    def _validate_format(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate path format (path-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_naming(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate naming conventions (path-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_structure(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate directory structure (path-specific implementation)."""
        pass
    
    def _check_forbidden_patterns(self, path: Path) -> ValidationResult:
        """Check for forbidden patterns in path."""
        path_str = str(path).lower()
        
        for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, path_str):
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Path contains forbidden pattern '{pattern_name}': {path}",
                    suggestion=f"Avoid using {pattern_name} in paths"
                )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="No forbidden patterns found"
        )

# =============================================================================
# Dataset Path Validator
# =============================================================================

class DatasetPathValidator(BasePathValidator):
    """
    Validator for dataset paths following standardized structure.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle dataset-specific path validation logic
    
    Educational Note:
    Shows how to validate complex hierarchical path structures while
    maintaining grid-size agnostic design.
    """
    
    def __init__(self):
        super().__init__("Dataset")
        self.path_pattern = re.compile(DATASET_PATH_PATTERN)
    
    def _validate_format(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate dataset path format."""
        path_str = str(path)
        
        if not self.path_pattern.match(path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Dataset path doesn't match required pattern: {path}",
                details={
                    "expected_pattern": DATASET_PATH_PATTERN,
                    "actual_path": path_str
                },
                suggestion="Use standardized dataset path format: logs/extensions/datasets/grid-size-N/extension_vX.XX_timestamp/"
            )
        
        # Extract and validate components
        try:
            match = self.path_pattern.match(path_str)
            if match:
                grid_size = int(match.group(1))
                extension = match.group(2)
                version = match.group(3)
                timestamp = match.group(4)
                
                # Validate grid size
                if grid_size < 8 or grid_size > 20:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Unusual grid size: {grid_size}",
                        suggestion="Typically use grid sizes between 8 and 20"
                    )
                
                # Validate extension type
                valid_extensions = ["heuristics", "supervised", "reinforcement", "evolutionary", "llm"]
                if extension not in valid_extensions:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Unknown extension type: {extension}",
                        details={"valid_extensions": valid_extensions}
                    )
        
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Failed to parse dataset path components: {e}"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Dataset path format validation passed"
        )
    
    def _validate_naming(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate dataset naming conventions."""
        # Check timestamp format
        timestamp_pattern = r"\d{8}_\d{6}"
        path_str = str(path)
        
        if not re.search(timestamp_pattern, path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Dataset path missing valid timestamp format",
                suggestion="Use timestamp format: YYYYMMDD_HHMMSS"
            )
        
        # Check version format
        version_pattern = r"_v\d+\.\d+"
        if not re.search(version_pattern, path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Dataset path missing valid version format",
                suggestion="Use version format: _vX.XX"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Dataset naming convention validation passed"
        )
    
    def _validate_structure(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate dataset directory structure."""
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Dataset directory does not exist: {path}",
                suggestion="Create dataset directory before validation"
            )
        
        # Check for required subdirectories
        required_subdirs = ["processed_data"]  # Could be extended based on extension type
        missing_dirs = []
        
        for subdir in required_subdirs:
            subdir_path = path / subdir
            if not subdir_path.exists():
                missing_dirs.append(subdir)
        
        if missing_dirs:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Missing required subdirectories: {missing_dirs}",
                suggestion="Create required subdirectories for dataset organization"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Dataset directory structure validation passed"
        )

# =============================================================================
# Model Path Validator
# =============================================================================

class ModelPathValidator(BasePathValidator):
    """
    Validator for model paths following standardized structure.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle model-specific path validation logic
    
    Educational Note:
    Shows how to validate model storage paths while maintaining
    consistency with dataset path structure.
    """
    
    def __init__(self):
        super().__init__("Model")
        self.path_pattern = re.compile(MODEL_PATH_PATTERN)
    
    def _validate_format(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate model path format."""
        path_str = str(path)
        
        if not self.path_pattern.match(path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Model path doesn't match required pattern: {path}",
                details={
                    "expected_pattern": MODEL_PATH_PATTERN,
                    "actual_path": path_str
                },
                suggestion="Use standardized model path format: logs/extensions/models/grid-size-N/extension_vX.XX_timestamp/"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Model path format validation passed"
        )
    
    def _validate_naming(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate model naming conventions."""
        # Similar to dataset naming but for models
        timestamp_pattern = r"\d{8}_\d{6}"
        path_str = str(path)
        
        if not re.search(timestamp_pattern, path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Model path missing valid timestamp format",
                suggestion="Use timestamp format: YYYYMMDD_HHMMSS"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Model naming convention validation passed"
        )
    
    def _validate_structure(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate model directory structure."""
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Model directory does not exist: {path}",
                suggestion="Create model directory before validation"
            )
        
        # Check for model artifacts
        required_subdirs = ["model_artifacts"]
        missing_dirs = []
        
        for subdir in required_subdirs:
            subdir_path = path / subdir
            if not subdir_path.exists():
                missing_dirs.append(subdir)
        
        if missing_dirs:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Missing required subdirectories: {missing_dirs}",
                suggestion="Create required subdirectories for model organization"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Model directory structure validation passed"
        )

# =============================================================================
# Extension Path Validator
# =============================================================================

class ExtensionPathValidator(BasePathValidator):
    """
    Validator for extension paths and internal structure.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle extension-specific path validation logic
    
    Educational Note:
    Shows how to validate complex extension directory structures
    while ensuring compliance with naming conventions.
    """
    
    def __init__(self):
        super().__init__("Extension")
        self.path_pattern = re.compile(EXTENSION_PATH_PATTERN)
    
    def _validate_format(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate extension path format."""
        path_str = str(path)
        
        if not self.path_pattern.match(path_str):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Extension path doesn't match required pattern: {path}",
                details={
                    "expected_pattern": EXTENSION_PATH_PATTERN,
                    "actual_path": path_str
                },
                suggestion="Use format: extensions/algorithm-vX.XX/"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Extension path format validation passed"
        )
    
    def _validate_naming(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate extension naming conventions."""
        path_name = path.name
        
        # Check for proper algorithm-version format
        naming_pattern = r"^[a-z]+(-[a-z]+)*-v\d+\.\d+$"
        if not re.match(naming_pattern, path_name):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Extension name doesn't follow naming convention: {path_name}",
                suggestion="Use format: algorithm-vX.XX (e.g., heuristics-v0.03)"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Extension naming convention validation passed"
        )
    
    def _validate_structure(self, path: Path, context: Dict[str, Any]) -> ValidationResult:
        """Validate extension directory structure."""
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Extension directory does not exist: {path}",
                suggestion="Create extension directory before validation"
            )
        
        # Check for required files and directories based on version
        version_match = re.search(r"v(\d+)\.(\d+)", path.name)
        if version_match:
            major, minor = int(version_match.group(1)), int(version_match.group(2))
            
            # Version-specific requirements
            if major == 0 and minor >= 2:
                # v0.02+ requires agents directory
                agents_dir = path / "agents"
                if not agents_dir.exists():
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="v0.02+ extensions must have 'agents' directory",
                        suggestion="Create agents directory and move agent files"
                    )
            
            if major == 0 and minor >= 3:
                # v0.03+ requires dashboard and scripts directories
                required_dirs = ["dashboard", "scripts"]
                missing_dirs = []
                
                for req_dir in required_dirs:
                    if not (path / req_dir).exists():
                        missing_dirs.append(req_dir)
                
                if missing_dirs:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"v0.03+ extensions missing directories: {missing_dirs}",
                        suggestion="Create required directories for v0.03+ compliance"
                    )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Extension directory structure validation passed"
        )

# =============================================================================
# Naming Convention Validator
# =============================================================================

class NamingConventionValidator:
    """
    Specialized validator for naming conventions across different file types.
    
    Design Pattern: Strategy Pattern
    Purpose: Centralized naming convention validation
    
    Educational Note:
    Shows how to implement comprehensive naming validation that ensures
    consistency across a large codebase.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("NamingConventionValidator")
    
    def validate_file_naming(self, file_path: Path, file_type: str) -> ValidationResult:
        """Validate file naming conventions."""
        conventions = NAMING_CONVENTIONS.get(file_type, {})
        if not conventions:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"No naming conventions defined for {file_type}"
            )
        
        file_name = file_path.name
        pattern = conventions.get("pattern")
        
        if pattern and not re.match(pattern, file_name):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"File name '{file_name}' doesn't match {file_type} convention",
                details={
                    "expected_pattern": pattern,
                    "examples": conventions.get("examples", [])
                },
                suggestion=f"Follow {file_type} naming convention"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"File naming convention validation passed for {file_type}"
        )
    
    def validate_agent_naming(self, agent_file: Path) -> ValidationResult:
        """Validate agent file naming specifically."""
        file_name = agent_file.name
        
        # Agent files should follow agent_{algorithm}.py pattern
        agent_pattern = r"^agent_[a-z][a-z0-9_]*\.py$"
        if not re.match(agent_pattern, file_name):
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Agent file '{file_name}' doesn't follow naming convention",
                suggestion="Use format: agent_{algorithm}.py (e.g., agent_bfs.py)"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Agent naming convention validation passed"
        )

# =============================================================================
# Factory and High-Level Functions
# =============================================================================

class PathValidatorFactory:
    """
    Factory for creating path validators.
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate validator based on path type
    
    Educational Note:
    Demonstrates how to use the Factory pattern to encapsulate
    validator creation logic based on path characteristics.
    """
    
    @classmethod
    def create_validator(cls, path_type: str) -> BasePathValidator:
        """Create appropriate validator based on path type."""
        validators = {
            "dataset": DatasetPathValidator,
            "model": ModelPathValidator,
            "extension": ExtensionPathValidator,
        }
        
        validator_class = validators.get(path_type)
        if validator_class is None:
            supported_types = list(validators.keys())
            raise ValueError(f"Unsupported path type: {path_type}. "
                           f"Supported types: {supported_types}")
        
        return validator_class()

# High-level validation functions
def validate_path_structure(
    path: Union[str, Path],
    path_type: str,
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    High-level function to validate path structure.
    
    Args:
        path: Path to validate
        path_type: Type of path (dataset, model, extension)
        context: Optional context for validation
    
    Returns:
        ValidationResult with validation outcome
    
    Educational Note:
    This function demonstrates how to provide a simple interface
    that hides the complexity of the factory and strategy patterns.
    """
    path = Path(path)
    
    try:
        validator = PathValidatorFactory.create_validator(path_type)
        return validator.validate(path, context)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.CRITICAL,
            message=f"Path validation failed: {str(e)}",
            details={"path": str(path), "path_type": path_type, "error": str(e)}
        )

def validate_extension_naming(extension_path: Union[str, Path]) -> ValidationResult:
    """
    Validate extension naming conventions.
    
    Args:
        extension_path: Path to extension directory
    
    Returns:
        ValidationResult with validation outcome
    """
    extension_path = Path(extension_path)
    validator = ExtensionPathValidator()
    return validator._validate_naming(extension_path, {})

# Convenience class exports
PathValidator = BasePathValidator 