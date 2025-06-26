"""
Model Validation Utilities for Snake Game AI Extensions.

This module provides comprehensive model validation for different formats
and performance thresholds used across all extension types.

Design Patterns:
- Strategy Pattern: Different validation strategies for different model types
- Template Method Pattern: Common validation workflow with model-specific details
- Observer Pattern: Performance monitoring and threshold validation

Educational Value:
Demonstrates how to build robust model validation systems that ensure
model quality and performance compliance across different ML paradigms.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import pickle
import json

# Import configuration constants
from ..config.model_registry import MODEL_METADATA, ModelType
from ..config.validation_rules import DATA_QUALITY_THRESHOLDS

# Validation result classes
from .validation_types import ValidationResult, ValidationLevel

# =============================================================================
# Base Model Validator
# =============================================================================

class BaseModelValidator(ABC):
    """
    Abstract base class for model validators.
    
    Design Pattern: Template Method Pattern
    Purpose: Define common validation workflow with model-specific customization
    
    Educational Note:
    This demonstrates how to use the Template Method pattern to provide
    consistent validation behavior while allowing model-specific validation logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"ModelValidator.{name}")
    
    def validate(self, model_path: Path, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Template method for model validation.
        
        This method defines the common workflow:
        1. Validate model file exists and is readable
        2. Load and inspect model
        3. Validate model structure
        4. Check performance metrics
        5. Return comprehensive result
        """
        try:
            # Step 1: File validation
            file_result = self._validate_file(model_path)
            if not file_result.is_valid:
                return file_result
            
            # Step 2: Load model
            model = self._load_model(model_path)
            
            # Step 3: Structure validation
            structure_result = self._validate_structure(model, metadata)
            if not structure_result.is_valid:
                return structure_result
            
            # Step 4: Performance validation (if metadata provided)
            if metadata and "performance" in metadata:
                performance_result = self._validate_performance(metadata["performance"])
                if not performance_result.is_valid:
                    return performance_result
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Model validation passed for {self.name}",
                details={"model_path": str(model_path), "model_info": self._get_model_info(model)}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Model validation failed: {str(e)}",
                details={"model_path": str(model_path), "error": str(e)}
            )
    
    def _validate_file(self, model_path: Path) -> ValidationResult:
        """Validate that model file exists and is accessible."""
        if not model_path.exists():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Model file not found: {model_path}",
                suggestion="Check file path and ensure model file exists"
            )
        
        if not model_path.is_file():
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Path is not a file: {model_path}",
                suggestion="Provide path to a model file, not a directory"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Model file validation passed"
        )
    
    @abstractmethod
    def _load_model(self, model_path: Path) -> Any:
        """Load model from file (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_structure(self, model: Any, metadata: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate model structure (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _validate_performance(self, performance_metrics: Dict[str, float]) -> ValidationResult:
        """Validate model performance (format-specific implementation)."""
        pass
    
    @abstractmethod
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get model information (format-specific implementation)."""
        pass

# =============================================================================
# PyTorch Model Validator
# =============================================================================

class PyTorchModelValidator(BaseModelValidator):
    """
    Validator for PyTorch models (.pth, .pt files).
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle PyTorch-specific validation logic
    
    Educational Note:
    Shows how to validate neural network models and their architectures.
    """
    
    def __init__(self):
        super().__init__("PyTorch")
    
    def _load_model(self, model_path: Path) -> Any:
        """Load PyTorch model."""
        try:
            import torch
            return torch.load(model_path, map_location='cpu')
        except ImportError:
            raise ValueError("PyTorch not installed - cannot validate PyTorch models")
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {e}")
    
    def _validate_structure(self, model: Any, metadata: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate PyTorch model structure."""
        try:
            import torch.nn as nn
            
            # Check if model is a state dict or model instance
            if isinstance(model, dict):
                # State dict validation
                if not model:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="PyTorch state dict is empty",
                        suggestion="Ensure model was saved properly"
                    )
                
                # Check for common layer types
                layer_types = set()
                for key in model.keys():
                    if 'weight' in key or 'bias' in key:
                        layer_types.add(key.split('.')[0])
                
                if not layer_types:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message="No recognizable layers found in state dict",
                        suggestion="Verify model architecture"
                    )
            
            elif isinstance(model, nn.Module):
                # Model instance validation
                total_params = sum(p.numel() for p in model.parameters())
                if total_params == 0:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Model has no parameters",
                        suggestion="Check model architecture"
                    )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="PyTorch model structure validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"PyTorch structure validation failed: {str(e)}"
            )
    
    def _validate_performance(self, performance_metrics: Dict[str, float]) -> ValidationResult:
        """Validate PyTorch model performance."""
        issues = []
        
        # Check for common performance metrics
        expected_metrics = ["accuracy", "loss", "f1_score"]
        for metric in expected_metrics:
            if metric in performance_metrics:
                value = performance_metrics[metric]
                
                # Validate metric ranges
                if metric == "accuracy" and not (0.0 <= value <= 1.0):
                    issues.append(f"Accuracy should be between 0 and 1, got {value}")
                elif metric == "loss" and value < 0:
                    issues.append(f"Loss should be non-negative, got {value}")
                elif metric == "f1_score" and not (0.0 <= value <= 1.0):
                    issues.append(f"F1 score should be between 0 and 1, got {value}")
        
        # Check performance thresholds
        thresholds = DATA_QUALITY_THRESHOLDS.get("performance_thresholds", {}).get("supervised", {})
        for metric, threshold in thresholds.items():
            if metric in performance_metrics:
                if performance_metrics[metric] < threshold:
                    issues.append(f"{metric}: {performance_metrics[metric]:.3f} < {threshold:.3f}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Performance issues: {'; '.join(issues)}",
                details={"issues": issues},
                suggestion="Consider model improvements or hyperparameter tuning"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="PyTorch performance validation passed"
        )
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get PyTorch model information."""
        try:
            import torch.nn as nn
            
            info = {"type": "PyTorch"}
            
            if isinstance(model, dict):
                info["format"] = "state_dict"
                info["num_parameters"] = len(model)
                info["layers"] = list(set(key.split('.')[0] for key in model.keys() if '.' in key))
            elif isinstance(model, nn.Module):
                info["format"] = "model_instance"
                info["num_parameters"] = sum(p.numel() for p in model.parameters())
                info["layers"] = [str(type(module).__name__) for module in model.modules()]
            
            return info
        except Exception:
            return {"type": "PyTorch", "error": "Could not extract model info"}

# =============================================================================
# Scikit-learn Model Validator
# =============================================================================

class SklearnModelValidator(BaseModelValidator):
    """
    Validator for scikit-learn models (.pkl files).
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle scikit-learn specific validation logic
    
    Educational Note:
    Shows how to validate traditional ML models and their configurations.
    """
    
    def __init__(self):
        super().__init__("Scikit-learn")
    
    def _load_model(self, model_path: Path) -> Any:
        """Load scikit-learn model from pickle."""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load scikit-learn model: {e}")
    
    def _validate_structure(self, model: Any, metadata: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate scikit-learn model structure."""
        try:
            # Check if model has required sklearn interface
            required_methods = ["fit", "predict"]
            missing_methods = [method for method in required_methods if not hasattr(model, method)]
            
            if missing_methods:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Model missing required methods: {missing_methods}",
                    suggestion="Ensure model follows scikit-learn interface"
                )
            
            # Check if model is fitted
            try:
                from sklearn.exceptions import NotFittedError
                # Try to access a fitted attribute
                if hasattr(model, 'n_features_in_'):
                    # Model is fitted
                    pass
                else:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message="Model may not be fitted",
                        suggestion="Ensure model is trained before saving"
                    )
            except (NotFittedError, AttributeError):
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Model appears to be unfitted",
                    suggestion="Train model before validation"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Scikit-learn model structure validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Scikit-learn structure validation failed: {str(e)}"
            )
    
    def _validate_performance(self, performance_metrics: Dict[str, float]) -> ValidationResult:
        """Validate scikit-learn model performance."""
        issues = []
        
        # Check for common sklearn metrics
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            if metric in performance_metrics:
                value = performance_metrics[metric]
                
                # Validate metric ranges
                if metric in ["accuracy", "precision", "recall", "f1_score"]:
                    if not (0.0 <= value <= 1.0):
                        issues.append(f"{metric} should be between 0 and 1, got {value}")
        
        # Check performance thresholds
        thresholds = DATA_QUALITY_THRESHOLDS.get("performance_thresholds", {}).get("supervised", {})
        for metric, threshold in thresholds.items():
            if metric in performance_metrics:
                if performance_metrics[metric] < threshold:
                    issues.append(f"{metric}: {performance_metrics[metric]:.3f} < {threshold:.3f}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Performance issues: {'; '.join(issues)}",
                details={"issues": issues},
                suggestion="Consider model improvements or feature engineering"
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Scikit-learn performance validation passed"
        )
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get scikit-learn model information."""
        try:
            info = {
                "type": "Scikit-learn",
                "model_class": type(model).__name__
            }
            
            # Get model parameters
            if hasattr(model, 'get_params'):
                info["parameters"] = model.get_params()
            
            # Get feature information if available
            if hasattr(model, 'n_features_in_'):
                info["n_features"] = model.n_features_in_
            
            if hasattr(model, 'feature_names_in_'):
                info["feature_names"] = list(model.feature_names_in_)
            
            return info
        except Exception:
            return {"type": "Scikit-learn", "error": "Could not extract model info"}

# =============================================================================
# ONNX Model Validator
# =============================================================================

class ONNXModelValidator(BaseModelValidator):
    """
    Validator for ONNX models (.onnx files).
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle ONNX-specific validation logic
    
    Educational Note:
    Shows how to validate cross-platform model formats for deployment.
    """
    
    def __init__(self):
        super().__init__("ONNX")
    
    def _load_model(self, model_path: Path) -> Any:
        """Load ONNX model."""
        try:
            import onnx
            return onnx.load(str(model_path))
        except ImportError:
            raise ValueError("ONNX not installed - cannot validate ONNX models")
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {e}")
    
    def _validate_structure(self, model: Any, metadata: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate ONNX model structure."""
        try:
            import onnx
            
            # Check model validity
            try:
                onnx.checker.check_model(model)
            except onnx.checker.ValidationError as e:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"ONNX model validation failed: {str(e)}",
                    suggestion="Fix model structure issues"
                )
            
            # Check graph structure
            graph = model.graph
            if not graph.node:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="ONNX model has no computational nodes",
                    suggestion="Ensure model contains operations"
                )
            
            # Check inputs and outputs
            if not graph.input:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="ONNX model has no inputs defined",
                    suggestion="Define model inputs"
                )
            
            if not graph.output:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="ONNX model has no outputs defined",
                    suggestion="Define model outputs"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="ONNX model structure validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"ONNX structure validation failed: {str(e)}"
            )
    
    def _validate_performance(self, performance_metrics: Dict[str, float]) -> ValidationResult:
        """Validate ONNX model performance."""
        # ONNX models typically don't have embedded performance metrics
        # Performance validation would be based on external benchmarking
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="ONNX performance validation passed (external metrics required)"
        )
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get ONNX model information."""
        try:
            graph = model.graph
            
            info = {
                "type": "ONNX",
                "ir_version": model.ir_version,
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "num_nodes": len(graph.node),
                "num_inputs": len(graph.input),
                "num_outputs": len(graph.output),
                "input_names": [inp.name for inp in graph.input],
                "output_names": [out.name for out in graph.output]
            }
            
            return info
        except Exception:
            return {"type": "ONNX", "error": "Could not extract model info"}

# =============================================================================
# Factory and High-Level Functions
# =============================================================================

class ModelValidatorFactory:
    """
    Factory for creating model validators.
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate validator based on file format
    
    Educational Note:
    Demonstrates how to use the Factory pattern to encapsulate
    validator creation logic based on file extension.
    """
    
    _validators = {
        ".pth": PyTorchModelValidator,
        ".pt": PyTorchModelValidator,
        ".pkl": SklearnModelValidator,
        ".pickle": SklearnModelValidator,
        ".onnx": ONNXModelValidator,
    }
    
    @classmethod
    def create_validator(cls, model_path: Path) -> BaseModelValidator:
        """Create appropriate validator based on file extension."""
        extension = model_path.suffix.lower()
        validator_class = cls._validators.get(extension)
        
        if validator_class is None:
            supported_formats = list(cls._validators.keys())
            raise ValueError(f"Unsupported format: {extension}. "
                           f"Supported formats: {supported_formats}")
        
        return validator_class()

# High-level validation functions
def validate_model_output(
    model_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    High-level function to validate model format and performance.
    
    Args:
        model_path: Path to model file
        metadata: Optional metadata with performance metrics
    
    Returns:
        ValidationResult with validation outcome
    
    Educational Note:
    This function demonstrates how to provide a simple interface
    that hides the complexity of the factory and strategy patterns.
    """
    model_path = Path(model_path)
    
    try:
        validator = ModelValidatorFactory.create_validator(model_path)
        return validator.validate(model_path, metadata)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.CRITICAL,
            message=f"Model validation failed: {str(e)}",
            details={"model_path": str(model_path), "error": str(e)}
        )

def validate_performance_thresholds(
    metrics: Dict[str, float],
    extension_type: str
) -> ValidationResult:
    """
    Validate that performance metrics meet minimum thresholds.
    
    Args:
        metrics: Dictionary of performance metrics
        extension_type: Extension type for threshold lookup
    
    Returns:
        ValidationResult with validation outcome
    """
    thresholds = DATA_QUALITY_THRESHOLDS.get("performance_thresholds", {}).get(extension_type, {})
    
    if not thresholds:
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"No performance thresholds defined for {extension_type}"
        )
    
    failures = []
    for metric, threshold in thresholds.items():
        if metric in metrics:
            if metrics[metric] < threshold:
                failures.append(f"{metric}: {metrics[metric]:.3f} < {threshold:.3f}")
    
    if failures:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.WARNING,
            message=f"Performance below thresholds: {'; '.join(failures)}",
            suggestion="Consider model improvements or hyperparameter tuning"
        )
    
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="Performance validation passed"
    )

# Convenience class exports
ModelValidator = BaseModelValidator
PerformanceValidator = PyTorchModelValidator  # Alias for most common use case 