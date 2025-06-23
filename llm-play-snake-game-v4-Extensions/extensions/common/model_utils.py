"""
Common Model Utilities for Extensions
--------------------

Provides standardized model saving and loading utilities for all extensions.
Ensures cross-platform compatibility and time-proof model persistence.

Design Pattern: Template Method + Strategy Pattern
- Template method defines the saving/loading workflow
- Strategy pattern for different framework implementations
- Factory pattern for creating appropriate save/load handlers

Motivation:
- Eliminates code duplication across extensions
- Ensures consistent model metadata and directory structure
- Provides cross-platform and time-proof model persistence
- Centralizes best practices for model management

Trade-offs:
- Additional abstraction layer vs direct framework usage
- Centralized utilities vs extension-specific implementations
- Standardization vs framework-specific optimizations

Author: Snake Game Extensions
Version: 1.0
"""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


def get_model_directory(grid_size: int, framework: str = None, 
                       extension_type: str = None, version: str = None) -> Path:
    """
    Get model directory for given grid size and framework.
    
    Args:
        grid_size: Game grid size
        framework: ML framework (optional)
        extension_type: Extension type for versioned directories (optional)
        version: Extension version for versioned directories (optional)
        
    Returns:
        Path to model directory
    """
    # Use versioned directory structure if extension_type and version provided
    if extension_type and version:
        try:
            from .versioned_directory_manager import create_model_directory
            return create_model_directory(
                extension_type=extension_type,
                version=version,
                grid_size=grid_size,
                framework=framework or "pytorch"
            )
        except ImportError:
            # Fallback to legacy structure if versioned manager not available
            pass
    
    # Legacy structure for backward compatibility
    base_dir = Path("logs/extensions/models")
    grid_dir = base_dir / f"grid-size-{grid_size}"
    
    if framework:
        return grid_dir / framework.lower()
    return grid_dir


def create_model_metadata(framework: str, grid_size: int, model_class: str,
                         input_size: int, output_size: int, 
                         training_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized model metadata.
    
    Args:
        framework: ML framework name
        grid_size: Game grid size
        model_class: Model class name
        input_size: Number of input features
        output_size: Number of output classes/values
        training_params: Training parameters
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'framework': framework,
        'framework_version': _get_framework_version(framework),
        'python_version': platform.python_version(),
        'grid_size': grid_size,
        'timestamp': datetime.utcnow().isoformat(),
        'model_class': model_class,
        'input_size': input_size,
        'output_size': output_size,
        'training_params': training_params or {}
    }
    return metadata


def _get_framework_version(framework: str) -> str:
    """Get framework version dynamically."""
    try:
        if framework.lower() == "pytorch":
            import torch
            return torch.__version__
        elif framework.lower() == "xgboost":
            import xgboost as xgb
            return xgb.__version__
        elif framework.lower() == "lightgbm":
            import lightgbm as lgb
            return lgb.__version__
        elif framework.lower() == "catboost":
            import catboost as cb
            return cb.__version__
        else:
            return "unknown"
    except ImportError:
        return "not_installed"


def save_model_standardized(model: Any, framework: str, grid_size: int, 
                           model_name: str, model_class: str, input_size: int, 
                           output_size: int, training_params: Dict[str, Any] = None,
                           export_onnx: bool = False, extension_type: str = None,
                           version: str = None) -> str:
    """
    Save model with standardized directory structure and metadata.
    
    Args:
        model: Model object to save
        framework: ML framework name
        grid_size: Game grid size
        model_name: Name for the model file
        model_class: Model class name
        input_size: Number of input features
        output_size: Number of output classes/values
        training_params: Training parameters
        export_onnx: Whether to export ONNX format (PyTorch only)
        extension_type: Extension type for versioned directories (optional)
        version: Extension version for versioned directories (optional)
        
    Returns:
        Path to saved model file
    """
    # Create metadata
    metadata = create_model_metadata(
        framework, grid_size, model_class, input_size, output_size, training_params
    )
    
    # Add versioning info to metadata if provided
    if extension_type and version:
        metadata['extension_type'] = extension_type
        metadata['extension_version'] = version
    
    # Create directory (use versioned structure if possible)
    model_dir = get_model_directory(grid_size, framework, extension_type, version)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file extension and save
    if framework.lower() == 'pytorch':
        filepath = model_dir / f"{model_name}.pth"
        _save_pytorch_model(model, filepath, metadata, export_onnx)
    elif framework.lower() == 'xgboost':
        filepath = model_dir / f"{model_name}.json"
        _save_xgboost_model(model, filepath, metadata)
    elif framework.lower() == 'lightgbm':
        filepath = model_dir / f"{model_name}.txt"
        _save_lightgbm_model(model, filepath, metadata)
    else:
        filepath = model_dir / f"{model_name}.model"
        _save_generic_model(model, filepath, metadata)
    
    return str(filepath)


def _save_pytorch_model(model: Any, filepath: Path, metadata: Dict[str, Any], 
                       export_onnx: bool = False) -> None:
    """Save PyTorch model with state_dict and metadata."""
    import torch
    
    # Save model state_dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': getattr(model, 'optimizer', None).state_dict() if hasattr(model, 'optimizer') else None,
        'grid_size': metadata['grid_size'],
        'metadata': metadata
    }, filepath)
    
    # Save metadata separately
    metadata_path = filepath.with_suffix('.json').with_name(filepath.stem + '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Export ONNX if requested
    if export_onnx and hasattr(model, 'forward'):
        try:
            dummy_input = torch.randn(1, metadata['input_size'])
            onnx_path = filepath.with_suffix('.onnx')
            torch.onnx.export(model, dummy_input, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=11)
            print(f"ONNX model exported to: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    print(f"PyTorch model saved to: {filepath}")
    print(f"Metadata: {metadata}")


def _save_xgboost_model(model: Any, filepath: Path, metadata: Dict[str, Any]) -> None:
    """Save XGBoost model in JSON format."""
    # Save model in JSON format
    model.get_booster().save_model(str(filepath))
    
    # Save metadata separately
    metadata_path = filepath.with_name(filepath.stem + '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"XGBoost model saved to: {filepath}")
    print(f"Metadata: {metadata}")


def _save_lightgbm_model(model: Any, filepath: Path, metadata: Dict[str, Any]) -> None:
    """Save LightGBM model in text format."""
    # Save model in text format
    model.save_model(str(filepath))
    
    # Save metadata separately
    metadata_path = filepath.with_name(filepath.stem + '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"LightGBM model saved to: {filepath}")
    print(f"Metadata: {metadata}")


def _save_generic_model(model: Any, filepath: Path, metadata: Dict[str, Any]) -> None:
    """Save generic model with metadata."""
    # For generic models, just save metadata
    metadata_path = filepath.with_name(filepath.stem + '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Generic model metadata saved to: {metadata_path}")
    print(f"Metadata: {metadata}")


def load_model_standardized(filepath: str, framework: str, model_class: type, **kwargs) -> Any:
    """
    Load model from file with validation.
    
    Args:
        filepath: Path to model file
        framework: ML framework name
        model_class: Model class to instantiate
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Loaded model instance
    """
    if framework.lower() == 'pytorch':
        return _load_pytorch_model(filepath, model_class, **kwargs)
    elif framework.lower() == 'xgboost':
        return _load_xgboost_model(filepath, model_class, **kwargs)
    elif framework.lower() == 'lightgbm':
        return _load_lightgbm_model(filepath, model_class, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _load_pytorch_model(filepath: str, model_class: type, **kwargs) -> Any:
    """Load PyTorch model from file."""
    import torch
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Create model instance
    model = model_class(**kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer if available
    if hasattr(model, 'optimizer') and checkpoint.get('optimizer_state_dict'):
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Validate grid size
    loaded_grid_size = checkpoint.get('grid_size')
    if loaded_grid_size and loaded_grid_size != kwargs.get('grid_size'):
        print(f"Warning: Loaded model grid_size {loaded_grid_size} != current {kwargs.get('grid_size')}")
    
    model.eval()
    return model


def _load_xgboost_model(filepath: str, model_class: type, **kwargs) -> Any:
    """Load XGBoost model from file."""
    
    model = model_class(**kwargs)
    model.load_model(filepath)
    return model


def _load_lightgbm_model(filepath: str, model_class: type, **kwargs) -> Any:
    """Load LightGBM model from file."""
    import lightgbm as lgb
    
    model = lgb.Booster(model_file=filepath)
    return model


__all__ = [
    'get_model_directory',
    'create_model_metadata',
    'save_model_standardized',
    'load_model_standardized'
] 