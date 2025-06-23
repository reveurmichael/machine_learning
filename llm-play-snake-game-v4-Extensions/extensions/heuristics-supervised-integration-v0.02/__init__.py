"""Heuristics-Supervised Integration v0.02

Evolution from v0.01: Multi-framework support, enhanced CLI, and production-ready
supervised learning pipeline with comprehensive model comparison capabilities.

This extension demonstrates the natural progression from proof-of-concept (v0.01) 
to production-ready system (v0.02) with multiple ML frameworks, advanced training
configurations, and robust evaluation metrics.

Key Improvements in v0.02:
- Multi-framework support (PyTorch, XGBoost, LightGBM, scikit-learn)
- Advanced CLI interface with model selection and configuration
- Comprehensive evaluation suite with model comparison
- Enhanced dataset processing with multiple format support
- Production-ready training pipelines with checkpointing
- Statistical analysis and performance benchmarking
- Integration with common utilities for code reuse

Design Philosophy:
- Template Method Pattern: Base pipeline with customizable steps
- Strategy Pattern: Multiple model strategies (Neural, Tree, Graph)
- Factory Pattern: Model and dataset creation
- Observer Pattern: Training progress monitoring  
- Command Pattern: Training and evaluation task execution
- Facade Pattern: Unified interface to complex ML systems

Architecture:
- Inherits infrastructure patterns from extensions.common
- Extends v0.01 foundation with multi-framework support
- Maintains clean separation between data processing and model training
- Provides pluggable evaluation framework for extensibility
- Supports both heuristic dataset generation and model training

Supported Frameworks:
- PyTorch: Neural networks (MLP, CNN, LSTM, GRU)
- XGBoost: Gradient boosting with advanced features
- LightGBM: Efficient gradient boosting 
- scikit-learn: Classical ML algorithms
- PyTorch Geometric: Graph neural networks

Usage Examples:
    # Multi-framework training pipeline
    python pipeline.py --algorithms BFS ASTAR --models MLP XGBOOST --epochs 100
    
    # Model comparison and evaluation
    python evaluation.py --models MLP XGBOOST LIGHTGBM --dataset-path data/
    
    # Training configuration management
    python training_config.py --create --framework pytorch --model MLP
    
    # Statistical model comparison
    python comparison.py --model-a output/mlp --model-b output/xgboost
"""

import sys as _sys
from pathlib import Path

# Package metadata
__version__ = "0.02"
__author__ = "Snake AI Extensions Team"
__description__ = "Multi-framework heuristics to supervised learning pipeline"

# Add alias for underscore-friendly imports
_pkg_tail = __name__.split('.', 1)[1] if '.' in __name__ else __name__
_pkg_alias = f"extensions.{_pkg_tail.replace('-', '_')}"
_sys.modules.setdefault(_pkg_alias, _sys.modules[__name__])

# Lazy loading pattern for optional dependencies
def _lazy_import_pytorch():
    """Lazily import PyTorch components."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return True
    except ImportError:
        return False

def _lazy_import_xgboost():
    """Lazily import XGBoost components."""
    try:
        import xgboost as xgb
        return True
    except ImportError:
        return False

def _lazy_import_lightgbm():
    """Lazily import LightGBM components."""
    try:
        import lightgbm as lgb
        return True
    except ImportError:
        return False

def _lazy_import_sklearn():
    """Lazily import scikit-learn components."""
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        return True
    except ImportError:
        return False

def _lazy_import_torch_geometric():
    """Lazily import PyTorch Geometric for graph models."""
    try:
        import torch_geometric
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv
        return True
    except ImportError:
        return False

# Dependency availability checks
_PYTORCH_AVAILABLE = None
_XGBOOST_AVAILABLE = None  
_LIGHTGBM_AVAILABLE = None
_SKLEARN_AVAILABLE = None
_TORCH_GEOMETRIC_AVAILABLE = None

def check_pytorch_availability():
    """Check if PyTorch is available."""
    global _PYTORCH_AVAILABLE
    if _PYTORCH_AVAILABLE is None:
        _PYTORCH_AVAILABLE = _lazy_import_pytorch()
    return _PYTORCH_AVAILABLE

def check_xgboost_availability():
    """Check if XGBoost is available."""
    global _XGBOOST_AVAILABLE
    if _XGBOOST_AVAILABLE is None:
        _XGBOOST_AVAILABLE = _lazy_import_xgboost()
    return _XGBOOST_AVAILABLE

def check_lightgbm_availability():
    """Check if LightGBM is available."""
    global _LIGHTGBM_AVAILABLE
    if _LIGHTGBM_AVAILABLE is None:
        _LIGHTGBM_AVAILABLE = _lazy_import_lightgbm()
    return _LIGHTGBM_AVAILABLE

def check_sklearn_availability():
    """Check if scikit-learn is available."""
    global _SKLEARN_AVAILABLE
    if _SKLEARN_AVAILABLE is None:
        _SKLEARN_AVAILABLE = _lazy_import_sklearn()
    return _SKLEARN_AVAILABLE

def check_torch_geometric_availability():
    """Check if PyTorch Geometric is available."""
    global _TORCH_GEOMETRIC_AVAILABLE
    if _TORCH_GEOMETRIC_AVAILABLE is None:
        _TORCH_GEOMETRIC_AVAILABLE = _lazy_import_torch_geometric()
    return _TORCH_GEOMETRIC_AVAILABLE

def get_available_frameworks():
    """Get list of available ML frameworks.
    
    Returns:
        Dict[str, bool]: Framework availability status
    """
    return {
        "pytorch": check_pytorch_availability(),
        "xgboost": check_xgboost_availability(), 
        "lightgbm": check_lightgbm_availability(),
        "sklearn": check_sklearn_availability(),
        "torch_geometric": check_torch_geometric_availability(),
    }

def get_system_info():
    """Get system information for debugging and compatibility."""
    import platform
    import psutil
    
    frameworks = get_available_frameworks()
    
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cpu_count": psutil.cpu_count(),
        "available_frameworks": frameworks,
        "package_version": __version__,
    }

def print_system_info():
    """Print formatted system information."""
    info = get_system_info()
    
    print("🔧 Heuristics-Supervised Integration v0.02 System Information")
    print("=" * 65)
    print(f"📦 Package Version: {info['package_version']}")
    print(f"🐍 Python Version: {info['python_version']}")
    print(f"💻 Platform: {info['platform']}")
    print(f"🧠 Memory: {info['memory_gb']} GB")
    print(f"⚡ CPU Cores: {info['cpu_count']}")
    print()
    print("📚 Framework Availability:")
    for framework, available in info['available_frameworks'].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {framework.ljust(15)}: {status}")
    print()

# Export main components (lazy loaded)
__all__ = [
    "MultiFrameworkPipeline",
    "TrainingConfigManager", 
    "ModelEvaluator",
    "ModelComparator",
    "get_available_frameworks",
    "get_system_info", 
    "print_system_info",
    "__version__",
]

# Lazy loading of main components
def __getattr__(name: str):
    """Implement lazy loading for main components."""
    if name == "MultiFrameworkPipeline":
        from .pipeline import MultiFrameworkPipeline
        return MultiFrameworkPipeline
    elif name == "TrainingConfigManager":
        from .training_config import TrainingConfigManager
        return TrainingConfigManager
    elif name == "ModelEvaluator":
        from .evaluation import ModelEvaluator
        return ModelEvaluator
    elif name == "ModelComparator":
        from .comparison import ModelComparator
        return ModelComparator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 