"""Heuristics → LLM Fine-Tuning Integration v0.02

Evolution from v0.01: Multi-dataset support, enhanced training pipelines, and 
comprehensive evaluation framework.

This extension demonstrates the natural progression from proof-of-concept (v0.01)
to production-ready system (v0.02) with multiple dataset sources, advanced training
configurations, and robust evaluation metrics.

Key Improvements in v0.02:
- Multi-dataset training from different heuristic algorithms
- Advanced fine-tuning configurations (LoRA, QLoRA, full fine-tuning)
- Comprehensive evaluation suite with multiple metrics
- Model comparison and analysis tools
- Integration with common utilities for code reuse
- Enhanced CLI interface with argument validation

Design Philosophy:
- Template Method Pattern: Base pipeline with customizable steps
- Strategy Pattern: Multiple training strategies (LoRA, QLoRA, SFT)
- Factory Pattern: Model and tokenizer creation
- Observer Pattern: Training progress monitoring
- Command Pattern: Evaluation task execution

Architecture:
- Inherits infrastructure patterns from extensions.common
- Extends v0.01 foundation with multi-algorithm support
- Maintains clean separation between data processing and model training
- Provides pluggable evaluation framework for extensibility
"""

import sys as _sys
from pathlib import Path

# Package metadata
__version__ = "0.02"
__author__ = "Snake AI Extensions Team"
__description__ = "Multi-dataset heuristics to LLM fine-tuning pipeline"

# Add alias for underscore-friendly imports
_pkg_tail = __name__.split('.', 1)[1] if '.' in __name__ else __name__
_pkg_alias = f"extensions.{_pkg_tail.replace('-', '_')}"
_sys.modules.setdefault(_pkg_alias, _sys.modules[__name__])

# Export main components
__all__ = [
    "MultiDatasetPipeline",
    "AdvancedTrainingConfig", 
    "EvaluationSuite",
    "ModelComparator",
    "__version__",
]

# Lazy imports for performance
def __getattr__(name: str):
    if name == "MultiDatasetPipeline":
        from .pipeline import MultiDatasetPipeline
        return MultiDatasetPipeline
    elif name == "AdvancedTrainingConfig":
        from .training_config import AdvancedTrainingConfig
        return AdvancedTrainingConfig
    elif name == "EvaluationSuite":
        from .evaluation import EvaluationSuite
        return EvaluationSuite
    elif name == "ModelComparator":
        from .comparison import ModelComparator
        return ModelComparator
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Package validation
def _validate_dependencies():
    """Validate that required dependencies are available."""
    required_packages = [
        "transformers",
        "datasets", 
        "torch",
        "peft",
        "evaluate",
        "accelerate"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: Missing packages for heuristics-llm-fine-tuning-integration-v0.02: {missing_packages}")
        print("Install with: pip install transformers datasets torch peft evaluate accelerate")

# Validate on import
_validate_dependencies() 