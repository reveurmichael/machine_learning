"""
Extensions Common Package

This package provides shared utilities and configurations for all Snake Game AI
extensions. It follows the principle that each extension + common folder = standalone unit.

Structure:
    config/         - Configuration packages and constants
    validation/     - Validation utilities and rules
    utils/          - Utility modules (reorganized from *_utils.py files)
        - dataset_utils.py (from dataset_loader.py)
        - extension_utils.py
        - factory_utils.py
        - metrics_utils.py
        - path_utils.py
        - test_utils.py
        - csv_schema.py

Key Design Principles:
- SUPREME_RULE NO.3: Flexibility and non-restrictiveness
- SUPREME_RULE NO.4: OOP extensibility for exceptional needs
- Single Source of Truth for shared functionality
- Standalone principle (extension + common = self-contained)

SUPREME_RULE NO.4 Implementation:
All utility classes are designed with inheritance-ready patterns:
- 90% standard usage: Most extensions use utilities as-is
- 10% specialized usage: Extensions can inherit and customize
- Protected extension points for algorithm-specific needs
- Virtual methods for complete behavior replacement
"""

# Import key utilities from the new structure
from .utils.dataset_utils import BaseDatasetLoader, DatasetLoaderFactory
from .utils.extension_utils import (
    ExtensionEnvironment, ExtensionConfig, ExtensionType,
    create_extension_environment, setup_extension_logging
)
from .utils.factory_utils import (
    BaseFactory, AgentFactory, ModelFactory, ComponentRegistry,
    create_agent, create_model, list_available_components
)
from .utils.metrics_utils import (
    ExtensionGameData, ExtensionGameStatistics, ExtensionStepStats,
    MetricsCollector
)
from .utils.path_utils import (
    ensure_project_root_on_path, setup_extension_paths,
    get_extension_path, get_dataset_path, get_model_path
)
from .utils.test_utils import TestRunner, TestCase, run_common_utilities_tests
from .utils.csv_schema import (
    TabularFeatureExtractor, CSVDatasetGenerator, CSVValidator,
    load_and_validate_csv
)

# Import configuration packages
from .config import *

# Import validation utilities  
from .validation import *

__all__ = [
    # Dataset utilities
    "BaseDatasetLoader",
    "DatasetLoaderFactory",
    
    # Extension utilities
    "ExtensionEnvironment",
    "ExtensionConfig", 
    "ExtensionType",
    "create_extension_environment",
    "setup_extension_logging",
    
    # Factory utilities
    "BaseFactory",
    "AgentFactory", 
    "ModelFactory",
    "ComponentRegistry",
    "create_agent",
    "create_model",
    "list_available_components",
    
    # Metrics utilities
    "ExtensionGameData",
    "ExtensionGameStatistics",
    "ExtensionStepStats", 
    "MetricsCollector",
    
    # Path utilities
    "ensure_project_root_on_path",
    "setup_extension_paths",
    "get_extension_path",
    "get_dataset_path",
    "get_model_path",
    
    # Testing utilities
    "TestRunner",
    "TestCase", 
    "run_common_utilities_tests",
    
    # CSV schema utilities
    "TabularFeatureExtractor",
    "CSVDatasetGenerator",
    "CSVValidator",
    "load_and_validate_csv",
] 