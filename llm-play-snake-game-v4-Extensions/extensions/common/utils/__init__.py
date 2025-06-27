"""
Utilities package for extensions/common/.

This package contains all utility modules that support extension development
across the Snake Game AI project. Following SUPREME_RULE NO.4, these utilities
are designed with OOP extensibility to handle exceptional needs while serving
most extensions without modification.

Modules:
    dataset_utils: Dataset loading and preprocessing utilities
    extension_utils: Extension environment management utilities  
    factory_utils: Factory pattern implementations for agent/model creation
    metrics_utils: Performance tracking and analysis utilities
    path_utils: Path management and validation utilities
    test_utils: Testing infrastructure and utilities

Design Philosophy:
    - 90% standard usage: Most extensions use utilities as-is
    - 10% specialized usage: Extensions with exceptional needs can inherit
    - OOP extensibility: Template Method and Strategy patterns enable customization
    - Future-proof design: Enables innovation without breaking existing code

Usage:
    from extensions.common.utils.dataset_utils import BaseDatasetLoader
    from extensions.common.utils.factory_utils import BaseFactory
    from extensions.common.utils.metrics_utils import MetricsCollector
    from extensions.common.utils.path_utils import get_extension_path
"""

# Core utility imports for convenience
from .dataset_utils import (
    BaseDatasetLoader,
    CSVDatasetLoader, 
    JSONLDatasetLoader,
    NPZDatasetLoader,
    DatasetLoaderFactory,
    load_dataset_for_training,
    split_dataset
)

from .extension_utils import (
    ExtensionEnvironment,
    setup_extension_logging,
    create_extension_directories,
    validate_extension_structure
)

from .factory_utils import (
    BaseFactory,
    AgentFactory,
    ModelFactory,
    create_agent,
    create_model,
    register_agent_type,
    register_model_type
)

from .metrics_utils import (
    ExtensionGameData,
    ExtensionGameStatistics, 
    ExtensionStepStats,
    MetricsCollector,
    PerformanceAnalyzer,
    compare_algorithm_performance,
    generate_performance_report
)

from .path_utils import (
    get_extension_path,
    get_dataset_path,
    get_model_path,
    ensure_extension_directories,
    validate_path_structure
)

from .test_utils import (
    TestRunner,
    run_extension_tests,
    validate_extension_compliance,
    generate_test_report
)

__all__ = [
    # Dataset utilities
    'BaseDatasetLoader',
    'CSVDatasetLoader',
    'JSONLDatasetLoader', 
    'NPZDatasetLoader',
    'DatasetLoaderFactory',
    'load_dataset_for_training',
    'split_dataset',
    
    # Extension utilities
    'ExtensionEnvironment',
    'setup_extension_logging',
    'create_extension_directories',
    'validate_extension_structure',
    
    # Factory utilities
    'BaseFactory',
    'AgentFactory',
    'ModelFactory',
    'create_agent',
    'create_model',
    'register_agent_type',
    'register_model_type',
    
    # Metrics utilities
    'ExtensionGameData',
    'ExtensionGameStatistics',
    'ExtensionStepStats', 
    'MetricsCollector',
    'PerformanceAnalyzer',
    'compare_algorithm_performance',
    'generate_performance_report',
    
    # Path utilities
    'get_extension_path',
    'get_dataset_path',
    'get_model_path',
    'ensure_extension_directories',
    'validate_path_structure',
    
    # Test utilities
    'TestRunner',
    'run_extension_tests',
    'validate_extension_compliance',
    'generate_test_report'
] 