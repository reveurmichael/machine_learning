"""
Supervised Learning v0.03 - Utilities Module
--------------------

Elegant utilities for configuration, CLI, and logging.
Organized by responsibility with clear module boundaries.

Design Pattern: Module Organization
- Single responsibility per module
- Clean API exports
- Consistent naming conventions
"""

# Configuration utilities
from .config_utils import (
    ModelConfig,
    TrainingConfig,
    load_config,
    save_config,
    get_default_config,
    validate_config
)

# CLI utilities
from .cli_utils import (
    create_parser,
    validate_args,
    parse_model_list,
    args_to_config
)

# Logging utilities
from .logging_utils import (
    TrainingLogger,
    MetricsLogger,
    setup_logging,
    log_experiment_start,
    log_experiment_complete
)

__all__ = [
    # Configuration
    "ModelConfig",
    "TrainingConfig", 
    "load_config",
    "save_config",
    "get_default_config",
    "validate_config",
    
    # CLI
    "create_parser",
    "validate_args",
    "parse_model_list",
    "args_to_config",
    
    # Logging
    "TrainingLogger",
    "MetricsLogger",
    "setup_logging",
    "log_experiment_start",
    "log_experiment_complete"
] 