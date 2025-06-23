"""
Common utilities for extensions.

This package provides shared utilities that can be used across different extensions.
"""

from .path_utils import ensure_project_root_on_path
from .config import (
    EXTENSIONS_LOGS_DIR,
    HEURISTICS_LOG_PREFIX,
    DEFAULT_GRID_SIZE,
    SUPPORTED_GRID_SIZES,
    get_extension_log_path,
    ensure_extensions_logs_dir,
    get_heuristic_log_path,
    get_dataset_dir,
    get_dataset_path,
    ensure_datasets_dir,
    validate_grid_size,
)
from .dataset_utils import (
    GameState,
    generate_training_dataset,
    extract_game_states_from_json,
    TabularEncoder,
    SequentialEncoder,
    GraphEncoder,
    CSVWriter,
    NPZWriter,
    ParquetWriter,
)

# Expose training helpers (import lazily to avoid heavy deps)
from . import training_cli_utils, training_config_utils, training_logging_utils, rl_utils, rl_helpers

# Import new heuristic utilities
try:
    from .heuristic_utils import (
        HeuristicSessionConfig,
        HeuristicLogger,
        HeuristicPerformanceTracker,
        setup_heuristic_logging,
        format_heuristic_console_output,
        save_heuristic_session_summary,
        validate_algorithm_name,
        execute_heuristic_game_loop,
    )
    _has_heuristic_utils = True
except ImportError:
    _has_heuristic_utils = False

try:
    from .heuristic_replay_utils import (
        ALGORITHM_DISPLAY_NAMES,
        ALGORITHM_DESCRIPTIONS,
        get_algorithm_display_name,
        get_algorithm_description,
        extract_heuristic_replay_data,
        calculate_heuristic_performance_metrics,
        format_algorithm_insights,
        build_heuristic_state_dict,
        validate_replay_navigation,
    )
    _has_heuristic_replay_utils = True
except ImportError:
    _has_heuristic_replay_utils = False

try:
    from .heuristic_web_utils import (
        create_algorithm_selector,
        create_parameter_inputs,
        create_performance_display,
        format_web_state_response,
        build_streamlit_tabs,
        create_replay_controls,
        format_algorithm_metrics,
        create_algorithm_comparison_table,
        create_progress_tracker,
    )
    _has_heuristic_web_utils = True
except ImportError:
    _has_heuristic_web_utils = False

__all__ = [
    "ensure_project_root_on_path",
    "EXTENSIONS_LOGS_DIR",
    "HEURISTICS_LOG_PREFIX",
    "DEFAULT_GRID_SIZE",
    "SUPPORTED_GRID_SIZES", 
    "get_extension_log_path",
    "ensure_extensions_logs_dir",
    "get_heuristic_log_path",
    "get_dataset_dir",
    "get_dataset_path",
    "ensure_datasets_dir",
    "validate_grid_size",
    "GameState",
    "generate_training_dataset",
    "extract_game_states_from_json",
    "TabularEncoder",
    "SequentialEncoder", 
    "GraphEncoder",
    "CSVWriter",
    "NPZWriter",
    "ParquetWriter",
    # Training helpers
    "training_cli_utils",
    "training_config_utils",
    "training_logging_utils",
    "rl_utils",
    "rl_helpers",
]

# Add heuristic utilities to exports if available
if _has_heuristic_utils:
    __all__.extend([
        "HeuristicSessionConfig",
        "HeuristicLogger",
        "HeuristicPerformanceTracker",
        "setup_heuristic_logging",
        "format_heuristic_console_output",
        "save_heuristic_session_summary",
        "validate_algorithm_name",
        "execute_heuristic_game_loop",
    ])

if _has_heuristic_replay_utils:
    __all__.extend([
        "ALGORITHM_DISPLAY_NAMES",
        "ALGORITHM_DESCRIPTIONS",
        "get_algorithm_display_name",
        "get_algorithm_description",
        "extract_heuristic_replay_data",
        "calculate_heuristic_performance_metrics",
        "format_algorithm_insights",
        "build_heuristic_state_dict",
        "validate_replay_navigation",
    ])

if _has_heuristic_web_utils:
    __all__.extend([
        "create_algorithm_selector",
        "create_parameter_inputs",
        "create_performance_display",
        "format_web_state_response",
        "build_streamlit_tabs",
        "create_replay_controls",
        "format_algorithm_metrics",
        "create_algorithm_comparison_table",
        "create_progress_tracker",
    ]) 