"""
This package contains common utilities for the extensions.

Design Philosophy: Standalone Extensions with Shared Commons
-------------------------------------------------------------
Each extension (e.g., `heuristics-v0.02`, `supervised-v0.01`) is designed
to be conceptually standalone. However, to avoid code duplication for
truly generic functionality (like path management, dataset loading, or
standardized web components), this `common` package exists.

An extension is considered "standalone" when combined with this `common`
package. There should be NO direct imports between different extensions
(e.g., `heuristics-v0.02` should not import from `supervised-v0.01`).

This `__init__.py` serves as a facade, exposing the key components from
the common modules. It uses try-except blocks to handle optional
dependencies gracefully, allowing extensions to only use the parts of
the common package they need without pulling in unnecessary libraries
(like streamlit or pytorch).

Key sub-modules:
- config.py: Shared constants.
- dataset_directory_manager.py: Manages paths for logs and datasets.
- csv_schema.py: Defines the data schema for CSV datasets.
- dataset_loader.py: Utilities for loading datasets for training.
- heuristic_*.py: Reusable components for heuristic-based extensions.
- ... and more for RL, supervised learning, etc.
"""

# --- Core Common Utilities ---

from .config import (
    PROJECT_ROOT,
    LOGS_ROOT,
    EXTENSIONS_LOGS_DIR,
    DATASETS_ROOT,
    HEURISTICS_LOG_PREFIX,
    DEFAULT_GRID_SIZE,
    SUPPORTED_GRID_SIZES,
)
from .dataset_directory_manager import DatasetDirectoryManager
from .csv_schema import (
    SUPPORTED_DATA_FORMATS,
    SUPPORTED_DATA_STRUCTURES,
    TABULAR_FEATURE_COLUMNS,
    CSVHeader,
    CSVSchema,
    create_csv_row,
    generate_csv_schema,
)
from .dataset_loader import DatasetLoader

# --- Optional Utilities (with soft dependency handling) ---

# Selective imports for optional dependencies
try:
    from .heuristic_utils import (
        ConsoleTable,
        HeuristicPerformanceTracker,
        HeuristicSessionManager,
        format_results_as_table,
        log_and_print,
        run_heuristic_game,
        setup_game_manager,
        validate_algorithm
    )
    HEURISTIC_UTILS_AVAILABLE = True
except ImportError:
    HEURISTIC_UTILS_AVAILABLE = False

try:
    from .heuristic_replay_utils import (
        ReplayManager,
        ReplayData,
        calculate_replay_metrics,
        create_replay_state_dict,
        get_replay_data,
        get_replay_display_name,
        get_replay_description,
        get_replay_insights,
        validate_replay_navigation
    )
    HEURISTIC_REPLAY_UTILS_AVAILABLE = True
except ImportError:
    HEURISTIC_REPLAY_UTILS_AVAILABLE = False

try:
    from .heuristic_web_utils import (
        StreamlitAlgorithmSelector,
        StreamlitMetricsDashboard,
        StreamlitReplayControls,
        StreamlitTabs,
        create_flask_test_client,
        format_for_flask_response,
        get_heuristic_comparison_table,
        render_leaderboard,
        track_progress
    )
    HEURISTIC_WEB_UTILS_AVAILABLE = True
except ImportError:
    HEURISTIC_WEB_UTILS_AVAILABLE = False

# TODO: Add similar blocks for RL, Supervised, etc.

# --- Public API (via __all__) ---

__all__ = [
    # config.py
    "PROJECT_ROOT",
    "LOGS_ROOT",
    "EXTENSIONS_LOGS_DIR",
    "DATASETS_ROOT",
    "HEURISTICS_LOG_PREFIX",
    "DEFAULT_GRID_SIZE",
    "SUPPORTED_GRID_SIZES",

    # dataset_directory_manager.py
    "DatasetDirectoryManager",

    # csv_schema.py
    "SUPPORTED_DATA_FORMATS",
    "SUPPORTED_DATA_STRUCTURES",
    "TABULAR_FEATURE_COLUMNS",
    "CSVHeader",
    "CSVSchema",
    "create_csv_row",
    "generate_csv_schema",

    # dataset_loader.py
    "DatasetLoader",
]

if HEURISTIC_UTILS_AVAILABLE:
    __all__.extend([
        # heuristic_utils.py
        "ConsoleTable",
        "HeuristicPerformanceTracker",
        "HeuristicSessionManager",
        "format_results_as_table",
        "log_and_print",
        "run_heuristic_game",
        "setup_game_manager",
        "validate_algorithm",
    ])

if HEURISTIC_REPLAY_UTILS_AVAILABLE:
    __all__.extend([
        # heuristic_replay_utils.py
        "ReplayManager",
        "ReplayData",
        "calculate_replay_metrics",
        "create_replay_state_dict",
        "get_replay_data",
        "get_replay_display_name",
        "get_replay_description",
        "get_replay_insights",
        "validate_replay_navigation",
    ])

if HEURISTIC_WEB_UTILS_AVAILABLE:
    __all__.extend([
        # heuristic_web_utils.py
        "StreamlitAlgorithmSelector",
        "StreamlitMetricsDashboard",
        "StreamlitReplayControls",
        "StreamlitTabs",
        "create_flask_test_client",
        "format_for_flask_response",
        "get_heuristic_comparison_table",
        "render_leaderboard",
        "track_progress",
    ])

from .dataset_utils import generate_training_dataset
from .dataset_directory_manager import DatasetDirectoryManager

def get_dataset_path(*args, **kwargs):
    """Public re-export of DatasetDirectoryManager.get_dataset_path."""
    return DatasetDirectoryManager.get_dataset_path(*args, **kwargs)

def ensure_datasets_dir(*args, **kwargs):
    """Public re-export to create dataset directory."""
    return DatasetDirectoryManager.ensure_datasets_dir(*args, **kwargs)

validate_grid_size = DatasetDirectoryManager.validate_grid_size

# Add to __all__
__all__.extend([
    "generate_training_dataset",
    "get_dataset_path",
    "ensure_datasets_dir",
    "validate_grid_size",
]) 