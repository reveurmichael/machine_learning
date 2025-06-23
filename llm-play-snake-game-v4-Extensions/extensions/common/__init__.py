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
] 