"""
Core utilities for the LLM-powered Snake game.
This package provides all essential functions and systems for the game, including:
- File and initialization management
- Game state processing and analytics
- JSON parsing and data handling
- Move calculation and game mechanics
"""


# JSON processing utilities
from .json_utils import (
    extract_valid_json,
    preprocess_json_string,
    validate_json_format,
    extract_json_from_code_block,
    extract_json_from_text,
    extract_moves_pattern,
    extract_moves_from_arrays,
    save_experiment_info_json,
    save_session_stats,
    merge_nested_dicts
)

# File and storage management
from .file_utils import (
    extract_game_summary,
    get_next_game_number,
    clean_prompt_files,
    save_to_file,
    find_valid_log_folders,
    get_game_json_filename,
    get_prompt_filename,
    join_log_path,
    load_summary_data,
    load_game_data,
    get_folder_display_name,
)

# Initialization and setup
from .initialization_utils import (
    setup_llm_clients,
    setup_log_directories,
    initialize_game_state,
)

# Game mechanics – split across two modules
from .moves_utils import (
    normalize_direction,
    normalize_directions,
    calculate_move_differences,
)

from .game_manager_utils import (
    format_body_cells_str,
)

# Text processing
from .text_utils import (
    process_response_for_display
)

# Continuation helpers
from .continuation_utils import (
    setup_continuation_session,
    handle_continuation_game_state,
    continue_from_directory,
)

# Public API for the utils package
__all__ = [
    # JSON processing
    'extract_valid_json',
    'preprocess_json_string',
    'validate_json_format',
    'extract_json_from_code_block',
    'extract_json_from_text',
    'extract_moves_pattern',
    'extract_moves_from_arrays',
    'save_experiment_info_json',
    'save_session_stats',
    'merge_nested_dicts',
    
    # File management
    'extract_game_summary',
    'get_next_game_number',
    'clean_prompt_files',
    'save_to_file',
    'find_valid_log_folders',
    'get_game_json_filename',
    'get_prompt_filename',
    'join_log_path',
    'load_summary_data',
    'load_game_data',
    'get_folder_display_name',
    
    # Initialization and setup
    'setup_llm_clients',
    'setup_log_directories',
    'initialize_game_state',
    
    # Game mechanics
    'normalize_direction',
    'normalize_directions',
    'calculate_move_differences',
    'format_body_cells_str',
    
    # Text processing
    'process_response_for_display',

    # Continuation helpers
    'setup_continuation_session',
    'handle_continuation_game_state',
    'continue_from_directory',
]
