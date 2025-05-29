"""
Utils package initialization.
This file exposes all functions, classes, and variables from utility modules.
"""

import sys
from colorama import Fore

# Import and expose all from json_utils
from .json_utils import (
    get_json_error_stats,
    reset_json_error_stats,
    extract_valid_json,
    preprocess_json_string,
    validate_json_format,
    extract_json_from_code_block,
    extract_json_from_text,
    extract_moves_pattern,
    extract_moves_from_arrays,
    save_experiment_info_json,
    update_experiment_info_json
)

# Import and expose all from file_utils first (because log_utils depends on it)
from .file_utils import (
    find_log_folders,
    extract_game_stats,
    extract_game_summary,
    get_next_game_number,
    clean_prompt_files,
    save_to_file
)

# Import from session_utils to avoid cyclic imports
from .session_utils import setup_llm_clients, setup_session_directories, initialize_game_state

# Delay importing the remaining modules to avoid cyclic imports
# They'll be imported on-demand when needed

# Make it easy to import from specific modules
__all__ = [
    # json_utils
    'get_json_error_stats',
    'reset_json_error_stats',
    'extract_valid_json',
    'preprocess_json_string',
    'validate_json_format',
    'extract_json_from_code_block',
    'extract_json_from_text',
    'extract_moves_pattern',
    'extract_moves_from_arrays',
    'save_experiment_info_json',
    'update_experiment_info_json',
    # file_utils
    'find_log_folders',
    'extract_game_stats',
    'extract_game_summary',
    'get_next_game_number',
    'clean_prompt_files',
    'save_to_file',
    # session_utils functions
    'setup_llm_clients',
    'setup_session_directories',
    'initialize_game_state'
]
