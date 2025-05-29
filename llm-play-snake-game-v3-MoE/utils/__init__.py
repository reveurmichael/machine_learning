"""
Core utilities for the LLM-powered Snake game.
This package provides all essential functions and systems for the game, including:
- File and initialization management
- Game state processing and analytics
- JSON parsing and data handling
- Move calculation and game mechanics
"""

import sys
from colorama import Fore

# JSON processing utilities
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
    save_session_stats,
    merge_nested_dicts
)

# File and storage management
from .file_utils import (
    find_log_folders,
    extract_game_stats,
    extract_game_summary,
    get_next_game_number,
    clean_prompt_files,
    save_to_file
)

# Initialization and setup
from .initialization_utils import (
    setup_llm_clients,
    setup_log_directories,
    initialize_game_state,
    read_game_data
)

# Game mechanics
from .move_utils import (
    calculate_move_differences, 
    format_body_cells_str
)

# Text processing
from .text_utils import (
    process_response_for_display
)

# Public API for the utils package
__all__ = [
    # JSON processing
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
    'save_session_stats',
    'merge_nested_dicts',
    
    # File management
    'find_log_folders',
    'extract_game_stats',
    'extract_game_summary',
    'get_next_game_number',
    'clean_prompt_files',
    'save_to_file',
    
    # Initialization and setup
    'setup_llm_clients',
    'setup_log_directories',
    'initialize_game_state',
    'read_game_data',
    
    # Game mechanics
    'calculate_move_differences',
    'format_body_cells_str',
    
    # Text processing
    'process_response_for_display'
]
