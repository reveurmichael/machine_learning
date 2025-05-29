"""
Core utilities for the LLM-powered Snake game.
This package provides all essential functions and systems for the game, including:
- Single and dual LLM integration and communication
- File and session management
- Game state processing and analytics
- JSON parsing and response handling
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

# Session management
from .session_utils import (
    setup_llm_clients,
    setup_session_directories,
    initialize_game_state
)

# LLM integration system
from .llm_utils import (
    # Prompt generation
    prepare_snake_prompt,
    create_parser_prompt,
    format_raw_llm_response,
    
    # Response parsing
    parse_and_format,
    parse_llm_response,
    handle_llm_response,
    
    # LLM communication
    check_llm_health,
    extract_state_for_parser,
    get_llm_response
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
    
    # Session management
    'setup_llm_clients',
    'setup_session_directories',
    'initialize_game_state',
    
    # LLM integration
    'prepare_snake_prompt',
    'create_parser_prompt',
    'format_raw_llm_response',
    'parse_and_format',
    'parse_llm_response',
    'handle_llm_response',
    'check_llm_health',
    'extract_state_for_parser',
    'get_llm_response'
]
