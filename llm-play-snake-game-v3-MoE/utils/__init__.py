"""
Utils package initialization.
This file exposes all functions, classes, and variables from utility modules.
"""

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

# Import and expose all from log_utils
from .log_utils import (
    format_raw_llm_response,
    format_parsed_llm_response,
    generate_game_summary_json
)

# Import and expose all from replay_utils
from .replay_utils import (
    run_replay,
    check_game_summary_for_moves,
    extract_apple_positions
)

# Import and expose all from game_stats_utils
from .game_stats_utils import (
    create_display_dataframe,
    create_game_performance_chart,
    create_game_dataframe,
    get_experiment_options,
    filter_experiments
)

# Import and expose all from game_manager_utils
from .game_manager_utils import (
    check_max_steps,
    process_game_over,
    handle_error,
    report_final_statistics
)

# Import and expose all from llm_utils
from .llm_utils import (
    parse_llm_response,
    handle_llm_response,
    prepare_snake_prompt,
    format_body_cells_str,
    parse_and_format,
    create_parser_prompt,
    check_llm_health
)

# Import and expose all from text_utils
from .text_utils import (
    process_response_for_display,
    format_code_blocks
)

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
    
    # log_utils
    'format_raw_llm_response',
    'format_parsed_llm_response',
    'generate_game_summary_json',
    
    # replay_utils
    'run_replay',
    'check_game_summary_for_moves',
    'extract_apple_positions',
    
    # game_stats_utils
    'create_display_dataframe',
    'create_game_performance_chart',
    'create_game_dataframe',
    'get_experiment_options',
    'filter_experiments',
    
    # game_manager_utils
    'check_max_steps',
    'process_game_over',
    'handle_error',
    'report_final_statistics',
    
    # llm_utils
    'parse_llm_response',
    'handle_llm_response',
    'prepare_snake_prompt',
    'format_body_cells_str',
    'parse_and_format',
    'create_parser_prompt',
    'check_llm_health',
    
    # text_utils
    'process_response_for_display',
    'format_code_blocks'
]
