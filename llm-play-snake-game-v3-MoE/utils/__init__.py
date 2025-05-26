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
    extract_moves_fallback,
    extract_moves_from_arrays,
    save_experiment_info_json,
    update_experiment_info_json
)

# Import and expose all from snake_utils
from .snake_utils import (
    filter_invalid_reversals,
    calculate_move_differences,
    parse_llm_response
)

# Import and expose all from log_utils
from .log_utils import (
    save_to_file,
    format_raw_llm_response,
    format_parsed_llm_response,
    generate_game_summary_json
)

# Import and expose all from file_utils
from .file_utils import (
    find_log_folders,
    extract_game_stats,
    extract_game_summary,
    extract_apple_positions
)

# Import and expose all from replay_utils
from .replay_utils import (
    run_replay,
    check_game_summary_for_moves
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
    report_final_statistics,
    handle_llm_response
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
    'extract_moves_fallback',
    'extract_moves_from_arrays',
    'save_experiment_info_json',
    'update_experiment_info_json',
    
    # snake_utils
    'filter_invalid_reversals',
    'calculate_move_differences',
    'parse_llm_response',
    
    # log_utils
    'save_to_file',
    'format_raw_llm_response',
    'format_parsed_llm_response',
    'generate_game_summary_json',
    
    # file_utils
    'find_log_folders',
    'extract_game_stats',
    'extract_game_summary',
    'extract_apple_positions',
    
    # replay_utils
    'run_replay',
    'check_game_summary_for_moves',
    
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
    'handle_llm_response'
]
