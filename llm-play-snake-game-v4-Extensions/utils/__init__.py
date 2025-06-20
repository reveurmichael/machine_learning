"""
Core Utilities for the LLM-powered Snake Game

This package provides comprehensive utility functions organized into logical modules:

**Core Game Mechanics:**
- Board state manipulation and collision detection
- Movement validation and direction processing

**Data Processing:**
- JSON parsing and validation from LLM responses
- Text formatting and display processing

**System Management:**
- Project setup, initialization, and session management
- File operations, statistics, and networking utilities

All utilities are designed to be type-safe, well-documented, and reusable
across different game tasks and extensions.
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
)

# File and storage management
from .file_utils import (
    extract_game_summary,
    get_next_game_number,
    find_valid_log_folders,
    load_summary_data,
    load_game_data,
    get_folder_display_name,
    get_game_json_filename,
    join_log_path,
)

# Initialization and setup
from .initialization_utils import (
    setup_llm_clients,
    setup_log_directories,
    initialize_game_state,
)

# Board manipulation utilities
from .board_utils import (
    generate_random_apple,
    update_board_array,
    is_position_valid,
    get_empty_positions,
)

# Collision detection utilities  
from .collision_utils import (
    check_collision,
    check_wall_collision,
    check_body_collision,
    check_apple_collision,
    positions_overlap,
)

# Movement and direction utilities
from .moves_utils import (
    normalize_direction,
    normalize_directions,
    get_relative_apple_direction_text,
    is_reverse,
)

# Text processing
from .text_utils import (
    process_response_for_display,
    format_code_blocks,
    truncate_text,
    clean_whitespace,
)

# Continuation helpers
from .continuation_utils import (
    setup_continuation_session,
    handle_continuation_game_state,
    continue_from_directory,
)

# Network helpers
from .network_utils import (
    find_free_port,
    is_port_free,
    ensure_free_port,
    random_free_port,
)

from .session_utils import (
    run_replay,
    run_web_replay,
    run_main_web,
    continue_game,
    continue_game_web,
    run_human_play,
    run_human_play_web,
    __all__ as _session_all,
)

from .game_stats_utils import (
    save_experiment_info_json,
    save_session_stats,
)

# Project setup and environment configuration
from .path_utils import (
    ensure_project_root,
    enable_headless_pygame,
    get_project_root,
)

# Reproducibility helpers
from .seed_utils import seed_everything

# Public API for the utils package
__all__ = [
    # ===== Core Game Mechanics =====
    # Board utilities
    "generate_random_apple",
    "update_board_array",
    "is_position_valid",
    "get_empty_positions",
    
    # Collision detection
    "check_collision",
    "check_wall_collision", 
    "check_body_collision",
    "check_apple_collision",
    "positions_overlap",
    
    # Movement utilities
    "normalize_direction",
    "normalize_directions",
    "get_relative_apple_direction_text",
    "is_reverse",
    
    # ===== Data Processing =====
    # JSON processing
    "extract_valid_json",
    "preprocess_json_string",
    "validate_json_format",
    "extract_json_from_code_block",
    "extract_json_from_text",
    "extract_moves_pattern",
    "extract_moves_from_arrays",
    
    # Text processing
    "process_response_for_display",
    "format_code_blocks",
    "truncate_text",
    "clean_whitespace",
    
    # ===== System Management =====
    # Project setup
    "ensure_project_root",
    "enable_headless_pygame",
    "get_project_root",
    
    # File management
    "extract_game_summary",
    "get_next_game_number",
    "find_valid_log_folders",
    "load_summary_data",
    "load_game_data",
    "get_folder_display_name",
    "get_game_json_filename",
    "join_log_path",
    
    # Initialization and setup
    "setup_llm_clients",
    "setup_log_directories",
    "initialize_game_state",
    
    # Continuation helpers
    "setup_continuation_session",
    "handle_continuation_game_state",
    "continue_from_directory",
    
    # Game statistics
    "save_experiment_info_json",
    "save_session_stats",
    
    # Network helpers
    "find_free_port",
    "is_port_free",
    "ensure_free_port",
    "random_free_port",
    
    # Session management (dynamically imported)
    *_session_all,
    
    # Reproducibility
    "seed_everything",
]
