"""
Utilities package.

This package contains various utility modules for the Snake game project.
Many utilities have been migrated to OOP systems in the core/ package:

MIGRATED TO OOP:
- File management utilities → core.file_manager (BaseFileManager, FileManager)
- Game statistics utilities → core.game_stats_manager (BaseGameStatsManager, GameStatsManager)  
- Game manager utilities → core.game_manager_helper (BaseGameManagerHelper, GameManagerHelper)

REMAINING FUNCTIONAL UTILITIES:
- board_utils: Board state analysis and manipulation
- collision_utils: Collision detection algorithms
- moves_utils: Move validation and processing
- initialization_utils: Setup and initialization helpers
- text_utils: Text formatting and parsing
- network_utils: Network communication helpers
- json_utils: JSON serialization utilities
- seed_utils: Random seed management
- path_utils: File path utilities
- web_utils: Web interface utilities
- session_utils: Session management
- continuation_utils: Game continuation logic

For new code, prefer the OOP systems in core/ over the functional utilities here.
"""

# Modern clean imports
from .board_utils import *
from .collision_utils import *
from .continuation_utils import *
from .initialization_utils import *
from .json_utils import *
from .moves_utils import *
from .network_utils import *
from .path_utils import *
from .seed_utils import *
from .session_utils import *
from .text_utils import *
from .web_utils import *

# Export lists for explicit imports
__all__ = [
    # Board utilities
    'get_board_representation',
    'format_board_for_display',
    'validate_board_state',
    
    # Collision utilities  
    'check_collision',
    'check_wall_collision',
    'check_body_collision',
    'check_apple_collision',
    'positions_overlap',
    
    # Continuation utilities
    'setup_continuation_session',
    'handle_continuation_game_state', 
    'continue_from_directory',
    
    # Initialization utilities
    'setup_log_directories',
    'setup_llm_clients',
    'initialize_game_state',
    'enforce_launch_sleep',
    
    # JSON utilities
    'NumPyJSONEncoder',
    'safe_json_load',
    'safe_json_save',
    
    # Move utilities
    'validate_move',
    'get_valid_moves',
    'get_direction_vector',
    'get_opposite_direction',
    
    # Network utilities  
    'check_network_connectivity',
    'retry_with_backoff',
    'handle_network_error',
    
    # Path utilities
    'ensure_directory_exists',
    'get_relative_path',
    'get_absolute_path',
    'validate_file_path',
    
    # Seed utilities
    'set_random_seed',
    'get_random_seed',
    'generate_seed',
    
    # Session utilities
    'create_session_id',
    'validate_session',
    'cleanup_session',
    
    # Text utilities
    'format_timestamp',
    'truncate_text',
    'sanitize_filename',
    'parse_coordinates',
    
    # Web utilities
    'format_json_response',
    'handle_web_error',
    'validate_web_request',
]
