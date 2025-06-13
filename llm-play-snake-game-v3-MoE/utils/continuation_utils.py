"""
Utility module for continuation functionality in Snake game.
Handles reading existing game data and continuing sessions.
"""

import os
import sys
import json
import traceback
from colorama import Fore
from datetime import datetime

def read_existing_game_data(log_dir, start_game_number):
    """Read existing game data from game files.
    
    Args:
        log_dir: Path to the log directory
        start_game_number: The game number to start from
        
    Returns:
        Tuple of (total_score, total_steps, game_scores, empty_steps, error_steps, parser_usage_count, 
                 time_stats, token_stats, valid_steps, invalid_reversals)
    """
    # Initialize counters
    total_score = 0
    total_steps = 0
    empty_steps = 0
    error_steps = 0
    valid_steps = 0
    invalid_reversals = 0
    parser_usage_count = 0
    game_scores = []
    missing_games = []
    corrupted_games = []
    
    # Initialize time and token statistics
    time_stats = {
        "llm_communication_time": 0,
        "game_movement_time": 0,
        "waiting_time": 0
    }
    
    token_stats = {
        "primary": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        },
        "secondary": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        }
    }
    
    print(Fore.GREEN + f"🔍 Reading data from existing {start_game_number-1} games...")
    
    # Get the previous game number
    for game_num in range(1, start_game_number):
        # Import centralized file naming utilities
        from utils.file_utils import get_game_json_filename, join_log_path
        
        # Get the game file path using utility functions
        game_filename = get_game_json_filename(game_num)
        game_file_path = join_log_path(log_dir, game_filename)
        
        if os.path.exists(game_file_path):
            total_score += get_game_score(game_file_path)
            total_steps += get_game_steps(game_file_path)
            game_scores.append(get_game_score(game_file_path))
            
            # Load the game data
            try:
                with open(game_file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                    
                # Track step types if available
                if 'step_stats' in game_data:
                    step_stats = game_data.get('step_stats', {})
                    empty_steps += step_stats.get('empty_steps', 0)
                    error_steps += step_stats.get('error_steps', 0)
                    valid_steps += step_stats.get('valid_steps', 0)
                    invalid_reversals += step_stats.get('invalid_reversals', 0)
                
                # Track parser usage
                parser_usage_count += game_data.get('metadata', {}).get('parser_usage_count', 0)
                
                # Extract time statistics
                if 'time_stats' in game_data:
                    game_time_stats = game_data.get('time_stats', {})
                    time_stats["llm_communication_time"] += game_time_stats.get("llm_communication_time", 0)
                    time_stats["game_movement_time"] += game_time_stats.get("game_movement_time", 0)
                    time_stats["waiting_time"] += game_time_stats.get("waiting_time", 0)
                
                # Extract token statistics
                if 'token_stats' in game_data:
                    game_token_stats = game_data.get('token_stats', {})
                    
                    # Primary LLM token stats
                    if 'primary' in game_token_stats:
                        primary = game_token_stats.get('primary', {})
                        token_stats["primary"]["total_tokens"] += primary.get("total_tokens", 0)
                        token_stats["primary"]["total_prompt_tokens"] += primary.get("total_prompt_tokens", 0)
                        token_stats["primary"]["total_completion_tokens"] += primary.get("total_completion_tokens", 0)
                    
                    # Secondary LLM token stats
                    if 'secondary' in game_token_stats:
                        secondary = game_token_stats.get('secondary', {})
                        token_stats["secondary"]["total_tokens"] += secondary.get("total_tokens", 0)
                        token_stats["secondary"]["total_prompt_tokens"] += secondary.get("total_prompt_tokens", 0)
                        token_stats["secondary"]["total_completion_tokens"] += secondary.get("total_completion_tokens", 0)
            except Exception as e:
                print(Fore.YELLOW + f"⚠️ Warning: Could not load game data from {game_file_path}: {e}")
                corrupted_games.append(game_num)
        else:
            missing_games.append(game_num)
    
    # Report any issues with game files
    if missing_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(missing_games)} game files missing: {missing_games}")
        
    if corrupted_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(corrupted_games)} game files corrupted: {corrupted_games}")
        
    # Successfully loaded games
    successful_games = start_game_number - 1 - len(missing_games) - len(corrupted_games)
    print(Fore.GREEN + f"✅ Successfully loaded {successful_games} game files")
    
    return total_score, total_steps, game_scores, empty_steps, error_steps, parser_usage_count, time_stats, token_stats, valid_steps, invalid_reversals

def setup_continuation_session(game_manager, log_dir, start_game_number):
    """Set up a game session for continuation.
    
    Args:
        game_manager: The GameManager instance
        log_dir: Path to the log directory to continue from
        start_game_number: The game number to start from
    """
    # Verify log directory exists and is valid
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Log directory does not exist: {log_dir}")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Load the original experiment's summary to preserve configuration
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Update the summary with continuation info if it doesn't exist
        if 'continuation_info' not in summary_data:
            summary_data['continuation_info'] = {
                'is_continuation': True,
                'continuation_count': 1,
                'continuation_timestamps': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'original_timestamp': summary_data.get('timestamp')
            }
        else:
            # Update existing continuation info
            continuation_info = summary_data['continuation_info']
            continuation_info['continuation_count'] = continuation_info.get('continuation_count', 0) + 1
            if 'continuation_timestamps' not in continuation_info:
                continuation_info['continuation_timestamps'] = []
            continuation_info['continuation_timestamps'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        # Save the updated summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
            
        print(Fore.GREEN + "📝 Updated continuation info in summary.json")
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not update continuation info in summary.json: {e}")
        
    # Set the log directory
    game_manager.log_dir = log_dir
    game_manager.prompts_dir = os.path.join(log_dir, "prompts")
    game_manager.responses_dir = os.path.join(log_dir, "responses")
    
    # Create directories if they don't exist
    os.makedirs(game_manager.prompts_dir, exist_ok=True)
    os.makedirs(game_manager.responses_dir, exist_ok=True)
    
    # Load and validate the previous game number
    if start_game_number < 1:
        print(Fore.RED + f"❌ Invalid starting game number: {start_game_number}")
        sys.exit(1)
        
    # Get the data from the last game for continuation
    from utils.file_utils import get_game_json_filename, join_log_path
    
    # Get the previous game's data
    prev_game_filename = get_game_json_filename(start_game_number-1)
    game_file_path = join_log_path(log_dir, prev_game_filename)
    
    # If the previous game's file doesn't exist, can't continue
    if not os.path.exists(game_file_path):
        print(f"Error: Cannot find previous game file: {game_file_path}")
        return None
    
    # Initialize time_stats and token_stats attributes if they don't exist
    if not hasattr(game_manager, 'time_stats'):
        game_manager.time_stats = {
            "llm_communication_time": 0,
            "game_movement_time": 0,
            "waiting_time": 0
        }
    
    if not hasattr(game_manager, 'token_stats'):
        game_manager.token_stats = {
            "primary": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        }
    
    # Load statistics from existing games
    (game_manager.total_score, 
     game_manager.total_steps, 
     game_manager.game_scores, 
     game_manager.empty_steps, 
     game_manager.error_steps, 
     game_manager.parser_usage_count,
     game_manager.time_stats,
     game_manager.token_stats,
     game_manager.valid_steps,
     game_manager.invalid_reversals) = read_existing_game_data(log_dir, start_game_number)
    
    # Set game count to continue from the next game
    game_manager.game_count = start_game_number - 1

def setup_llm_clients(game_manager):
    """Set up the LLM clients for continuation.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.initialization_utils import setup_llm_clients as common_setup_llm_clients
    
    # Print configuration being used
    print(Fore.GREEN + "🔄 Setting up LLM clients for continuation mode")
    print(Fore.GREEN + f"🤖 Using Primary LLM: {game_manager.args.provider}" + 
          (f" ({game_manager.args.model})" if game_manager.args.model else ""))
    
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != 'none':
        print(Fore.GREEN + f"🤖 Using Parser LLM: {game_manager.args.parser_provider}" + 
              (f" ({game_manager.args.parser_model})" if game_manager.args.parser_model else ""))
    
    # Use the common utility function to set up LLM clients
    common_setup_llm_clients(game_manager)

def handle_continuation_game_state(game_manager):
    """Handle game state for continuation mode.
    
    Args:
        game_manager: The GameManager instance
    """
    import pygame
    
    # Initialize pygame if using GUI
    if game_manager.use_gui:
        pygame.init()
        pygame.font.init()
    
    # Set up the game
    game_manager.setup_game()
    
    # Mark this as a continuation in the game data
    game_manager.game.game_state.record_continuation()
    print(Fore.GREEN + f"📝 Marked session as continuation ({game_manager.game.game_state.continuation_count})")
    
    # Load summary.json to get information about the previous session
    try:
        summary_path = os.path.join(game_manager.log_dir, "summary.json")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Synchronize game state with summary.json data
        game_manager.game.game_state.synchronize_with_summary_json(summary_data)
        
        # Record this continuation with the previous session data
        game_manager.game.game_state.record_continuation(summary_data)
        
        # Display continuation info
        prev_count = game_manager.game.game_state.continuation_count
        print(Fore.GREEN + f"ℹ️ This is continuation #{prev_count} of this experiment")
        
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not load continuation info from summary.json: {e}")
        # Still record continuation even if synchronization fails
        game_manager.game.game_state.record_continuation()
    
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    print(Fore.GREEN + f"📊 Continuing from game {game_manager.game_count + 1}, with {game_manager.total_score} total score so far")

def continue_from_directory(game_manager_class, args):
    """Factory method to create a GameManager instance for continuation.
    
    Args:
        game_manager_class: The GameManager class
        args: Command-line arguments with continue_with_game_in_dir set
        
    Returns:
        GameManager instance set up for continuation
    """
    from utils.file_utils import get_next_game_number, clean_prompt_files
    
    log_dir = args.continue_with_game_in_dir
    
    # Validate the continuation directory
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Save user-specified command line arguments that should override original settings
    user_max_games = args.max_games
    user_no_gui = args.no_gui
    
    # Load the original experiment configuration from summary.json
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Check if configuration exists in the summary
        if 'configuration' in summary_data:
            original_config = summary_data['configuration']
            
            # Apply the original experiment's configuration
            print(Fore.GREEN + "📝 Loading original experiment configuration from summary.json")
            
            # Copy the original provider and model
            args.provider = original_config.get('provider')
            args.model = original_config.get('model')
            
            # Copy the original parser settings
            args.parser_provider = original_config.get('parser_provider')
            args.parser_model = original_config.get('parser_model')
            
            # Copy other important configuration parameters
            args.move_pause = original_config.get('move_pause', args.move_pause)
            args.max_steps = original_config.get('max_steps', args.max_steps)
            args.max_consecutive_empty_moves_allowed = original_config.get('max_consecutive_empty_moves_allowed', args.max_consecutive_empty_moves_allowed)
            args.max_consecutive_errors_allowed = original_config.get('max_consecutive_errors_allowed', args.max_consecutive_errors_allowed)
            
            # Preserve the original GUI setting
            args.no_gui = original_config.get('no_gui', args.no_gui)
            
            # Now restore user-specified parameters that should override the original settings
            args.max_games = user_max_games
            if user_no_gui is not None:  # Only override if explicitly set
                args.no_gui = user_no_gui
            
            # Update the summary.json with the new max_games value
            summary_data['configuration']['max_games'] = args.max_games
            
            # Remove the continue_with_game_in_dir entry since it's confusing in the configuration
            if 'continue_with_game_in_dir' in summary_data['configuration']:
                del summary_data['configuration']['continue_with_game_in_dir']
            
            # Save the updated configuration
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            # Log the applied configuration
            print(Fore.GREEN + f"🤖 Primary LLM: {args.provider}" + (f" ({args.model})" if args.model else ""))
            if args.parser_provider and args.parser_provider.lower() != 'none':
                print(Fore.GREEN + f"🤖 Parser LLM: {args.parser_provider}" + (f" ({args.parser_model})" if args.parser_model else ""))
            print(Fore.GREEN + f"⏱️ Move pause: {args.move_pause} seconds")
            print(Fore.GREEN + f"⏱️ Max steps: {args.max_steps}")
            print(Fore.GREEN + f"⏱️ Max empty moves: {args.max_consecutive_empty_moves_allowed}")
            print(Fore.GREEN + f"⏱️ Max consecutive errors: {args.max_consecutive_errors_allowed}")
            print(Fore.GREEN + f"🎮 GUI enabled: {not args.no_gui}")
            print(Fore.GREEN + f"🎲 Max games: {args.max_games}")
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not load configuration from summary.json: {e}")
        print(Fore.YELLOW + "⚠️ Continuing with command-line arguments")
        
    # Check if any game files exist
    game_files = []
    for file in os.listdir(log_dir):
        if file.startswith("game_") or (file.startswith("game") and not file.startswith("game_")):
            if file.endswith(".json"):
                game_files.append(file)
    
    if not game_files:
        print(Fore.YELLOW + f"⚠️ Warning: No game files found in '{log_dir}'")
        print(Fore.YELLOW + "⚠️ Starting from game 1 but in continuation mode")
        next_game = 1
    else:
        # Determine the next game number
        next_game = get_next_game_number(log_dir)
    
    print(Fore.GREEN + f"🔄 Continuing from previous session in '{log_dir}'")
    print(Fore.GREEN + f"✅ Starting from game {next_game}")
    
    # Clean existing prompt and response files for games >= next_game
    clean_prompt_files(log_dir, next_game)
    
    # Create and run the game manager with continuation settings
    game_manager = game_manager_class(args)
    
    # Set the is_continuation flag explicitly
    args.is_continuation = True
    
    # Continue from the session
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager

def get_game_score(game_file_path):
    """Extract the score from a game file.
    
    Args:
        game_file_path: Path to the game JSON file
        
    Returns:
        Score value from the game file or 0 if not found
    """
    try:
        with open(game_file_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        return game_data.get('score', 0)
    except Exception as e:
        print(f"Warning: Could not read score from {game_file_path}: {e}")
        return 0

def get_game_steps(game_file_path):
    """Extract the step count from a game file.
    
    Args:
        game_file_path: Path to the game JSON file
        
    Returns:
        Step count from the game file or 0 if not found
    """
    try:
        with open(game_file_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        return game_data.get('steps', 0)
    except Exception as e:
        print(f"Warning: Could not read steps from {game_file_path}: {e}")
        return 0 