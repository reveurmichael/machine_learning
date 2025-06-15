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
    
    # We no longer mutate summary.json here; all summary editing is performed
    # once (and only once) inside *continue_from_directory* to avoid competing
    # writes.  Here we simply verify the file exists so that later helpers can
    # rely on its presence.
        
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
    
    # Load aggregated statistics from *summary.json* 
    from utils.file_utils import load_summary_data

    summary = load_summary_data(log_dir) or {}

    game_stats = summary.get("game_statistics", {})
    game_manager.total_score = game_stats.get("total_score", 0)
    game_manager.total_steps = game_stats.get("total_steps", 0)
    game_manager.game_scores = game_stats.get("scores", [])

    step_stats = summary.get("step_stats", {})
    game_manager.empty_steps = step_stats.get("empty_steps", 0)
    game_manager.something_is_wrong_steps = step_stats.get("something_is_wrong_steps", 0)
    game_manager.valid_steps = step_stats.get("valid_steps", 0)
    game_manager.invalid_reversals = step_stats.get("invalid_reversals", 0)

    # Time statistics (fall back to zeroed dict if missing)
    game_manager.time_stats = summary.get(
        "time_statistics",
        {
            "llm_communication_time": 0,
        },
    )

    # Token statistics – normalize to expected *primary/secondary* keys
    token_usage = summary.get("token_usage_stats", {})
    primary = token_usage.get("primary_llm", {})
    secondary = token_usage.get("secondary_llm", {})
    game_manager.token_stats = {
        "primary": {
            "total_tokens": primary.get("total_tokens", 0),
            "total_prompt_tokens": primary.get("total_prompt_tokens", 0),
            "total_completion_tokens": primary.get("total_completion_tokens", 0),
        },
        "secondary": {
            "total_tokens": secondary.get("total_tokens", 0),
            "total_prompt_tokens": secondary.get("total_prompt_tokens", 0),
            "total_completion_tokens": secondary.get("total_completion_tokens", 0),
        },
    }
    
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
    
    # We'll record the continuation exactly once, AFTER we attempt to load
    # the previous summary – this prevents the duplicate-timestamp artefact
    # that occurred when we recorded before *and* after reading summary.json.

    summary_data = None
    try:
        summary_path = os.path.join(game_manager.log_dir, "summary.json")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Bring over tunables & counters from the previous session
        game_manager.game.game_state.synchronize_with_summary_json(summary_data)
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not load continuation info from summary.json: {e}")

    # Record the continuation exactly once (with or without previous data)
    game_manager.game.game_state.record_continuation(summary_data)

    prev_count = game_manager.game.game_state.continuation_count
    print(Fore.GREEN + f"📝 Marked session as continuation ({prev_count})")
    
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
            args.max_consecutive_something_is_wrong_allowed = original_config.get('max_consecutive_something_is_wrong_allowed', args.max_consecutive_something_is_wrong_allowed)
            args.max_consecutive_invalid_reversals_allowed = original_config.get('max_consecutive_invalid_reversals_allowed', args.max_consecutive_invalid_reversals_allowed)
            
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
            
            # ------------------------------------------------------------------
            # Add / update continuation_info exactly once (single-writer principle)
            # ------------------------------------------------------------------
            cont_info = summary_data.get('continuation_info', {
                'is_continuation': True,
                'continuation_count': 0,
                'continuation_timestamps': [],
                'original_timestamp': summary_data.get('timestamp')
            })

            cont_info['continuation_count'] = cont_info.get('continuation_count', 0) + 1
            cont_info.setdefault('continuation_timestamps', []).append(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Write back the possibly new section
            summary_data['continuation_info'] = cont_info

            # Save the fully-updated summary once, after all edits
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            print(Fore.GREEN + "📝 Updated continuation info in summary.json")
            
            # Log the applied configuration
            print(Fore.GREEN + f"🤖 Primary LLM: {args.provider}" + (f" ({args.model})" if args.model else ""))
            if args.parser_provider and args.parser_provider.lower() != 'none':
                print(Fore.GREEN + f"🤖 Parser LLM: {args.parser_provider}" + (f" ({args.parser_model})" if args.parser_model else ""))
            print(Fore.GREEN + f"⏱️ Move pause: {args.move_pause} seconds")
            print(Fore.GREEN + f"⏱️ Max steps: {args.max_steps}")
            print(Fore.GREEN + f"⏱️ Max empty moves: {args.max_consecutive_empty_moves_allowed}")
            print(Fore.GREEN + f"⏱️ Max consecutive errors: {args.max_consecutive_something_is_wrong_allowed}")
            print(Fore.GREEN + f"⏱️ Max invalid reversals: {args.max_consecutive_invalid_reversals_allowed}")
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