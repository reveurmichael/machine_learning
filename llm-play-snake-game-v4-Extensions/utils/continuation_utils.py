"""
Utility module for continuation functionality in Snake game.
Handles reading existing game data and continuing sessions.

This module is Task0 specific. So no need for BaseGameManager.


IMPORTANT:
- In fact, we will have continuation mode for only for Task0.
- For Task1, Task2, Task3, Task4, Task5, we will NOT have continuation mode.

"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.game_manager import GameManager

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from colorama import Fore

from llm.log_utils import cleanup_game_artifacts, get_llm_directories
from core.game_file_manager import FileManager
from utils.path_utils import get_summary_json_filename

# Initialize file manager for continuation operations
_file_manager = FileManager()

# This function is Task0 specific.
def setup_continuation_session(
    game_manager: "GameManager",
    log_dir: str,
    start_game_number: int,
) -> None:
    """Set up a game session for continuation.
    
    Args:
        game_manager: The GameManager instance
        log_dir: Path to the log directory to continue from
        start_game_number: The game number to start from
    """
    # Verify log directory exists and is valid
    if not Path(log_dir).is_dir():
        print(Fore.RED + f"❌ Log directory does not exist: {log_dir}")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = Path(log_dir) / get_summary_json_filename()
    if not summary_path.exists():
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # We no longer mutate summary.json here; all summary editing is performed
    # once (and only once) inside *continue_from_directory* to avoid competing
    # writes.  Here we simply verify the file exists so that later helpers can
    # rely on its presence.
        
    # Set the log directory
    game_manager.log_dir = log_dir
    game_manager.prompts_dir, game_manager.responses_dir = get_llm_directories(log_dir)
    
    # Create directories if they don't exist
    game_manager.prompts_dir.mkdir(exist_ok=True)
    game_manager.responses_dir.mkdir(exist_ok=True)
    
    # Load and validate the previous game number
    if start_game_number < 1:
        print(Fore.RED + f"❌ Invalid starting game number: {start_game_number}")
        sys.exit(1)
        
    # Get the data from the last game for continuation
    prev_game_filename = _file_manager.get_game_json_filename(start_game_number-1)
    game_file_path = _file_manager.join_log_path(log_dir, prev_game_filename)
    
    # If the previous game's file doesn't exist, can't continue
    if not Path(game_file_path).exists():
        print(Fore.RED + f"❌ Cannot find previous game file: {game_file_path}")
        sys.exit(1)
    
    # Load aggregated statistics from *summary.json* 
    summary = _file_manager.load_summary_data(log_dir) or {}

    game_stats = summary.get("game_statistics", {})
    game_manager.total_score = game_stats.get("total_score", 0)
    game_manager.total_steps = game_stats.get("total_steps", 0)
    game_manager.game_scores = game_stats.get("scores", [])

    step_stats = summary.get("step_stats", {})
    game_manager.empty_steps = step_stats.get("empty_steps", 0)
    game_manager.something_is_wrong_steps = step_stats.get("something_is_wrong_steps", 0)
    game_manager.valid_steps = step_stats.get("valid_steps", 0)
    game_manager.invalid_reversals = step_stats.get("invalid_reversals", 0)
    game_manager.no_path_found_steps = step_stats.get("no_path_found_steps", 0)

    # Time statistics (guarantee all expected keys exist)
    ts = summary.get("time_statistics", {})
    game_manager.time_stats = {
        "llm_communication_time": ts.get("total_llm_communication_time", 0),
        "primary_llm_communication_time": ts.get("total_primary_llm_communication_time", 0),
        "secondary_llm_communication_time": ts.get("total_secondary_llm_communication_time", 0),
    }

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
    
    # ---------------- Round tracking ------------------
    game_manager.round_counts = game_stats.get("round_counts", [])
    game_manager.total_rounds = game_stats.get("total_rounds", sum(game_manager.round_counts))
    
    # Set game count to continue from the next game
    game_manager.game_count = start_game_number - 1

# This function is Task0 specific.
def handle_continuation_game_state(game_manager: "GameManager") -> None:
    """Handle game state for continuation mode.
    
    Args:
        game_manager: The GameManager instance
    """
    # Local import to avoid circular dependency
    from utils.initialization_utils import initialize_game_state
    
    # Initialise pygame & game state via shared helper to avoid duplication
    initialize_game_state(game_manager)

    # Mark this run as a continuation so stats & dashboard reflect it.
    game_manager.game.game_state.record_continuation()
    
    prev_count = game_manager.game.game_state.continuation_count
    print(Fore.GREEN + f"📝 Marked session as continuation ({prev_count})")
    
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    print(
        Fore.GREEN
        + f"📊 Continuing from game {game_manager.game_count + 1}, with {game_manager.total_score} total score so far"
    )

# This function is Task0 specific.
def continue_from_directory(
    game_manager_class: "type[GameManager]", args: argparse.Namespace
) -> "GameManager":
    """Factory method to create a GameManager instance for continuation.
    
    Args:
        game_manager_class: The GameManager class
        args: Command-line arguments with continue_with_game_in_dir set
        
    Returns:
        GameManager instance set up for continuation
    """
    log_dir = Path(args.continue_with_game_in_dir)
    
    # Validate the continuation directory
    if not log_dir.is_dir():
        print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = log_dir / get_summary_json_filename()
    if not summary_path.exists():
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Save user-specified command line arguments that should override original settings
    user_max_games = args.max_games
    user_no_gui = args.no_gui
    
    # Load the original experiment configuration from summary.json
    try:
        summary_data = json.loads(summary_path.read_text(encoding='utf-8'))
            
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
            args.pause_between_moves = original_config.get('pause_between_moves', args.pause_between_moves)
            args.max_steps = original_config.get('max_steps', args.max_steps)
            args.max_consecutive_empty_moves_allowed = original_config.get('max_consecutive_empty_moves_allowed', args.max_consecutive_empty_moves_allowed)
            args.max_consecutive_something_is_wrong_allowed = original_config.get('max_consecutive_something_is_wrong_allowed', args.max_consecutive_something_is_wrong_allowed)
            args.max_consecutive_invalid_reversals_allowed = original_config.get('max_consecutive_invalid_reversals_allowed', args.max_consecutive_invalid_reversals_allowed)
            args.max_consecutive_no_path_found_allowed = original_config.get('max_consecutive_no_path_found_allowed', args.max_consecutive_no_path_found_allowed)
            args.sleep_after_empty_step = original_config.get('sleep_after_empty_step', args.sleep_after_empty_step)
            
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
            
            # ---------------------
            # Add / update continuation_info exactly once (single-writer principle)
            # ---------------------
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
            print(Fore.GREEN + f"⏱️ Move pause: {args.pause_between_moves} seconds")
            print(Fore.GREEN + f"⏱️ Max steps: {args.max_steps}")
            print(Fore.GREEN + f"⏱️ Max empty moves: {args.max_consecutive_empty_moves_allowed}")
            print(Fore.GREEN + f"⏱️ Max consecutive errors: {args.max_consecutive_something_is_wrong_allowed}")
            print(Fore.GREEN + f"⏱️ Max invalid reversals: {args.max_consecutive_invalid_reversals_allowed}")
            print(Fore.GREEN + f"⏱️ Max no-path-found: {args.max_consecutive_no_path_found_allowed}")
            print(Fore.GREEN + f"⏱️ Sleep after EMPTY: {args.sleep_after_empty_step}s (skipped on NO_PATH_FOUND)")
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
        next_game = _file_manager.get_next_game_number(log_dir)
    
    print(Fore.GREEN + f"🔄 Continuing from previous session in '{log_dir}'")
    print(Fore.GREEN + f"✅ Starting from game {next_game}")
    
    # Remove artefacts from incomplete games >= next_game (fresh, no back-compat wrapper)
    cleanup_game_artifacts(log_dir, next_game)
    
    # Create and run the game manager with continuation settings
    game_manager = game_manager_class(args)
    
    # Set the is_continuation flag explicitly
    args.is_continuation = True
    
    # Set up LLM clients with the configuration from the original experiment
    # Local import to avoid circular dependency
    from utils.initialization_utils import setup_llm_clients
    setup_llm_clients(game_manager)
    
    # Continue from the session
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager