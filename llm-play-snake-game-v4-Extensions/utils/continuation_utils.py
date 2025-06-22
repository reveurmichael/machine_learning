"""
Utility module for continuation functionality in Snake game.
Handles reading existing game data and continuing sessions.

This module is Task0 specific. Future tasks (Task1-5) will NOT have continuation mode.

Design Pattern: Strategy + Factory
- ContinuationSession: Encapsulates all continuation logic and state
- Factory method: continue_from_directory() creates and configures sessions
- Strategy: Different validation and loading strategies can be easily added
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

from colorama import Fore

if TYPE_CHECKING:
    from core.game_manager import GameManager

from llm.log_utils import cleanup_game_artifacts, get_llm_directories
from core.game_file_manager import FileManager
from utils.path_utils import get_summary_json_filename


class ContinuationSession:
    """Encapsulates all continuation session logic and state.
    
    This class follows the Single Responsibility Principle by handling
    only continuation-related operations. It uses dependency injection
    for the file manager to improve testability.
    
    Design Pattern: Strategy Pattern
    - Different continuation strategies can be implemented by subclassing
    - Validation, loading, and updating logic are cleanly separated
    """
    
    def __init__(self, log_dir: str, file_manager: Optional[FileManager] = None):
        """Initialize continuation session.
        
        Args:
            log_dir: Source log directory to continue from
            file_manager: File manager instance (injected for testability)
        """
        self.log_dir = log_dir
        self.file_manager = file_manager or FileManager()
        self.summary_path = os.path.join(log_dir, "summary.json")
        self.summary_data: Dict[str, Any] = {}
        
    def validate_directory(self) -> None:
        """Validate that the continuation directory is valid.
        
        Raises:
            SystemExit: If directory or summary.json is invalid
        """
        if not os.path.isdir(self.log_dir):
            print(Fore.RED + f"❌ Continuation directory does not exist: '{self.log_dir}'")
            sys.exit(1)
            
        if not os.path.exists(self.summary_path):
            print(Fore.RED + f"❌ Missing summary.json in '{self.log_dir}'")
            sys.exit(1)
    
    def load_summary_data(self) -> None:
        """Load summary data from the continuation directory."""
        try:
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                self.summary_data = json.load(f)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Warning: Could not load configuration from summary.json: {e}")
            print(Fore.YELLOW + "⚠️ Continuing with command-line arguments")
            self.summary_data = {}
    
    def apply_original_configuration(self, args: argparse.Namespace) -> None:
        """Apply original experiment configuration to args.
        
        Args:
            args: Command-line arguments to update
        """
        # Preserve user-specified overrides
        user_max_games = args.max_games
        user_no_gui = args.no_gui
        
        # Check if configuration exists in the summary
        if 'configuration' in self.summary_data:
            original_config = self.summary_data['configuration']
            
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
            
            # Log the applied configuration - EXACTLY as original
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
    
    def update_continuation_info(self, max_games: int) -> None:
        """Update summary data with continuation information.
        
        Args:
            max_games: New maximum games setting
        """
        # Update the summary.json with the new max_games value
        self.summary_data['configuration']['max_games'] = max_games
        
        # Remove the continue_with_game_in_dir entry since it's confusing in the configuration
        if 'continue_with_game_in_dir' in self.summary_data['configuration']:
            del self.summary_data['configuration']['continue_with_game_in_dir']
        
        # Add / update continuation_info exactly once (single-writer principle)
        cont_info = self.summary_data.get('continuation_info', {
            'is_continuation': True,
            'continuation_count': 0,
            'continuation_timestamps': [],
            'original_timestamp': self.summary_data.get('timestamp')
        })

        cont_info['continuation_count'] = cont_info.get('continuation_count', 0) + 1
        cont_info.setdefault('continuation_timestamps', []).append(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Write back the possibly new section
        self.summary_data['continuation_info'] = cont_info
        
        # Save the fully-updated summary once, after all edits
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary_data, f, indent=2)
        print(Fore.GREEN + "📝 Updated continuation info in summary.json")
    
    def find_latest_game_number(self) -> int:
        """Find the latest completed game number using original logic.
        
        Returns:
            Next game number to start from
        """
        # Use original logic: check if any game files exist
        game_files = []
        for file in os.listdir(self.log_dir):
            if file.startswith("game_") or (file.startswith("game") and not file.startswith("game_")):
                if file.endswith(".json"):
                    game_files.append(file)
        
        if not game_files:
            print(Fore.YELLOW + f"⚠️ Warning: No game files found in '{self.log_dir}'")
            print(Fore.YELLOW + "⚠️ Starting from game 1 but in continuation mode")
            return 1
        else:
            # Use the original get_next_game_number function
            return self.file_manager.get_next_game_number(self.log_dir)
    
    def cleanup_artifacts(self, next_game: int) -> None:
        """Clean up artifacts using original logic."""
        self.file_manager.clean_prompt_files(self.log_dir, next_game)
    
    def setup_game_manager_session(self, game_manager: "GameManager", start_game_number: int) -> None:
        """Set up the game manager for continuation using ORIGINAL logic.
        
        Args:
            game_manager: The GameManager instance to configure
            start_game_number: The game number to start from
        """
        # Validate inputs
        if start_game_number < 1:
            print(Fore.RED + f"❌ Invalid starting game number: {start_game_number}")
            sys.exit(1)
        
        # Set the log directory - EXACTLY as original
        game_manager.log_dir = self.log_dir
        game_manager.prompts_dir = os.path.join(self.log_dir, "prompts")
        game_manager.responses_dir = os.path.join(self.log_dir, "responses")
        
        # Create directories if they don't exist - EXACTLY as original
        os.makedirs(game_manager.prompts_dir, exist_ok=True)
        os.makedirs(game_manager.responses_dir, exist_ok=True)
        
        # Get the data from the last game for continuation - EXACTLY as original
        from utils.path_utils import get_game_json_filename
        
        # Get the previous game's data
        prev_game_filename = get_game_json_filename(start_game_number-1)
        game_file_path = self.file_manager.join_log_path(self.log_dir, prev_game_filename)
        
        # If the previous game's file doesn't exist, can't continue
        if not os.path.exists(game_file_path):
            print(Fore.RED + f"❌ Cannot find previous game file: {game_file_path}")
            sys.exit(1)
        
        # Load aggregated statistics from *summary.json* - EXACTLY as original
        summary = self.file_manager.load_summary_data(self.log_dir) or {}

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

        # Time statistics (guarantee all expected keys exist) - EXACTLY as original
        ts = summary.get("time_statistics", {})
        game_manager.time_stats = {
            "llm_communication_time": ts.get("total_llm_communication_time", 0),
            "primary_llm_communication_time": ts.get("total_primary_llm_communication_time", 0),
            "secondary_llm_communication_time": ts.get("total_secondary_llm_communication_time", 0),
        }

        # Token statistics – normalize to expected *primary/secondary* keys - EXACTLY as original
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
        
        # Round tracking - EXACTLY as original
        game_manager.round_counts = game_stats.get("round_counts", [])
        game_manager.total_rounds = game_stats.get("total_rounds", sum(game_manager.round_counts))
        
        # Set game count to continue from the next game - EXACTLY as original
        game_manager.game_count = start_game_number - 1


# Public API functions - these are the main entry points for continuation functionality

def setup_continuation_session(game_manager: "GameManager", log_dir: str, start_game_number: int) -> None:
    """Set up a game session for continuation.
    
    Args:
        game_manager: The GameManager instance
        log_dir: Path to the log directory to continue from
        start_game_number: The game number to start from
    """
    session = ContinuationSession(log_dir)
    session.validate_directory()
    session.setup_game_manager_session(game_manager, start_game_number)


def handle_continuation_game_state(game_manager: "GameManager") -> None:
    """Handle game state for continuation mode.
    
    Args:
        game_manager: The GameManager instance
    """
    # Initialize game state using shared utilities - EXACTLY as original
    from utils.initialization_utils import initialize_game_state
    initialize_game_state(game_manager)
    
    # Mark as continuation and log status - EXACTLY as original
    game_manager.game.game_state.record_continuation()
    prev_count = game_manager.game.game_state.continuation_count
    
    print(Fore.GREEN + f"📝 Marked session as continuation ({prev_count})")
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    print(
        Fore.GREEN
        + f"📊 Continuing from game {game_manager.game_count + 1}, with {game_manager.total_score} total score so far"
    )


def continue_from_directory(game_manager_class: "type[GameManager]", args: argparse.Namespace) -> "GameManager":
    """Factory method to create a GameManager instance for continuation.
    
    This function implements the Factory Pattern to create properly configured
    GameManager instances for continuation sessions.
    
    Args:
        game_manager_class: The GameManager class
        args: Command-line arguments with continue_with_game_in_dir set
        
    Returns:
        GameManager instance set up for continuation
        
    Raises:
        SystemExit: If continuation setup fails
    """
    log_dir = args.continue_with_game_in_dir
    
    # Create and configure continuation session
    session = ContinuationSession(log_dir)
    session.validate_directory()
    session.load_summary_data()
    session.apply_original_configuration(args)
    session.update_continuation_info(args.max_games)
    
    # Find next game and clean up artifacts - EXACTLY as original
    next_game = session.find_latest_game_number()
    
    print(Fore.GREEN + f"🔄 Continuing from previous session in '{log_dir}'")
    print(Fore.GREEN + f"✅ Starting from game {next_game}")
    
    # Clean existing prompt and response files for games >= next_game - EXACTLY as original
    session.cleanup_artifacts(next_game)
    
    # Create and run the game manager with continuation settings - EXACTLY as original
    game_manager = game_manager_class(args)
    
    # Set the is_continuation flag explicitly - EXACTLY as original
    args.is_continuation = True
    
    # Set up LLM clients with the configuration from the original experiment - EXACTLY as original
    from utils.initialization_utils import setup_llm_clients
    setup_llm_clients(game_manager)
    
    # Continue from the session - EXACTLY as original
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager