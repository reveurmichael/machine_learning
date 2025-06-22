"""
Continuation utilities for resuming game sessions.

This module provides elegant, OOP-based utilities for continuing game sessions
from existing log directories. It follows the DRY principle by centralizing
all continuation logic in a single class.

Design Patterns Used:
- Strategy Pattern: Different continuation strategies can be implemented
- Dependency Injection: File manager is injected for better testability
- Single Responsibility: Each method has one clear purpose
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
from colorama import Fore

if TYPE_CHECKING:
    from core.game_manager import GameManager

from core.game_file_manager import FileManager


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
        
        # Set the log directory
        game_manager.log_dir = self.log_dir
        game_manager.prompts_dir = os.path.join(self.log_dir, "prompts")
        game_manager.responses_dir = os.path.join(self.log_dir, "responses")
        
        # Create directories if they don't exist
        os.makedirs(game_manager.prompts_dir, exist_ok=True)
        os.makedirs(game_manager.responses_dir, exist_ok=True)
        
        # Get the data from the last game for continuation
        from utils.path_utils import get_game_json_filename
        
        # Get the previous game's data
        prev_game_filename = get_game_json_filename(start_game_number-1)
        game_file_path = self.file_manager.join_log_path(self.log_dir, prev_game_filename)
        
        # If the previous game's file doesn't exist, can't continue
        if not os.path.exists(game_file_path):
            print(Fore.RED + f"❌ Cannot find previous game file: {game_file_path}")
            sys.exit(1)
        
        # Load aggregated statistics from *summary.json*
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
        
        # Round tracking
        game_manager.round_counts = game_stats.get("round_counts", [])
        game_manager.total_rounds = game_stats.get("total_rounds", sum(game_manager.round_counts))
        
        # Set game count to continue from the next game
        game_manager.game_count = start_game_number - 1


# Public API functions - these delegate to the elegant ContinuationSession class

def setup_continuation_session(
    game_manager: "GameManager",
    log_dir: str,
    start_game_number: int,
) -> None:
    """Set up a game session for continuation.
    
    This function delegates to ContinuationSession to avoid code duplication.
    
    Args:
        game_manager: GameManager instance to configure
        log_dir: Source log directory to continue from
        start_game_number: Game number to start from
    """
    session = ContinuationSession(log_dir)
    session.setup_game_manager_session(game_manager, start_game_number)


def handle_continuation_game_state(game_manager: "GameManager") -> None:
    """Handle game state for continuation mode.
    
    This function delegates to ContinuationSession to avoid code duplication.
    
    Args:
        game_manager: GameManager instance with continuation state
    """
    if hasattr(game_manager, 'log_dir'):
        session = ContinuationSession(game_manager.log_dir)
        next_game = session.find_latest_game_number()
        session.cleanup_artifacts(next_game)


def continue_from_directory(
    game_manager_class: "type[GameManager]", 
    args: argparse.Namespace
) -> "GameManager":
    """Continue from an existing game directory.
    
    This function delegates to ContinuationSession for elegant code organization.
    
    Args:
        game_manager_class: GameManager class to instantiate
        args: Command line arguments
        
    Returns:
        Configured GameManager instance ready for continuation
    """
    session = ContinuationSession(args.continue_with_game_in_dir)
    session.validate_directory()
    session.load_summary_data()
    session.apply_original_configuration(args)
    session.update_continuation_info(args.max_games)
    
    # Find the next game number and clean up artifacts
    next_game = session.find_latest_game_number()
    session.cleanup_artifacts(next_game)
    
    # Mark args as continuation to skip re-initialization
    args.is_continuation = True
    
    # Create and configure GameManager
    game_manager = game_manager_class(args)
    session.setup_game_manager_session(game_manager, next_game)
    
    print(Fore.GREEN + f"🔄 Continuing from previous session in '{session.log_dir}'")
    print(Fore.GREEN + f"✅ Starting from game {next_game}")
    
    return game_manager 