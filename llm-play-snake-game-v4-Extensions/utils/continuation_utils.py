"""
Continuation utilities for resuming game sessions.

This module provides elegant, OOP-based utilities for continuing game sessions
from existing log directories. It follows the DRY principle by centralizing
all continuation logic in a single class with proper error handling and
clear separation of concerns.

Design Patterns Used:
- Strategy Pattern: Different continuation strategies can be implemented
- Dependency Injection: File manager is injected for better testability
- Single Responsibility: Each method has one clear purpose
- Fail-Fast: Early validation prevents runtime errors
- Template Method: Consistent flow for continuation operations
"""

from __future__ import annotations
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
from colorama import Fore

if TYPE_CHECKING:
    from core.game_manager import GameManager

from core.game_file_manager import FileManager


class ContinuationError(Exception):
    """Custom exception for continuation-related errors."""
    pass


class ContinuationSession:
    """Encapsulates all continuation session logic and state.
    
    This class follows the Single Responsibility Principle by handling
    only continuation-related operations. It uses dependency injection
    for the file manager to improve testability and implements fail-fast
    validation to prevent runtime errors.
    
    Design Patterns:
    - Strategy Pattern: Different continuation strategies can be implemented
    - Template Method: Consistent flow for all continuation operations
    - Dependency Injection: External dependencies are injected
    - Fail-Fast: Early validation prevents runtime errors
    """
    
    def __init__(self, log_dir: str, file_manager: Optional[FileManager] = None):
        """Initialize continuation session with validation.
        
        Args:
            log_dir: Source log directory to continue from
            file_manager: File manager instance (injected for testability)
            
        Raises:
            ContinuationError: If log_dir is invalid
        """
        self.log_dir = Path(log_dir).resolve()
        self.file_manager = file_manager or FileManager()
        self.summary_path = self.log_dir / "summary.json"
        self.summary_data: Dict[str, Any] = {}
        
        # Validate early to fail fast
        self._validate_log_directory()
        
    def _validate_log_directory(self) -> None:
        """Validate log directory structure (fail-fast approach).
        
        Raises:
            ContinuationError: If directory structure is invalid
        """
        if not self.log_dir.exists():
            raise ContinuationError(f"Log directory does not exist: '{self.log_dir}'")
            
        if not self.log_dir.is_dir():
            raise ContinuationError(f"Path is not a directory: '{self.log_dir}'")
            
        if not self.summary_path.exists():
            raise ContinuationError(f"Missing summary.json in '{self.log_dir}'")
    
    def validate_directory(self) -> None:
        """Validate that the continuation directory is valid.
        
        This method provides backward compatibility while using internal validation.
        
        Raises:
            SystemExit: If directory or summary.json is invalid
        """
        try:
            self._validate_log_directory()
        except ContinuationError as e:
            print(Fore.RED + f"❌ {e}")
            sys.exit(1)
    
    def load_summary_data(self) -> None:
        """Load and validate summary data from the continuation directory.
        
        Raises:
            ContinuationError: If summary data is corrupted or missing required fields
        """
        try:
            self.summary_data = json.loads(self.summary_path.read_text(encoding='utf-8'))
            self._validate_summary_data()
        except json.JSONDecodeError as e:
            print(Fore.YELLOW + f"⚠️ Warning: Invalid JSON in summary.json: {e}")
            print(Fore.YELLOW + "⚠️ Continuing with command-line arguments")
            self.summary_data = {}
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Warning: Could not load configuration from summary.json: {e}")
            print(Fore.YELLOW + "⚠️ Continuing with command-line arguments")
            self.summary_data = {}
    
    def _validate_summary_data(self) -> None:
        """Validate summary data structure.
        
        Ensures that the summary data contains expected sections.
        """
        if not isinstance(self.summary_data, dict):
            raise ContinuationError("Summary data must be a dictionary")
        
        # Ensure required sections exist (create if missing)
        required_sections = ['configuration', 'game_statistics', 'step_stats', 'time_statistics', 'token_usage_stats']
        for section in required_sections:
            if section not in self.summary_data:
                self.summary_data[section] = {}
    
    def apply_original_configuration(self, args: argparse.Namespace) -> None:
        """Apply original experiment configuration to args with proper validation.
        
        This method follows the Template Method pattern by providing a consistent
        flow for applying configuration while allowing for customization.
        
        Args:
            args: Command-line arguments to update
        """
        # Store user-specified overrides that should take precedence
        user_overrides = {
            'max_games': args.max_games,
            'no_gui': getattr(args, 'no_gui', None)
        }
        
        # Apply configuration if available
        if 'configuration' not in self.summary_data:
            print(Fore.YELLOW + "⚠️ No configuration found in summary.json, using defaults")
            return
            
        original_config = self.summary_data['configuration']
        print(Fore.GREEN + "📝 Loading original experiment configuration from summary.json")
        
        # Apply LLM configuration
        self._apply_llm_configuration(args, original_config)
        
        # Apply game configuration
        self._apply_game_configuration(args, original_config)
        
        # Apply limit configurations
        self._apply_limit_configurations(args, original_config)
        
        # Restore user-specified overrides
        self._apply_user_overrides(args, user_overrides)
        
        # Display applied configuration
        self._display_configuration(args)
    
    def _apply_llm_configuration(self, args: argparse.Namespace, config: Dict[str, Any]) -> None:
        """Apply LLM-related configuration settings."""
        # Only set attributes if they exist on args or if we have a value from config
        if hasattr(args, 'provider') or 'provider' in config:
            args.provider = config.get('provider', getattr(args, 'provider', None))
        if hasattr(args, 'model') or 'model' in config:
            args.model = config.get('model', getattr(args, 'model', None))
        if hasattr(args, 'parser_provider') or 'parser_provider' in config:
            args.parser_provider = config.get('parser_provider', getattr(args, 'parser_provider', None))
        if hasattr(args, 'parser_model') or 'parser_model' in config:
            args.parser_model = config.get('parser_model', getattr(args, 'parser_model', None))
    
    def _apply_game_configuration(self, args: argparse.Namespace, config: Dict[str, Any]) -> None:
        """Apply game-related configuration settings."""
        if hasattr(args, 'pause_between_moves') or 'pause_between_moves' in config:
            args.pause_between_moves = config.get('pause_between_moves', getattr(args, 'pause_between_moves', 0.1))
        if hasattr(args, 'max_steps') or 'max_steps' in config:
            args.max_steps = config.get('max_steps', getattr(args, 'max_steps', 1000))
        if hasattr(args, 'sleep_after_empty_step') or 'sleep_after_empty_step' in config:
            args.sleep_after_empty_step = config.get('sleep_after_empty_step', getattr(args, 'sleep_after_empty_step', 1.0))
        if hasattr(args, 'no_gui') or 'no_gui' in config:
            args.no_gui = config.get('no_gui', getattr(args, 'no_gui', False))
    
    def _apply_limit_configurations(self, args: argparse.Namespace, config: Dict[str, Any]) -> None:
        """Apply limit-related configuration settings."""
        limit_configs = [
            'max_consecutive_empty_moves_allowed',
            'max_consecutive_something_is_wrong_allowed',
            'max_consecutive_invalid_reversals_allowed',
            'max_consecutive_no_path_found_allowed'
        ]
        
        for limit_config in limit_configs:
            if hasattr(args, limit_config) or limit_config in config:
                default_value = getattr(args, limit_config, 3)  # Default to 3 if not set
                setattr(args, limit_config, config.get(limit_config, default_value))
    
    def _apply_user_overrides(self, args: argparse.Namespace, overrides: Dict[str, Any]) -> None:
        """Apply user-specified overrides that should take precedence."""
        if overrides['max_games'] is not None and hasattr(args, 'max_games'):
            args.max_games = overrides['max_games']
        if overrides['no_gui'] is not None and hasattr(args, 'no_gui'):
            args.no_gui = overrides['no_gui']
    
    def _display_configuration(self, args: argparse.Namespace) -> None:
        """Display the applied configuration in a user-friendly format."""
        config_items = []
        
        # Build configuration items dynamically based on what exists
        if hasattr(args, 'provider'):
            config_items.append((f"🤖 Primary LLM: {args.provider}", f" ({args.model})" if hasattr(args, 'model') and args.model else ""))
        
        if hasattr(args, 'pause_between_moves'):
            config_items.append((f"⏱️ Move pause: {args.pause_between_moves} seconds", ""))
        
        if hasattr(args, 'max_steps'):
            config_items.append((f"⏱️ Max steps: {args.max_steps}", ""))
        
        if hasattr(args, 'max_consecutive_empty_moves_allowed'):
            config_items.append((f"⏱️ Max empty moves: {args.max_consecutive_empty_moves_allowed}", ""))
        
        if hasattr(args, 'max_consecutive_something_is_wrong_allowed'):
            config_items.append((f"⏱️ Max consecutive errors: {args.max_consecutive_something_is_wrong_allowed}", ""))
        
        if hasattr(args, 'max_consecutive_invalid_reversals_allowed'):
            config_items.append((f"⏱️ Max invalid reversals: {args.max_consecutive_invalid_reversals_allowed}", ""))
        
        if hasattr(args, 'max_consecutive_no_path_found_allowed'):
            config_items.append((f"⏱️ Max no-path-found: {args.max_consecutive_no_path_found_allowed}", ""))
        
        if hasattr(args, 'sleep_after_empty_step'):
            config_items.append((f"⏱️ Sleep after EMPTY: {args.sleep_after_empty_step}s (skipped on NO_PATH_FOUND)", ""))
        
        if hasattr(args, 'no_gui'):
            config_items.append((f"🎮 GUI enabled: {not args.no_gui}", ""))
        
        if hasattr(args, 'max_games'):
            config_items.append((f"🎲 Max games: {args.max_games}", ""))
        
        for main_text, extra_text in config_items:
            print(Fore.GREEN + main_text + extra_text)
        
        # Display parser LLM if configured
        if hasattr(args, 'parser_provider') and args.parser_provider and args.parser_provider.lower() != 'none':
            parser_text = f"🤖 Parser LLM: {args.parser_provider}"
            if hasattr(args, 'parser_model') and args.parser_model:
                parser_text += f" ({args.parser_model})"
            print(Fore.GREEN + parser_text)
    
    def update_continuation_info(self, max_games: int) -> None:
        """Update summary data with continuation information using atomic operations.
        
        This method follows the Single Writer Principle to ensure data consistency.
        
        Args:
            max_games: New maximum games setting
        """
        try:
            # Update configuration
            self.summary_data.setdefault('configuration', {})['max_games'] = max_games
            
            # Remove confusing entries
            self.summary_data['configuration'].pop('continue_with_game_in_dir', None)
            
            # Update continuation info atomically
            self._update_continuation_metadata()
            
            # Write updated data atomically
            self._write_summary_data()
            
            print(Fore.GREEN + "📝 Updated continuation info in summary.json")
            
        except Exception as e:
            print(Fore.RED + f"❌ Failed to update continuation info: {e}")
            raise ContinuationError(f"Failed to update summary.json: {e}")
    
    def _update_continuation_metadata(self) -> None:
        """Update continuation metadata in summary data."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get or create continuation info
        cont_info = self.summary_data.setdefault('continuation_info', {
            'is_continuation': True,
            'continuation_count': 0,
            'continuation_timestamps': [],
            'original_timestamp': self.summary_data.get('timestamp')
        })
        
        # Update continuation metadata
        cont_info['continuation_count'] = cont_info.get('continuation_count', 0) + 1
        cont_info.setdefault('continuation_timestamps', []).append(current_time)
        cont_info['is_continuation'] = True
    
    def _write_summary_data(self) -> None:
        """Write summary data to file with proper error handling."""
        temp_path = self.summary_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file first
            temp_path.write_text(
                json.dumps(self.summary_data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            
            # Atomic rename
            temp_path.replace(self.summary_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def find_latest_game_number(self) -> int:
        """Find the latest completed game number with improved logic.
        
        Returns:
            Next game number to start from
        """
        try:
            game_files = self._find_game_files()
            
            if not game_files:
                print(Fore.YELLOW + f"⚠️ Warning: No game files found in '{self.log_dir}'")
                print(Fore.YELLOW + "⚠️ Starting from game 1 but in continuation mode")
                return 1
            
            # Use file manager's logic for consistency
            return self.file_manager.get_next_game_number(str(self.log_dir))
            
        except Exception as e:
            print(Fore.RED + f"❌ Error finding latest game number: {e}")
            raise ContinuationError(f"Failed to determine latest game number: {e}")
    
    def _find_game_files(self) -> List[str]:
        """Find all game JSON files in the log directory."""
        game_files = []
        
        for file_path in self.log_dir.iterdir():
            if not file_path.is_file() or not file_path.suffix == '.json':
                continue
                
            filename = file_path.name
            if filename.startswith("game_") or (filename.startswith("game") and not filename.startswith("game_")):
                game_files.append(filename)
        
        return game_files
    
    def cleanup_artifacts(self, next_game: int) -> None:
        """Clean up artifacts with proper error handling.
        
        Args:
            next_game: Next game number to start from
        """
        try:
            self.file_manager.clean_prompt_files(str(self.log_dir), next_game)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Warning: Could not clean artifacts: {e}")
            # Continue execution as this is not critical
    
    def setup_game_manager_session(self, game_manager: "GameManager", start_game_number: int) -> None:
        """Set up the game manager for continuation with comprehensive validation.
        
        This method follows the Template Method pattern by providing a consistent
        setup flow while allowing for customization in subclasses.
        
        Args:
            game_manager: The GameManager instance to configure
            start_game_number: The game number to start from
            
        Raises:
            ContinuationError: If setup fails
        """
        try:
            self._validate_setup_parameters(start_game_number)
            self._configure_directories(game_manager)
            self._load_previous_game_state(game_manager, start_game_number)
            self._configure_game_statistics(game_manager)
            self._configure_step_statistics(game_manager)
            self._configure_time_statistics(game_manager)
            self._configure_token_statistics(game_manager)
            self._configure_round_tracking(game_manager)
            self._set_game_counter(game_manager, start_game_number)
            
        except Exception as e:
            print(Fore.RED + f"❌ Failed to set up game manager session: {e}")
            raise ContinuationError(f"Game manager setup failed: {e}")
    
    def _validate_setup_parameters(self, start_game_number: int) -> None:
        """Validate setup parameters."""
        if start_game_number < 1:
            raise ContinuationError(f"Invalid starting game number: {start_game_number}")
    
    def _configure_directories(self, game_manager: "GameManager") -> None:
        """Configure game manager directories."""
        game_manager.log_dir = str(self.log_dir)
        game_manager.prompts_dir = str(self.log_dir / "prompts")
        game_manager.responses_dir = str(self.log_dir / "responses")
        
        # Create directories if they don't exist
        Path(game_manager.prompts_dir).mkdir(exist_ok=True)
        Path(game_manager.responses_dir).mkdir(exist_ok=True)
    
    def _load_previous_game_state(self, game_manager: "GameManager", start_game_number: int) -> None:
        """Load previous game state for validation."""
        from utils.path_utils import get_game_json_filename
        
        if start_game_number > 1:
            prev_game_filename = get_game_json_filename(start_game_number - 1)
            game_file_path = self.file_manager.join_log_path(str(self.log_dir), prev_game_filename)
            
            if not Path(game_file_path).exists():
                raise ContinuationError(f"Cannot find previous game file: {game_file_path}")
    
    def _configure_game_statistics(self, game_manager: "GameManager") -> None:
        """Configure game statistics from summary data."""
        game_stats = self.summary_data.get("game_statistics", {})
        game_manager.total_score = game_stats.get("total_score", 0)
        game_manager.total_steps = game_stats.get("total_steps", 0)
        game_manager.game_scores = game_stats.get("scores", [])
    
    def _configure_step_statistics(self, game_manager: "GameManager") -> None:
        """Configure step statistics from summary data."""
        step_stats = self.summary_data.get("step_stats", {})
        game_manager.empty_steps = step_stats.get("empty_steps", 0)
        game_manager.something_is_wrong_steps = step_stats.get("something_is_wrong_steps", 0)
        game_manager.valid_steps = step_stats.get("valid_steps", 0)
        game_manager.invalid_reversals = step_stats.get("invalid_reversals", 0)
        game_manager.no_path_found_steps = step_stats.get("no_path_found_steps", 0)
    
    def _configure_time_statistics(self, game_manager: "GameManager") -> None:
        """Configure time statistics from summary data."""
        time_stats = self.summary_data.get("time_statistics", {})
        game_manager.time_stats = {
            "llm_communication_time": time_stats.get("total_llm_communication_time", 0),
            "primary_llm_communication_time": time_stats.get("total_primary_llm_communication_time", 0),
            "secondary_llm_communication_time": time_stats.get("total_secondary_llm_communication_time", 0),
        }
    
    def _configure_token_statistics(self, game_manager: "GameManager") -> None:
        """Configure token statistics from summary data."""
        token_usage = self.summary_data.get("token_usage_stats", {})
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
    
    def _configure_round_tracking(self, game_manager: "GameManager") -> None:
        """Configure round tracking from summary data."""
        game_stats = self.summary_data.get("game_statistics", {})
        game_manager.round_counts = game_stats.get("round_counts", [])
        game_manager.total_rounds = game_stats.get("total_rounds", sum(game_manager.round_counts))
    
    def _set_game_counter(self, game_manager: "GameManager", start_game_number: int) -> None:
        """Set the game counter to continue from the correct game."""
        game_manager.game_count = start_game_number - 1


# Public API functions - these delegate to the elegant ContinuationSession class
# These functions provide backward compatibility while using the improved internal implementation

def setup_continuation_session(
    game_manager: "GameManager",
    log_dir: str,
    start_game_number: int,
) -> None:
    """Set up a game session for continuation.
    
    This function delegates to ContinuationSession to avoid code duplication
    and provide a consistent, elegant API.
    
    Args:
        game_manager: GameManager instance to configure
        log_dir: Source log directory to continue from
        start_game_number: Game number to start from
        
    Raises:
        ContinuationError: If setup fails
    """
    try:
        session = ContinuationSession(log_dir)
        session.setup_game_manager_session(game_manager, start_game_number)
    except ContinuationError:
        raise
    except Exception as e:
        raise ContinuationError(f"Failed to set up continuation session: {e}")


def handle_continuation_game_state(game_manager: "GameManager") -> None:
    """Handle game state for continuation mode.
    
    This function delegates to ContinuationSession to avoid code duplication
    and provide consistent error handling.
    
    Args:
        game_manager: GameManager instance with continuation state
    """
    try:
        if hasattr(game_manager, 'log_dir') and game_manager.log_dir:
            session = ContinuationSession(game_manager.log_dir)
            next_game = session.find_latest_game_number()
            session.cleanup_artifacts(next_game)
    except ContinuationError as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not handle continuation state: {e}")
        # Continue execution as this is not critical
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Unexpected error in continuation state handling: {e}")
        # Continue execution as this is not critical


def continue_from_directory(
    game_manager_class: "type[GameManager]", 
    args: argparse.Namespace
) -> "GameManager":
    """Continue from an existing game directory with comprehensive error handling.
    
    This function implements the Template Method pattern by providing a consistent
    flow for continuation operations while delegating specific tasks to the
    ContinuationSession class.
    
    Args:
        game_manager_class: GameManager class to instantiate
        args: Command line arguments
        
    Returns:
        Configured GameManager instance ready for continuation
        
    Raises:
        ContinuationError: If continuation setup fails
        SystemExit: If critical errors occur that prevent continuation
    """
    try:
        # Initialize continuation session
        session = ContinuationSession(args.continue_with_game_in_dir)
        
        # Execute continuation workflow
        session.validate_directory()
        session.load_summary_data()
        session.apply_original_configuration(args)
        session.update_continuation_info(args.max_games)
        
        # Prepare for continuation
        next_game = session.find_latest_game_number()
        session.cleanup_artifacts(next_game)
        
        # Configure for continuation mode
        args.is_continuation = True
        
        # Create and configure GameManager
        game_manager = game_manager_class(args)
        session.setup_game_manager_session(game_manager, next_game)
        
        # Success messages
        print(Fore.GREEN + f"🔄 Continuing from previous session in '{session.log_dir}'")
        print(Fore.GREEN + f"✅ Starting from game {next_game}")
        
        return game_manager
        
    except ContinuationError:
        raise
    except Exception as e:
        print(Fore.RED + f"❌ Unexpected error during continuation setup: {e}")
        raise ContinuationError(f"Failed to continue from directory: {e}") 