"""
Game Manager Helper classes with singleton pattern for efficient utility functions.

This module provides helper classes for managing game state, statistics, and event processing.
BaseGameManagerHelper implements the Singleton pattern since it contains mostly stateless
utility functions that should be shared across all game sessions.

Design Patterns Used:
1. **Singleton Pattern**: Ensures single instance of helper utilities
2. **Template Method Pattern**: Base helper defines algorithm structure  
3. **Strategy Pattern**: Different helper strategies for different tasks

The singleton pattern is appropriate here because:
- Helper functions are stateless and purely functional
- Multiple instances would waste memory without benefit
- Shared utility functions should have consistent behavior
- Thread safety is important for web applications
"""

from __future__ import annotations

import os
from abc import ABC
from typing import Any, Dict, Tuple, TYPE_CHECKING

from colorama import Fore

from config.game_constants import END_REASON_MAP
from core.game_stats_manager import GameStatsManager
from core.game_file_manager import FileManager
from utils.singleton_utils import SingletonABCMeta

if TYPE_CHECKING:
    from core.game_logic import GameLogic
    from core.game_manager import GameManager

__all__ = [
    "BaseGameManagerHelper",
    "GameManagerHelper",
]


class BaseGameManagerHelper(ABC, metaclass=SingletonABCMeta):
    """
    Base class for game manager utilities using Singleton pattern.
    
    This class implements the Singleton pattern because it contains mostly stateless
    utility functions that should be shared across all game sessions. The pattern
    ensures thread safety and memory efficiency while providing consistent behavior.
    
    Design Patterns Implemented:
    1. **Singleton Pattern**: Single instance for thread-safe operations
    2. **Template Method Pattern**: Defines algorithm structure, subclasses fill details
    3. **Strategy Pattern**: Different utility strategies per task type
    
    The class provides utility functions while ensuring:
    - Thread safety through singleton implementation
    - Memory efficiency (no duplicate utility instances)
    - Consistent behavior across all usage points
    - Extensibility for future tasks through inheritance
    """
    
    def __init__(self):
        """
        Initialize the singleton helper instance.
        
        Note: Due to singleton pattern, this will only execute once
        per class, regardless of how many times the class is instantiated.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_helper()
    
    def _setup_helper(self) -> None:
        """
        Setup method called only once during singleton initialization.
        Override in subclasses for specific setup requirements.
        """
        pass
    
    @staticmethod
    def safe_add(target: Dict[str, Any], key: str, delta: Any) -> None:
        """
        Accumulate *delta* onto ``target[key]`` when *delta* is truthy.
        
        This is a utility function for safely adding values to dictionary
        counters, only when the delta value is meaningful (non-zero, non-None).
        
        Args:
            target: Dictionary to update
            key: Key to update in the target dictionary  
            delta: Value to add (only added if truthy)
        """
        if delta:
            target[key] = target.get(key, 0) + delta
    
    @staticmethod
    def check_max_steps(game, max_steps: int) -> bool:
        """
        Check if the game has reached the maximum number of steps.
        
        This is a generic check that can be used by any task to enforce
        step limits and prevent infinite games.
        
        Args:
            game: The snake game instance
            max_steps: Maximum number of steps allowed
            
        Returns:
            Boolean indicating if max steps has been reached
        """
        if game.steps >= max_steps:
            game.last_collision_type = "MAX_STEPS_REACHED"
            human_msg = END_REASON_MAP.get("MAX_STEPS_REACHED", "Max Steps Reached")
            print(Fore.RED + f"❌ Game over: {human_msg} ({max_steps}).")
            return True
        return False
    
    @staticmethod
    def process_events(game_manager) -> None:
        """
        Process pygame events.
        
        Handles basic pygame event processing including quit events and
        keyboard shortcuts. This is generic functionality that works for
        any task using pygame.
        
        Args:
            game_manager: The GameManager instance
        """
        
        # Skip entirely when GUI is disabled – avoids importing pygame in headless mode.
        if not game_manager.use_gui:
            return

        # Lazy import – only executed in GUI runs.  Keeps head-less tests free
        # of the SDL/pygame dependency.
        import pygame  # noqa: WPS433 – intentional local import

        if not pygame.get_init():
            return
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_manager.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_manager.running = False
                elif event.key == pygame.K_r:
                    # Reset game
                    game_manager.game.reset()
                    game_manager.game_active = True
                    game_manager.need_new_plan = True
                    game_manager.consecutive_empty_steps = 0  # Reset on game reset
                    game_manager.consecutive_invalid_reversals = 0  # Reset counter
                    game_manager.current_game_moves = []  # Reset moves for new game
                    print(Fore.GREEN + "🔄 Game reset")


class GameManagerHelper(BaseGameManagerHelper):
    """
    Task-0 specific game manager helper.
    
    Extends BaseGameManagerHelper with LLM-specific functionality including:
    - Game over processing with token and time statistics
    - Final statistics reporting with LLM metrics
    - Game manager initialization with LLM client setup
    - Integration with Task-0 specific managers (GameStatsManager, FileManager)
    """
    
    def process_game_over(self, game: "GameLogic", game_state_info: Dict[str, Any]) -> Tuple[int, int, int, list[int], int, dict, dict, int, int, int, int, int]:
        """
        Process game over state with full Task-0 statistics.
        
        Handles the game over state including:
        - Saving game statistics with LLM token and time data
        - Updating counters for the next game
        - Creating the game summary JSON file with LLM provider info
        - Accumulating session-wide statistics
        
        Args:
            game: The Game instance
            game_state_info: Dictionary with game state info
            
        Returns:
            Tuple of (game_count, total_score, total_steps, game_scores, round_count, 
                     time_stats, token_stats, valid_steps, invalid_reversals, 
                     empty_steps, something_is_wrong_steps, no_path_found_steps)
        """
        args = game_state_info["args"]
        log_dir = game_state_info["log_dir"]
        
        # Extract or initialize counters
        game_count = game_state_info["game_count"]
        total_score = game_state_info["total_score"]
        total_steps = game_state_info["total_steps"]
        game_scores = game_state_info["game_scores"]
        round_count = game_state_info["round_count"]
        # List of per-game round counts collected so far (mutated in-place)
        round_counts = game_state_info.get("round_counts", [])
        time_stats = game_state_info.get("time_stats", {})
        token_stats = game_state_info.get("token_stats", {})
        valid_steps = game_state_info.get("valid_steps", 0)
        invalid_reversals = game_state_info.get("invalid_reversals", 0)
        empty_steps = game_state_info.get("empty_steps", 0)
        something_is_wrong_steps = game_state_info.get("something_is_wrong_steps", 0)
        no_path_found_steps = game_state_info.get("no_path_found_steps", 0)
        
        # Print game over message with reason
        if hasattr(game, "last_collision_type"):
            collision_type = game.last_collision_type

            # Fallback-safe lookup: default to the raw key when not mapped.
            reason_readable = END_REASON_MAP.get(collision_type, collision_type)

            print(Fore.RED + f"❌ Game over: {reason_readable}!")
        
        # Update counters for next game
        game_count += 1
        total_score += game.score
        total_steps += game.steps
        game_scores.append(game.score)
        
        # Increment valid_steps counter
        valid_steps += game.game_state.valid_steps
        
        # Update invalid_reversals counter
        # This ensures we're keeping track of invalid_reversals across all games
        invalid_reversals += game.game_state.invalid_reversals
        
        # Update empty_steps, something_is_wrong_steps and no_path_found_steps counters - add current game's to the running total
        empty_steps += game.game_state.empty_steps
        something_is_wrong_steps += game.game_state.something_is_wrong_steps
        # no_path_found is tracked in StepStats
        if hasattr(game.game_state.stats.step_stats, "no_path_found"):
            no_path_found_steps += game.game_state.stats.step_stats.no_path_found
        
        # Print game stats using the EXECUTED moves that are stored on the GameData
        # instance to avoid counting duplicate/un-executed planned moves.
        executed_moves = game.game_state.moves  # authoritative list
        move_str = ", ".join(executed_moves)
        print(Fore.BLUE + f"Game {game_count} Stats:")
        print(Fore.BLUE + f"- Score: {game.score}")
        print(Fore.BLUE + f"- Steps: {game.steps}")
        print(Fore.BLUE + f"- Valid Steps: {game.game_state.valid_steps}")
        print(Fore.BLUE + f"- Invalid Reversals: {game.game_state.invalid_reversals}")
        print(Fore.BLUE + f"- Moves: {move_str}")
        
        # Update time statistics - Task-0 specific LLM timing
        if hasattr(game.game_state, "get_time_stats"):
            game_time_stats = game.game_state.get_time_stats()
            
            # ── aggregate time statistics ────────────────────────────────
            if game_time_stats:
                self.safe_add(time_stats, "llm_communication_time",
                              game_time_stats.get("llm_communication_time"))
                # game_movement_time / waiting_time were removed from the schema → no aggregation needed
                
                # Also keep track of primary vs. secondary LLM communication time
                primary_time = sum(getattr(game.game_state, "primary_response_times", []))
                secondary_time = sum(getattr(game.game_state, "secondary_response_times", []))
                
                self.safe_add(time_stats, "primary_llm_communication_time", primary_time)
                self.safe_add(time_stats, "secondary_llm_communication_time", secondary_time)
        
        # Update token statistics - Task-0 specific LLM token tracking
        if hasattr(game.game_state, "get_token_stats"):
            game_token_stats = game.game_state.get_token_stats()
            
            # Initialize token stats if not present
            if "primary" not in token_stats:
                token_stats["primary"] = {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0
                }
                
            if "secondary" not in token_stats:
                token_stats["secondary"] = {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0
                }
            
            # Add primary LLM token stats (flat-key schema)
            primary_total = game_token_stats.get("primary_total_tokens")
            primary_prompt = game_token_stats.get("primary_total_prompt_tokens")
            primary_completion = game_token_stats.get("primary_total_completion_tokens")

            if primary_total is not None:
                token_stats["primary"]["total_tokens"] = token_stats["primary"].get("total_tokens", 0) + primary_total
            if primary_prompt is not None:
                token_stats["primary"]["total_prompt_tokens"] = token_stats["primary"].get("total_prompt_tokens", 0) + primary_prompt
            if primary_completion is not None:
                token_stats["primary"]["total_completion_tokens"] = token_stats["primary"].get("total_completion_tokens", 0) + primary_completion

            # Add secondary LLM token stats (flat-key schema)
            secondary_total = game_token_stats.get("secondary_total_tokens")
            secondary_prompt = game_token_stats.get("secondary_total_prompt_tokens")
            secondary_completion = game_token_stats.get("secondary_total_completion_tokens")

            if secondary_total is not None:
                token_stats["secondary"]["total_tokens"] = token_stats["secondary"].get("total_tokens", 0) + secondary_total
            if secondary_prompt is not None:
                token_stats["secondary"]["total_prompt_tokens"] = token_stats["secondary"].get("total_prompt_tokens", 0) + secondary_prompt
            if secondary_completion is not None:
                token_stats["secondary"]["total_completion_tokens"] = token_stats["secondary"].get("total_completion_tokens", 0) + secondary_completion
        
        # Use the actual number of rounds that contain data to avoid the
        # off-by-one "phantom round" that appeared after a wall/self collision.
        if hasattr(game, "game_state") and hasattr(game.game_state, "get_round_count"):
            round_count = game.game_state.get_round_count()

        # ── aggregate round counts ──────────────────────────────────────────
        round_counts.append(round_count)
        total_rounds = sum(round_counts)

        # Save individual game JSON file using the canonical writer
        file_manager = FileManager()
        game_file = file_manager.join_log_path(
            log_dir,
            file_manager.get_game_json_filename(game_count)
        )

        parser_provider = (
            args.parser_provider
            if args.parser_provider
               and args.parser_provider.lower() != "none"
            else None
        )

        # tag the state with the right game number
        game.game_state.game_number = game_count

        # Save game summary with LLM provider information (Task-0 specific)
        game.game_state.save_game_summary(
            game_file,
            primary_provider=args.provider,
            primary_model=args.model,
            parser_provider=parser_provider,
            parser_model=args.parser_model if parser_provider else None,
        )

        print(
            Fore.GREEN +
            f"💾 Saved data for game {game_count} "
            f"(rounds: {round_count}, moves: {len(executed_moves)})"
        )
        
        # Update session stats with LLM-specific data (now includes round-level aggregates)
        stats_manager = GameStatsManager()
        stats_manager.save_session_stats(
            log_dir,
            game_count=game_count,
            total_score=total_score,
            total_steps=total_steps,
            game_scores=game_scores,
            empty_steps=empty_steps,
            something_is_wrong_steps=something_is_wrong_steps,
            no_path_found_steps=no_path_found_steps,
            valid_steps=valid_steps,
            invalid_reversals=invalid_reversals,
            time_stats=time_stats,
            token_stats=token_stats,
            round_counts=round_counts,
            total_rounds=total_rounds,
        )
        
        return (
            game_count,
            total_score,
            total_steps,
            game_scores,
            round_count,
            time_stats,
            token_stats,
            valid_steps,
            invalid_reversals,
            empty_steps,
            something_is_wrong_steps,
            no_path_found_steps,
        )

    def report_final_statistics(self, stats_info: Dict[str, Any]) -> None:
        """
        Report final statistics for the Task-0 experiment.
        
        Includes comprehensive reporting of LLM-specific metrics such as
        token usage, response times, and provider-specific statistics.
        
        Args:
            stats_info: Dictionary containing statistics information:
            - log_dir: Directory containing the summary.json file
            - game_count: Number of games played
            - total_score: Total score across all games
            - total_steps: Total number of steps taken
            - game_scores: List of scores for each game
            - empty_steps: Total empty steps across all games
            - something_is_wrong_steps: Total SOMETHING_IS_WRONG steps across all games
            - valid_steps: Total valid steps across all games (optional)
            - invalid_reversals: Total invalid reversals across all games (optional)
            - no_path_found_steps: Total NO_PATH_FOUND steps across all games (optional)
        """
        # Extract statistics
        log_dir = stats_info["log_dir"]
        game_count = stats_info["game_count"]
        total_score = stats_info["total_score"]
        total_steps = stats_info["total_steps"]
        game_scores = stats_info["game_scores"]
        empty_steps = stats_info["empty_steps"]
        something_is_wrong_steps = stats_info["something_is_wrong_steps"]
        valid_steps = stats_info.get("valid_steps", 0)
        invalid_reversals = stats_info.get("invalid_reversals", 0)
        no_path_found_steps = stats_info.get("no_path_found_steps", 0)
        
        # Get time and token statistics from the game instance if available
        time_stats = {}
        token_stats = {}
        game = stats_info.get("game")
        
        # If we have access to the game instance, update our statistics from it
        if game and hasattr(game, "game_state"):
            game_state = game.game_state
            
            # Get time stats (LLM-specific)
            time_stats = game_state.get_time_stats()
            
            # Get token stats (LLM-specific)
            token_stats = game_state.get_token_stats()
        
        # Get token stats specifically from the game manager if available
        if "token_stats" in stats_info:
            token_stats = stats_info["token_stats"]
            
        # Get time stats specifically from the game manager if available
        if "time_stats" in stats_info:
            time_stats = stats_info["time_stats"]
        
        # Get round counts and total rounds
        round_counts = stats_info.get("round_counts", [])
        total_rounds = stats_info.get("total_rounds", 0)
        
        # Save session statistics to summary file with LLM data
        stats_manager = GameStatsManager()
        stats_manager.save_session_stats(
            log_dir, 
            game_count=game_count, 
            total_score=total_score, 
            total_steps=total_steps, 
            game_scores=game_scores, 
            empty_steps=empty_steps, 
            something_is_wrong_steps=something_is_wrong_steps,
            valid_steps=valid_steps,
            invalid_reversals=invalid_reversals,
            time_stats=time_stats,
            token_stats=token_stats,
            round_counts=round_counts,
            total_rounds=total_rounds,
            no_path_found_steps=no_path_found_steps,
        )
        
        # Print final statistics
        print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
        print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
        print(Fore.GREEN + f"🏁 Final Score: {total_score}")
        print(Fore.GREEN + f"👣 Total Steps: {total_steps}")
        
        # Calculate and print average score
        avg_score = total_score / game_count if game_count > 0 else 0
        print(Fore.GREEN + f"📊 Average Score: {avg_score:.2f}")
        
        # Calculate and print apples per step
        apples_per_step = total_score / total_steps if total_steps > 0 else 0
        print(Fore.GREEN + f"📈 Apples per Step: {apples_per_step:.4f}")
        
        # Print step statistics
        print(Fore.GREEN + f"📈 Empty Moves: {empty_steps}")
        print(Fore.GREEN + f"📈 SOMETHING_IS_WRONG steps: {something_is_wrong_steps}")
        print(Fore.GREEN + f"📈 NO_PATH_FOUND steps: {no_path_found_steps}")
        print(Fore.GREEN + f"📈 Valid Steps: {valid_steps}")
        print(Fore.GREEN + f"📈 Invalid Reversals: {invalid_reversals}")
        
        # End message based on max games reached
        if game_count >= stats_info.get("max_games", float('inf')):
            print(Fore.GREEN + f"🏁 Reached maximum games ({game_count}). Session complete.")

    def initialize_game_manager(self, game_manager: "GameManager") -> None:
        """
        Initialize the Task-0 game manager with LLM-specific setup.
        
        Sets up the LLM clients (primary and optional secondary),
        creates session directories, saves experiment information with
        provider details, and initializes game state tracking.
        
        Args:
            game_manager: The GameManager instance
        """
        from utils.initialization_utils import setup_log_directories, setup_llm_clients, initialize_game_state, enforce_launch_sleep

        # Set up the LLM clients (primary and optional secondary) - Task-0 specific
        setup_llm_clients(game_manager)

        # Handle sleep before launching if specified
        enforce_launch_sleep(game_manager.args)

        # Set up session directories (handles both provided and auto-generated cases)
        setup_log_directories(game_manager)

        # Save experiment information with LLM provider details - Task-0 specific
        stats_manager = GameStatsManager()
        model_info_path = stats_manager.save_experiment_info_json(game_manager.args, game_manager.log_dir)
        print(Fore.GREEN + f"📝 Experiment information saved to {model_info_path}")

        # Initialize game state
        initialize_game_state(game_manager) 