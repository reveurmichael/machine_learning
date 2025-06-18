import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from utils.game_stats_utils import NumPyJSONEncoder
from utils.moves_utils import normalize_direction
from core.game_stats import GameStatistics
from core.game_rounds import RoundManager


class GameData:
    """Tracks and manages statistics for Snake game sessions."""
    
    def __init__(self) -> None:
        """Initialize the game data tracking."""
        self.stats = GameStatistics()
        self.round_manager = RoundManager()
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracking data to initial state."""
        from config.game_constants import (
            MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
        )
        
        self.game_number = 0
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.apple_positions = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_end_reason = None
        self.snake_positions = []
        self.apple_position = None
        self.moves = []
        
        self.max_consecutive_empty_moves_allowed = MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED
        self.max_consecutive_something_is_wrong_allowed = (
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED
        )
        self.max_consecutive_invalid_reversals_allowed = (
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED
        )
        # Dedicated safeguard for repeated NO_PATH_FOUND responses
        self.max_consecutive_no_path_found_allowed = (
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED
        )
        
        self.stats = GameStatistics()
        self.round_manager = RoundManager()
        
    def start_new_round(self, apple_position) -> None:
        """Start a new round of moves."""
        self.round_manager.start_new_round(apple_position)
    
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics."""
        move = normalize_direction(move)
        self.steps += 1
        self.stats.step_stats.valid += 1
        
        if apple_eaten:
            self.score += 1
            
        self.moves.append(move)
        self.round_manager.round_buffer.add_move(move)
    
    def record_apple_position(self, position) -> None:
        """Record an apple position."""
        x, y = position
        self.apple_positions.append({"x": x, "y": y})
        self.round_manager.record_apple_position(position)
    
    def record_empty_move(self) -> None:
        """Record an *EMPTY* sentinel.

        EMPTY represents a tick where the snake stayed in place because **no
        valid move was produced at all**.  It is *not* used for NO_PATH_FOUND
        or parser/LLM failures – those have their own dedicated sentinel move
        names so statistics stay strictly independent.
        """
        self.stats.step_stats.empty += 1
        self.steps += 1
        self.moves.append("EMPTY")
        self.round_manager.round_buffer.add_move("EMPTY")
    
    def record_invalid_reversal(self) -> None:
        """Record an invalid reversal move."""
        self.stats.step_stats.invalid_reversals += 1
        self.steps += 1
        self.moves.append("INVALID_REVERSAL")
        self.round_manager.round_buffer.add_move("INVALID_REVERSAL")
    
    def record_something_is_wrong_move(self) -> None:
        """Record a *SOMETHING_IS_WRONG* sentinel.

        Indicates a parser/LLM failure (e.g. JSON could not be extracted).
        Logged independently so that EMPTY and NO_PATH_FOUND statistics are
        unaffected by error handling.
        """
        self.stats.step_stats.something_wrong += 1
        self.steps += 1
        self.moves.append("SOMETHING_IS_WRONG")
        self.round_manager.round_buffer.add_move("SOMETHING_IS_WRONG")
    
    def record_game_end(self, reason: str) -> None:
        """Record the end of a game."""
        if not self.game_over:
            self.stats.time_stats.record_end_time()
        self.game_over = True
        self.game_end_reason = reason

    def record_parsed_llm_response(self, response: Any, is_primary: bool) -> None:
        """Records the parsed response from an LLM."""
        self.round_manager.record_parsed_llm_response(response, is_primary)
    
    def get_prompt_response_stats(self) -> Dict[str, Any]:
        """Returns a dictionary with prompt and response time statistics."""
        primary_times = self.stats.primary_response_times
        secondary_times = self.stats.secondary_response_times
        return {
            "primary_response_times": primary_times,
            "secondary_response_times": secondary_times,
            "avg_primary_response_time": (
                float(np.mean(primary_times)) if primary_times else 0.0
            ),
            "avg_secondary_response_time": (
                float(np.mean(secondary_times)) if secondary_times else 0.0
            ),
        }
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Returns a dictionary with token usage statistics."""
        return {
            "primary_total_tokens": self.stats.primary_total_tokens,
            "primary_avg_total_tokens": self.stats.primary_avg_total_tokens,
            "primary_total_prompt_tokens": self.stats.primary_total_prompt_tokens,
            "primary_avg_prompt_tokens": self.stats.primary_avg_prompt_tokens,
            "primary_total_completion_tokens": self.stats.primary_total_completion_tokens,
            "primary_avg_completion_tokens": self.stats.primary_avg_completion_tokens,
            "secondary_total_tokens": self.stats.secondary_total_tokens,
            "secondary_avg_total_tokens": self.stats.secondary_avg_total_tokens,
            "secondary_total_prompt_tokens": self.stats.secondary_total_prompt_tokens,
            "secondary_avg_prompt_tokens": self.stats.secondary_avg_prompt_tokens,
            "secondary_total_completion_tokens": self.stats.secondary_total_completion_tokens,
            "secondary_avg_completion_tokens": self.stats.secondary_avg_completion_tokens,
        }
    
    def generate_game_summary(
        self,
        primary_provider: str,
        primary_model: Optional[str],
        parser_provider: Optional[str],
        parser_model: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a JSON-serialisable dictionary that fully captures a completed game.

        The returned structure still caters for the replay modules by including a
        `detailed_history` block composed of flat arrays alongside per-round
        data.  This keeps downstream visualisation unchanged while allowing the
        core implementation to evolve independently.
        """
        summary: dict = {
            # High-level outcome ----------------------------------------
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": self.game_over,
            "game_end_reason": self.game_end_reason,
            "round_count": self.round_manager.round_count,
            # LLM configuration ---------------------------------------
            "llm_info": {
                "primary_provider": primary_provider,
                "primary_model": primary_model,
                "parser_provider": parser_provider,
                "parser_model": parser_model,
            },
            # Timings / stats -------------------------------
            "time_stats": self.stats.time_stats.asdict(),
            "prompt_response_stats": self.get_prompt_response_stats(),
            "token_stats": self.get_token_stats(),
            "step_stats": self.stats.step_stats.asdict(),
            # Misc metadata --------------------------------
            "metadata": {
                "timestamp": self.timestamp,
                "game_number": self.game_number,
                "round_count": self.round_manager.round_count,
                # Copy through any extra metadata the caller supplies.
                **kwargs.get("metadata", {}),
            },
            # Full replay data ----------------------------------------
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": self.moves,
                "rounds_data": self.round_manager.get_ordered_rounds_data(),
            },
            }
        
        return summary
    
    def save_game_summary(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Save the game summary to a file."""
        # Ensure any in-progress round data (typically the last one at game
        # over) is persisted before we serialise.  Without this, the final
        # round's `moves` array may be empty in game_N.json.
        self.round_manager.flush_buffer()

        summary_dict = self.generate_game_summary(**kwargs)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, cls=NumPyJSONEncoder, indent=4)
        return summary_dict

    @property
    def snake_length(self) -> int:
        """Returns the length of the snake."""
        return len(self.snake_positions)

    # -------------------------------
    # Delegating wrappers for GameStatistics
    # -------------------------------

    def record_llm_communication_start(self) -> None:
        """Proxy to GameStatistics."""
        self.stats.record_llm_communication_start()

    def record_llm_communication_end(self) -> None:
        """Proxy to GameStatistics."""
        self.stats.record_llm_communication_end()

    def record_primary_response_time(self, duration: float) -> None:
        self.stats.record_primary_response_time(duration)

    def record_secondary_response_time(self, duration: float) -> None:
        self.stats.record_secondary_response_time(duration)

    def record_primary_token_stats(
        self, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self.stats.record_primary_token_stats(prompt_tokens, completion_tokens)

    def record_secondary_token_stats(
        self, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self.stats.record_secondary_token_stats(prompt_tokens, completion_tokens)


    # -------------------------------
    # Continuation-mode helpers (needed by utils/continuation_utils.py)
    # -------------------------------

    def record_continuation(self) -> None:
        """Mark this run as a continuation of a previous experiment."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lazily create the attributes the first time we are called.
        if not hasattr(self, "is_continuation"):
            self.is_continuation = True
            self.continuation_count = 1
            self.continuation_timestamps = [timestamp]
        else:
            self.continuation_count += 1
            self.continuation_timestamps.append(timestamp)

    # -------------------------------
    # Quick accessors required by utils/game_manager_utils
    # -------------------------------

    @property
    def valid_steps(self) -> int:
        return self.stats.step_stats.valid

    @property
    def invalid_reversals(self) -> int:
        return self.stats.step_stats.invalid_reversals

    @property
    def empty_steps(self) -> int:
        return self.stats.step_stats.empty

    @property
    def something_is_wrong_steps(self) -> int:
        return self.stats.step_stats.something_wrong

    # -------------------------------
    # Convenience accessors for session-level aggregation
    # -------------------------------

    @property
    def primary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the primary LLM."""
        return self.stats.primary_response_times

    @property
    def secondary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the secondary LLM."""
        return self.stats.secondary_response_times

    # -------------------------------
    # Time statistics view (used by game_manager_utils)
    # -------------------------------

    def get_time_stats(self) -> Dict[str, Any]:
        """Return wall-clock timings needed for session aggregation."""
        # No longer track movement/waiting breakdowns – just return the
        # coarse timing summary.
        return self.stats.time_stats.asdict()

    # -------------------------------
    # Misc helpers expected by utils.game_manager_utils
    # -------------------------------

    def _calculate_actual_round_count(self) -> int:
        """Return the number of rounds that actually hold data."""
        return len([r for r in self.round_manager.rounds_data.values() if r])

    # --------------------------------
    # Public wrapper – prefer this over direct access to the underscore
    # helper so external modules avoid pylint W0212.
    # --------------------------------

    def get_round_count(self) -> int:
        """Return the number of rounds that actually contain gameplay data.

        This simply delegates to the internal computation method but offers a
        public, linter-friendly API for callers such as
        ``utils.game_manager_utils``.
        """
        return self._calculate_actual_round_count()

    def record_no_path_found_move(self) -> None:
        """Record a *NO_PATH_FOUND* sentinel.

        This marks a tick where the LLM explicitly stated that **no safe path
        exists**.  It is logged separately from EMPTY so the two counters do
        not interfere with each other.
        """
        self.stats.step_stats.no_path_found += 1
        self.steps += 1
        self.moves.append("NO_PATH_FOUND")
        self.round_manager.round_buffer.add_move("NO_PATH_FOUND")
  