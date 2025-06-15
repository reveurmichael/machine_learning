import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from utils.json_utils import NumPyJSONEncoder
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
        from config import MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED, MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED, MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED

        self.game_number = 0
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.apple_positions = []
        self.score = 0
        self.steps = 0
        self.last_move = None
        self.game_over = False
        self.game_end_reason = None
        self.snake_positions = []
        self.apple_position = None
        self.moves = []

        self.max_consecutive_empty_moves_allowed = MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED
        self.max_consecutive_something_is_wrong_allowed = MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED
        self.max_consecutive_invalid_reversals_allowed = MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED

        self.stats = GameStatistics()
        self.round_manager = RoundManager()

        self.consecutive_empty_moves_count = 0
        self.consecutive_something_is_wrong_count = 0
        self.consecutive_invalid_reversals_count = 0

    def start_new_round(self, apple_position) -> None:
        """Start a new round of moves."""
        self.round_manager.start_new_round(apple_position)

    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics."""
        move = normalize_direction(move)
        self.steps += 1
        self.stats.step_stats.valid += 1
        self.consecutive_empty_moves_count = 0
        self.consecutive_invalid_reversals_count = 0

        if apple_eaten:
            self.score += 1

        self.last_move = move
        self.moves.append(move)
        self.round_manager.round_buffer.add_move(move)

    def record_apple_position(self, position) -> None:
        """Record an apple position."""
        x, y = position
        self.apple_positions.append({"x": x, "y": y})
        self.round_manager.record_apple_position(position)

    def record_empty_move(self) -> None:
        """Record an empty move."""
        self.stats.step_stats.empty += 1
        self.steps += 1
        self.moves.append("EMPTY")
        self.round_manager.round_buffer.add_move("EMPTY")

    def record_invalid_reversal(self, attempted_move: str, current_direction: str) -> None:
        """Record an invalid reversal move."""
        self.stats.step_stats.invalid_reversals += 1
        self.steps += 1
        self.consecutive_invalid_reversals_count += 1
        self.moves.append("INVALID_REVERSAL")
        self.round_manager.round_buffer.add_move("INVALID_REVERSAL")

    def record_something_is_wrong_move(self) -> None:
        """Record an error move."""
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

    def record_llm_output(self, llm_output: str, is_primary: bool) -> None:
        """Records the raw output from an LLM."""
        self.round_manager.record_llm_output(llm_output, is_primary)

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
            "avg_primary_response_time": float(np.mean(primary_times)) if primary_times else 0.0,
            "avg_secondary_response_time": float(np.mean(secondary_times)) if secondary_times else 0.0,
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

    def get_error_stats(self) -> Dict[str, Any]:
        """Returns a dictionary with LLM error statistics."""
        return {
            "primary_llm_errors": self.stats.primary_llm_errors,
            "secondary_llm_errors": self.stats.secondary_llm_errors,
            "primary_error_rate": self.stats.primary_llm_errors / self.stats.primary_llm_requests if self.stats.primary_llm_requests > 0 else 0,
            "secondary_error_rate": self.stats.secondary_llm_errors / self.stats.secondary_llm_requests if self.stats.secondary_llm_requests > 0 else 0,
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
            # High-level outcome ------------------------------------------------
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": self.game_over,
            "game_end_reason": self.game_end_reason,
            "round_count": self.round_manager.round_count,
            
            # LLM configuration -------------------------------------------------
            "llm_info": {
                "primary_provider": primary_provider,
                "primary_model": primary_model,
                "parser_provider": parser_provider,
                "parser_model": parser_model,
            },
            
            # Timings / stats ---------------------------------------------------
            "time_stats": self.stats.time_stats.summary(),
            "prompt_response_stats": self.get_prompt_response_stats(),
            "token_stats": self.get_token_stats(),
            "step_stats": self.stats.step_stats.asdict(),
            "error_stats": self.get_error_stats(),
            # Misc metadata ----------------------------------------------------
            "metadata": {
                "timestamp": self.timestamp,
                "game_number": self.game_number,
                "round_count": self.round_manager.round_count,
                # Copy through any extra metadata the caller supplies.
                **kwargs.get("metadata", {}),
            },
            # Full replay data --------------------------------------------------
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
        with open(filepath, "w") as f:
            json.dump(summary_dict, f, cls=NumPyJSONEncoder, indent=4)
        return summary_dict

    @property
    def snake_length(self) -> int:
        """Returns the length of the snake."""
        return len(self.snake_positions)

    # ------------------------------------------------------------------
    # Delegating wrappers for GameStatistics
    # ------------------------------------------------------------------

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

    def record_primary_token_stats(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.stats.record_primary_token_stats(prompt_tokens, completion_tokens)

    def record_secondary_token_stats(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.stats.record_secondary_token_stats(prompt_tokens, completion_tokens)

    def record_primary_llm_error(self) -> None:
        self.stats.primary_llm_errors += 1

    def record_secondary_llm_error(self) -> None:
        self.stats.secondary_llm_errors += 1

    # ------------------------------------------------------------------
    # Continuation-mode helpers (needed by utils/continuation_utils.py)
    # ------------------------------------------------------------------

    def record_continuation(self, previous_session_data: Optional[dict] = None) -> None:
        """Mark this run as a continuation of a previous experiment.

        The old monolithic version just kept some metadata lists; we
        reproduce the same fields so that code reading summary.json later
        still finds them.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lazily create the attributes the first time we are called.
        if not hasattr(self, "is_continuation"):
            self.is_continuation = True
            self.continuation_count = 1
            self.continuation_timestamps = [timestamp]
            self.continuation_metadata = []
        else:
            self.continuation_count += 1
            self.continuation_timestamps.append(timestamp)

        meta = {
            "timestamp": timestamp,
            "continuation_number": self.continuation_count,
        }

        # If the caller passes a summary-dict from the previous run, keep
        # the compact stats block that the old code recorded.
        if previous_session_data and "game_count" in previous_session_data:
            meta["previous_session"] = {
                "total_games": previous_session_data.get("game_count", 0),
                "total_score": previous_session_data.get("total_score", 0),
                "total_steps": previous_session_data.get("total_steps", 0),
                "scores": previous_session_data.get("game_scores", []),
            }

        self.continuation_metadata.append(meta)

    def synchronize_with_summary_json(self, summary_data: Dict[str, Any]):
        """Pull tunable limits and step counters from an existing summary.json.

        Only the handful of fields that `utils.continuation_utils` relies on
        are copied; everything else remains unchanged.
        """
        if not summary_data:
            return

        # copy limit settings so the new session respects the old rules
        self.max_consecutive_empty_moves_allowed = summary_data.get(
            "max_consecutive_empty_moves_allowed",
            self.max_consecutive_empty_moves_allowed,
        )
        self.max_consecutive_invalid_reversals_allowed = summary_data.get(
            "max_consecutive_invalid_reversals_allowed",
            self.max_consecutive_invalid_reversals_allowed,
        )

        # step counters
        step_stats = summary_data.get("step_stats", {})
        self.stats.step_stats.valid = step_stats.get("valid_steps", self.stats.step_stats.valid)
        self.stats.step_stats.invalid_reversals = step_stats.get(
            "invalid_reversals", self.stats.step_stats.invalid_reversals
        ) 

    # --- Quick accessors required by utils/game_manager_utils ----

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

    # ------------------------------------------------------------------
    # Convenience accessors for session-level aggregation
    # ------------------------------------------------------------------

    @property
    def primary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the primary LLM."""
        return self.stats.primary_response_times

    @property
    def secondary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the secondary LLM."""
        return self.stats.secondary_response_times

    # ------------------------------------------------------------------
    # Time statistics view (used by game_manager_utils)
    # ------------------------------------------------------------------

    def get_time_stats(self) -> Dict[str, Any]:
        """Return wall-clock timings needed for session aggregation."""
        # No longer track movement/waiting breakdowns – just return the
        # coarse timing summary.
        return self.stats.time_stats.summary()

    # ------------------------------------------------------------------
    # Misc helpers expected by utils.game_manager_utils
    # ------------------------------------------------------------------

    def _calculate_actual_round_count(self) -> int:
        """Return the number of rounds that actually hold data."""
        return len([r for r in self.round_manager.rounds_data.values() if r])
