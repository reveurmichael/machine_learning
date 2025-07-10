"""Game data management with BaseClass architecture for clean task separation.

=== SINGLE SOURCE OF TRUTH FOR GAME STATE ===
This module defines the canonical game state structure used across ALL tasks.
BaseGameData contains ONLY generic attributes that are useful to any algorithm.

UNIVERSAL GAME STATE (Tasks 0-5):
- Core game state: score, steps, game_over, snake_positions, apple_position
- Move tracking: moves, current_game_moves, planned_moves
- Error counters: consecutive_invalid_reversals, consecutive_no_path_found
- Round management: round_manager (for planning algorithms)
- Statistics: stats (BaseGameStatistics instance)

LLM-SPECIFIC EXTENSIONS (Task-0 only):
- LLM counters: consecutive_empty_steps, consecutive_something_is_wrong
- LLM limits: max_consecutive_empty_moves_allowed, max_consecutive_something_is_wrong_allowed
- LLM statistics: stats (GameStatistics instance with token/response data)

=== ELEGANT JSON HANDLING ===
The generate_game_summary() method creates perfectly structured game_N.json files:
- Consistent schema across all tasks (BaseGameData ensures this)
- LLM-specific fields only appear in Task-0 JSON files
- Round data is automatically synchronized via round_manager
- Statistics are serialized using asdict() methods for consistency

=== TASK INHERITANCE EXAMPLES ===
```python
# Task-0 (LLM): Full GameData with LLM extensions
class GameData(BaseGameData):
    def __init__(self):
        super().__init__()
        self.stats = GameStatistics()  # LLM-specific stats

# Task-1 (Heuristics): Uses BaseGameData directly  
class HeuristicGameData(BaseGameData):
    def __init__(self):
        super().__init__()
        # Inherits: consecutive_invalid_reversals, consecutive_no_path_found
        # Does NOT inherit: consecutive_empty_steps, consecutive_something_is_wrong

# Task-2 (RL): Could extend BaseGameData for RL-specific state
class RLGameData(BaseGameData):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []  # RL-specific extension
```

=== JSON OUTPUT GUARANTEE ===
All game_N.json files follow the same schema for shared fields:
- step_stats contains identical field names
- detailed_history uses same apple_positions/moves format  
- metadata section is consistent across tasks
- Task-specific extensions appear as additional fields without conflicts
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from core.game_stats_manager import NumPyJSONEncoder
from utils.moves_utils import normalize_direction
from core.game_stats import BaseGameStatistics, GameStatistics
from core.game_rounds import RoundManager


# ------------------
# Generic game data base class (shared by all tasks) 
# ------------------


# This class is NOT Task0 specific.
class BaseGameData:
    """Base class for game data tracking - generic for all task types.
    
    Contains the core game state and move tracking functionality that
    all tasks (Task-0 through Task-5) will need. Follows SOLID principles
    by being open for extension but closed for modification.
    """
    
    def __init__(self) -> None:
        """Initialize the base game data tracking."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracking data to initial state."""
        from config.game_constants import (
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
        )
        
        # Core game state (generic for all tasks)
        self.game_number = 0
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_end_reason = None
        
        # Game board state (generic for all tasks)
        self.snake_positions = []
        self.apple_position = None
        self.apple_positions = []
        self.moves = []
        
        # Move limits common across tasks
        self.max_consecutive_invalid_reversals_allowed = (
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED
        )
        self.max_consecutive_no_path_found_allowed = (
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED
        )
        
        # Counter attributes for tracking consecutive moves (generic)
        self.consecutive_invalid_reversals = 0
        self.consecutive_no_path_found = 0
        self.no_path_found_steps = 0
        
        # Game flow control (generic for all tasks)
        self.need_new_plan = True
        self.planned_moves = []
        self.current_direction = None
        self.last_collision_type = None
        
        # Move tracking (generic for all tasks)
        self.move_index = 0
        self.total_moves = 0
        self.current_game_moves = []
        
        # Apple history tracking (generic for all tasks)
        self.apple_positions_history = []

        # ---------------------
        # Statistics (generic) and round tracking
        # ---------------------
        # Generic per-game statistics container (LLM-agnostic)
        self.stats = BaseGameStatistics()

        # Round tracking (optional but useful for all tasks)
        self.round_manager = RoundManager()
    
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics.
        
        This method is generic and will be used by all tasks.
        """
        move = normalize_direction(move)
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        
        if apple_eaten:
            self.score += 1
            
        self.moves.append(move)
        self.current_game_moves.append(move)
        
        # Reset consecutive counters on valid move
        if move not in ["INVALID_REVERSAL", "NO_PATH_FOUND"]:
            self.consecutive_invalid_reversals = 0
            self.consecutive_no_path_found = 0
        
        # ---------------------
        # Round-level bookkeeping (generic – safe for ALL tasks)
        # ---------------------
        # If a RoundManager is attached we mirror the move into the
        # per-round buffer.  This allows replay engines and analytics to
        # reconstruct the exact sequence of actions without scanning the
        # coarse *moves* array for round boundaries.
        if hasattr(self, "round_manager") and self.round_manager:
            buffer = getattr(self.round_manager, "round_buffer", None)
            if buffer is not None:
                buffer.add_move(move)
        
        # Call subclass hook for valid moves
        if move not in ["INVALID_REVERSAL", "NO_PATH_FOUND"]:
            self._record_valid_step()
    
    def record_apple_position(self, position) -> None:
        """Record an apple position.
        
        This method is generic and will be used by all tasks.  It also seeds
        the current round so replay files always have an apple reference.
        """
        x, y = position
        apple_data = {"x": x, "y": y}
        self.apple_positions.append(apple_data)
        self.apple_positions_history.append(apple_data)
        self.apple_position = position
        
        # Keep RoundManager in sync so replays & per-round stats work for
        # every task that chooses to use rounds.
        if hasattr(self, "round_manager") and self.round_manager:
            self.round_manager.record_apple_position(position)
    
    def record_game_end(self, reason: str) -> None:
        """Record the end of a game."""
        self.game_over = True
        self.game_end_reason = reason

    def record_invalid_reversal(self) -> None:
        """Record an invalid reversal move (generic for all tasks)."""
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        self.moves.append("INVALID_REVERSAL")
        self.current_game_moves.append("INVALID_REVERSAL")
        self.consecutive_invalid_reversals += 1
        
        # Call subclass hook
        self._record_invalid_reversal_step()
    
    def record_no_path_found_move(self) -> None:
        """Record a NO_PATH_FOUND move (generic for all tasks)."""
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        self.no_path_found_steps += 1
        self.moves.append("NO_PATH_FOUND")
        self.current_game_moves.append("NO_PATH_FOUND")
        self.consecutive_no_path_found += 1
        
        # Call subclass hook
        self._record_no_path_found_step()

    @property
    def snake_length(self) -> int:
        """Returns the length of the snake."""
        return len(self.snake_positions)
    
    @property
    def head_position(self):
        """Returns the head position of the snake."""
        return self.snake_positions[0] if self.snake_positions else None
    
    def get_basic_game_state(self) -> Dict[str, Any]:
        """Get basic game state information (generic for all tasks)."""
        return {
            "game_number": self.game_number,
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "head_position": self.head_position,
            "apple_position": self.apple_position,
            "game_over": self.game_over,
            "game_end_reason": self.game_end_reason,
        }
    
    def reset_game_data(self) -> None:
        """Reset data for a new game (keeping session-level data)."""
        game_num = self.game_number
        self.reset()
        self.game_number = game_num + 1
        self.current_game_moves = []
    
    # Hook methods for subclasses to override
    def _record_valid_step(self) -> None:
        """Hook for recording valid steps - override in subclasses."""
        pass
    
    def _record_step_start_time(self) -> None:
        """Hook for recording step timing - override in subclasses."""
        pass
    
    def _record_step_end_time(self) -> None:
        """Hook for recording step timing - override in subclasses."""
        pass
    
    def _record_invalid_reversal_step(self) -> None:
        """Hook for recording invalid reversal statistics - override in subclasses."""
        pass
    
    def _record_no_path_found_step(self) -> None:
        """Hook for recording no path found statistics - override in subclasses."""
        pass

    def start_new_round(self, apple_position=None) -> None:
        """Public helper to advance to the next round (no-op if RoundManager absent).

        Tasks that maintain the *round* concept (LLM policy, heuristic
        agents with look-ahead plans, etc.) should call this whenever a new
        top-level plan is generated.
        """
        if hasattr(self, "round_manager") and self.round_manager:
            self.round_manager.start_new_round(apple_position)

# This class is Task0 specific.
class GameData(BaseGameData):
    """LLM-specific game data tracking for Task-0."""
    
    def __init__(self) -> None:
        """Initialize the LLM-specific game data tracking."""
        super().__init__()
        
        # LLM-specific components (RoundManager already initialised by base)
        self.stats = GameStatistics()
    
    def reset(self) -> None:
        """Reset all tracking data to initial state."""
        super().reset()
        
        from config.game_constants import (
            MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
        )

        # Task-0-specific move limits
        self.max_consecutive_empty_moves_allowed = MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED
        self.max_consecutive_something_is_wrong_allowed = MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED

        # Task-0-specific consecutive counters
        self.consecutive_empty_steps = 0
        self.consecutive_something_is_wrong = 0

        # Reset LLM-specific components (RoundManager reset lives in base)
        self.stats = GameStatistics()
    
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics."""
        # Call base class method
        super().record_move(move, apple_eaten)
        
        # LLM-specific tracking
        self.stats.step_stats.valid += 1
        
        # Reset LLM-specific consecutive counters on valid move
        if move not in ["EMPTY", "SOMETHING_IS_WRONG"]:
            self.consecutive_empty_steps = 0
            self.consecutive_something_is_wrong = 0
    
    def record_apple_position(self, position) -> None:
        """Record an apple position."""
        # Call base class method
        super().record_apple_position(position)
        
        # LLM-specific tracking
        self.round_manager.record_apple_position(position)
        
    def start_new_round(self, apple_position=None) -> None:
        """Start a new round of moves."""
        self.round_manager.start_new_round(apple_position)
    
    def record_empty_move(self) -> None:
        """Record an *EMPTY* sentinel.

        EMPTY represents a tick where the snake stayed in place because **no
        valid move was produced at all**.  It is *not* used for NO_PATH_FOUND
        or parser/LLM failures – those have their own dedicated sentinel move
        names so statistics stay strictly independent.
        """
        self.stats.step_stats.empty += 1
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        self.moves.append("EMPTY")
        self.current_game_moves.append("EMPTY")
        self.round_manager.round_buffer.add_move("EMPTY")
        self.consecutive_empty_steps += 1
    
    def _record_invalid_reversal_step(self) -> None:
        """Hook implementation for recording invalid reversal statistics."""
        self.stats.step_stats.invalid_reversals += 1
        self.round_manager.round_buffer.add_move("INVALID_REVERSAL")
    
    def record_something_is_wrong_move(self) -> None:
        """Record a *SOMETHING_IS_WRONG* sentinel.

        Indicates a parser/LLM failure (e.g. JSON could not be extracted).
        Logged independently so that EMPTY and NO_PATH_FOUND statistics are
        unaffected by error handling.
        """
        self.stats.step_stats.something_wrong += 1
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        self.moves.append("SOMETHING_IS_WRONG")
        self.current_game_moves.append("SOMETHING_IS_WRONG")
        self.round_manager.round_buffer.add_move("SOMETHING_IS_WRONG")
        self.consecutive_something_is_wrong += 1

    def _record_no_path_found_step(self) -> None:
        """Hook implementation for recording no path found statistics."""
        self.stats.step_stats.no_path_found += 1
        self.round_manager.round_buffer.add_move("NO_PATH_FOUND")

    def record_game_end(self, reason: str) -> None:
        """Record the end of a game."""
        if not self.game_over:
            self.stats.time_stats.record_end_time()
        super().record_game_end(reason)

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
            # High-level outcome ---------------------
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": self.game_over,
            "game_end_reason": self.game_end_reason,
            "round_count": self.round_manager.round_count,
            # LLM configuration ---------------------
            "llm_info": {
                "primary_provider": primary_provider,
                "primary_model": primary_model,
                "parser_provider": parser_provider,
                "parser_model": parser_model,
            },
            # Timings / stats ---------------------
            "time_stats": self.stats.time_stats.asdict(),
            "prompt_response_stats": self.get_prompt_response_stats(),
            "token_stats": self.get_token_stats(),
            "step_stats": self.stats.step_stats.asdict(),
            # Misc metadata ---------------------
            "metadata": {
                "timestamp": self.timestamp,
                "game_number": self.game_number,
                "round_count": self.round_manager.round_count,
                # Copy through any extra metadata the caller supplies.
                **kwargs.get("metadata", {}),
            },
            # Full replay data ---------------------
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

    # ---------------------
    # Delegating wrappers for GameStatistics
    # ---------------------

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

    # ---------------------
    # Continuation-mode helpers (needed by utils/continuation_utils.py)
    # ---------------------

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

    # ---------------------
    # Quick accessors required by game_manager_helper
    # ---------------------

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

    # ---------------------
    # Convenience accessors for session-level aggregation
    # ---------------------

    @property
    def primary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the primary LLM."""
        return self.stats.primary_response_times

    @property
    def secondary_response_times(self) -> List[float]:
        """List of response-time durations (seconds) from the secondary LLM."""
        return self.stats.secondary_response_times

    # ---------------------
    # Time statistics view (used by game_manager_helper)
    # ---------------------

    def get_time_stats(self) -> Dict[str, Any]:
        """Return wall-clock timings needed for session aggregation."""
        # No longer track movement/waiting breakdowns – just return the
        # coarse timing summary.
        return self.stats.time_stats.asdict()

    # ---------------------
    # Misc helpers expected by game_manager_helper
    # ---------------------

    def _calculate_actual_round_count(self) -> int:
        """Return the number of rounds that actually hold data."""
        return len([r for r in self.round_manager.rounds_data.values() if r])

    # ---------------------
    # Public wrapper – prefer this over direct access to the underscore
    # helper so external modules avoid pylint W0212.
    # ---------------------

    def get_round_count(self) -> int:
        """Return the number of rounds that actually contain gameplay data.

        This simply delegates to the internal computation method but offers a
        public, linter-friendly API for callers such as
        ``core.game_manager_helper``.
        """
        return self._calculate_actual_round_count()
  