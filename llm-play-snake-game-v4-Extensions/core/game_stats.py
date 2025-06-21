"""Game statistics with BaseClass architecture for future extensibility.

=== SINGLE SOURCE OF TRUTH FOR STATISTICS ===
This module defines the canonical statistics structure used across ALL tasks.
The BaseClass architecture enables clean separation:

UNIVERSAL STATISTICS (Tasks 0-5):
- BaseStepStats: valid, invalid_reversals, no_path_found
- BaseGameStatistics: time_stats, step_stats (base versions)
- TimeStats: Wall-clock timing for any algorithm type

LLM-SPECIFIC STATISTICS (Task-0 only):
- StepStats: Extends BaseStepStats with empty, something_wrong
- GameStatistics: Extends BaseGameStatistics with LLM token/response data

=== WHY THIS ARCHITECTURE MATTERS ===
1. **Single Source of Truth**: All statistics definitions live here
2. **Clean Inheritance**: Tasks 1-5 inherit BaseXxx classes (no LLM pollution)
3. **JSON Consistency**: asdict() methods ensure identical JSON schema
4. **Future-Proof**: New tasks can extend base classes without breaking existing code

=== TASK USAGE EXAMPLES ===
```python
# Task-0 (LLM): Uses full GameStatistics with all features
game_data.stats = GameStatistics()  # Gets empty_steps, token_stats, etc.

# Task-1 (Heuristics): Uses BaseGameStatistics only  
game_data.stats = BaseGameStatistics()  # Gets valid, invalid_reversals, no_path_found

# Task-2 (RL): Could extend BaseGameStatistics for RL-specific metrics
class RLGameStatistics(BaseGameStatistics):
    episode_rewards: List[float] = field(default_factory=list)
```

=== JSON OUTPUT GUARANTEE ===
All asdict() methods produce identical JSON structure for the same fields,
ensuring game_N.json and summary.json files are compatible across tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import List, Optional

__all__ = [
    "TimeStats",
    "TokenStats",
    "BaseStepStats",
    "BaseGameStatistics",
    "RoundData",
    "GameStatistics",
    "StepStats",
]


@dataclass
class TimeStats:
    """Lightweight container for wall-clock timings."""

    start_time: float
    llm_communication_time: float = 0.0
    end_time: float | None = None

    # ---------------------
    # Mutation helpers
    # ---------------------
    def add_llm_comm(self, delta: float) -> None:
        """Accumulate LLM communication seconds."""
        self.llm_communication_time += delta

    # ---------------------
    # JSON-ready view
    # ---------------------
    def asdict(self) -> dict:
        end = self.end_time or time.time()
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": end - self.start_time,
            "llm_communication_time": self.llm_communication_time,
        }

    # ---------------------
    # Simple setter used by GameData.record_game_end
    # ---------------------
    def record_end_time(self) -> None:
        self.end_time = time.time()


@dataclass
class TokenStats:
    """Prompt / completion token pair for a single request."""

    prompt_tokens: int
    completion_tokens: int

    # Cached property avoids recomputation when accessed repeatedly
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def asdict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# ---------------------
# Base statistics – generic across all tasks (Task-0 … Task-5)
# ---------------------

@dataclass
class BaseStepStats:
    """Minimal move-type counters shared by all agent types.

    EMPTY / SOMETHING_IS_WRONG are *LLM-specific* and therefore live in the
    subclass ``StepStats`` below.  Every task, however, can benefit from
    tracking how many actions were *valid* versus "invalid reversal" or the
    agent explicitly admitting "NO_PATH_FOUND".
    """

    valid: int = 0
    invalid_reversals: int = 0
    no_path_found: int = 0

    def asdict(self) -> dict:  # JSON-friendly view
        return {
            "valid_steps": self.valid,
            "invalid_reversals": self.invalid_reversals,
            "no_path_found_steps": self.no_path_found,
        }


# ---------------------
# LLM-enhanced statistics (Task-0 only – extends the base)
# ---------------------

@dataclass
class StepStats(BaseStepStats):
    """Track various move-type counts within a game."""

    empty: int = 0  # LLM could not output *any* move
    something_wrong: int = 0  # Parser/LLM failure (invalid JSON, etc.)

    # ``valid``, ``invalid_reversals`` and ``no_path_found`` inherited

    def asdict(self) -> dict:  # keep JSON-friendly structure
        base = super().asdict()
        base.update({
            "empty_steps": self.empty,
            "something_is_wrong_steps": self.something_wrong,
        })
        return base


# ---------------------
# Base game-level statistics (generic for all tasks)
# ---------------------

@dataclass
class BaseGameStatistics:
    """Lightweight, provider-agnostic statistics for one game session.

    It captures *time* and *step* counters that are universally useful,
    leaving heavy LLM-centric fields (token counts, response times, etc.) to
    the subclass ``GameStatistics`` so they remain entirely opt-in.
    """

    time_stats: "TimeStats" = field(
        default_factory=lambda: TimeStats(start_time=time.time())
    )
    step_stats: BaseStepStats = field(default_factory=BaseStepStats)

    # ---------------------
    # Convenience helpers (generic – safe for any agent type)
    # ---------------------
    @property
    def valid_steps(self) -> int:
        return self.step_stats.valid

    @property
    def invalid_reversals(self) -> int:
        return self.step_stats.invalid_reversals

    @property
    def no_path_found_steps(self) -> int:
        return self.step_stats.no_path_found

    def asdict(self) -> dict:
        return {
            "time_stats": self.time_stats.asdict(),
            "step_stats": self.step_stats.asdict(),
        }


# ---------------------
# Higher-level containers (still opt-in; not yet wired into GameData)
# ---------------------

@dataclass
class RoundData:
    """Statistics captured for a single prompt/response round."""

    apple_position: Optional[list[int]] = None  # [x, y]
    moves: List[str] = field(default_factory=list)
    primary_response_times: List[float] = field(default_factory=list)
    secondary_response_times: List[float] = field(default_factory=list)
    primary_token_stats: List[dict] = field(default_factory=list)
    secondary_token_stats: List[dict] = field(default_factory=list)
    planned_moves: List[str] = field(default_factory=list)

    def asdict(self) -> dict:
        return {
            "apple_position": self.apple_position,
            "moves": self.moves,
            "planned_moves": self.planned_moves,
            "primary_response_times": self.primary_response_times,
            "secondary_response_times": self.secondary_response_times,
            "primary_token_stats": self.primary_token_stats,
            "secondary_token_stats": self.secondary_token_stats,
        }


@dataclass
class RoundBuffer:
    """Mutable container for data being collected for the *current* round."""

    number: int
    apple_position: Optional[list[int]] = None
    moves: List[str] = field(default_factory=list)
    planned_moves: List[str] = field(default_factory=list)
    primary_times: List[float] = field(default_factory=list)
    secondary_times: List[float] = field(default_factory=list)

    # ------------- convenience helpers ------------------
    def add_move(self, move: str) -> None:
        self.moves.append(move)

    def set_apple(self, pos: list[int]) -> None:
        self.apple_position = pos

    # Helper used by RoundManager.flush_buffer()
    def is_empty(self) -> bool:
        """Return True if nothing noteworthy has been recorded yet."""
        return not (self.moves or self.planned_moves)

@dataclass
class GameStatistics(BaseGameStatistics):
    """Collects step/token/time stats for a single game session."""

    # Override with richer step counters
    step_stats: StepStats = field(default_factory=StepStats)

    # Response times 
    primary_response_times: list[float] = field(default_factory=list)
    secondary_response_times: list[float] = field(default_factory=list)

    # Token stats
    primary_token_stats: list[TokenStats] = field(default_factory=list)
    secondary_token_stats: list[TokenStats] = field(default_factory=list)

    primary_total_tokens: int = 0
    primary_total_prompt_tokens: int = 0
    primary_total_completion_tokens: int = 0
    primary_avg_total_tokens: float = 0
    primary_avg_prompt_tokens: float = 0
    primary_avg_completion_tokens: float = 0

    secondary_total_tokens: int = 0
    secondary_total_prompt_tokens: int = 0
    secondary_total_completion_tokens: int = 0
    secondary_avg_total_tokens: float = 0
    secondary_avg_prompt_tokens: float = 0
    secondary_avg_completion_tokens: float = 0

    primary_llm_requests: int = 0
    secondary_llm_requests: int = 0

    last_action_time: float | None = None

    # ---------------------
    # Timers
    # ---------------------
    def record_llm_communication_start(self):
        self.last_action_time = time.perf_counter()

    def record_llm_communication_end(self):
        if self.last_action_time is not None:
            self.time_stats.add_llm_comm(time.perf_counter() - self.last_action_time)
            self.last_action_time = None

    # ---------------------
    # Response-time accumulators
    # ---------------------
    def record_primary_response_time(self, duration: float):
        self.primary_response_times.append(duration)

    def record_secondary_response_time(self, duration: float):
        self.secondary_response_times.append(duration)

    # ---------------------
    # Token-usage accumulators
    # ---------------------
    def _update_primary_averages(self):
        if self.primary_llm_requests:
            self.primary_avg_prompt_tokens = self.primary_total_prompt_tokens / self.primary_llm_requests
            self.primary_avg_completion_tokens = self.primary_total_completion_tokens / self.primary_llm_requests
            self.primary_avg_total_tokens = self.primary_total_tokens / self.primary_llm_requests

    def _update_secondary_averages(self):
        if self.secondary_llm_requests:
            self.secondary_avg_prompt_tokens = self.secondary_total_prompt_tokens / self.secondary_llm_requests
            self.secondary_avg_completion_tokens = self.secondary_total_completion_tokens / self.secondary_llm_requests
            self.secondary_avg_total_tokens = self.secondary_total_tokens / self.secondary_llm_requests

    def record_primary_token_stats(self, prompt_tokens: int, completion_tokens: int):
        self.primary_token_stats.append(TokenStats(prompt_tokens, completion_tokens))
        self.primary_llm_requests += 1
        self.primary_total_prompt_tokens += prompt_tokens
        self.primary_total_completion_tokens += completion_tokens
        self.primary_total_tokens = self.primary_total_prompt_tokens + self.primary_total_completion_tokens
        self._update_primary_averages()

    def record_secondary_token_stats(self, prompt_tokens: int, completion_tokens: int):
        self.secondary_token_stats.append(TokenStats(prompt_tokens, completion_tokens))
        self.secondary_llm_requests += 1
        self.secondary_total_prompt_tokens += prompt_tokens
        self.secondary_total_completion_tokens += completion_tokens
        self.secondary_total_tokens = self.secondary_total_prompt_tokens + self.secondary_total_completion_tokens
        self._update_secondary_averages() 
