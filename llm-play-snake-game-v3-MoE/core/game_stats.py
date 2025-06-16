from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import List, Optional

__all__ = [
    "TimeStats",
    "TokenStats",
    "RoundData",
    "GameStatistics",
]


@dataclass
class TimeStats:
    """Lightweight container for wall-clock timings."""

    start_time: float
    llm_communication_time: float = 0.0
    end_time: float | None = None

    # -------------------------------------------
    # Mutation helpers
    # -------------------------------------------
    def add_llm_comm(self, delta: float) -> None:
        """Accumulate LLM communication seconds."""
        self.llm_communication_time += delta

    # -------------------------------------------
    # JSON-ready view
    # -------------------------------------------
    def asdict(self) -> dict:
        end = self.end_time or time.time()
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": end - self.start_time,
            "llm_communication_time": self.llm_communication_time,
        }

    # -------------------------------------------
    # Simple setter used by GameData.record_game_end
    # -------------------------------------------
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


# ----------------------------------------
# Higher-level containers (still opt-in; not yet wired into GameData)
# ----------------------------------------

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
class StepStats:
    """Track various move-type counts within a game."""

    valid: int = 0
    empty: int = 0
    something_wrong: int = 0
    invalid_reversals: int = 0

    def asdict(self) -> dict:  # keep JSON-friendly structure
        return {
            "valid_steps": self.valid,
            "empty_steps": self.empty,
            "something_is_wrong_steps": self.something_wrong,
            "invalid_reversals": self.invalid_reversals,
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
    primary_tokens: List[TokenStats] = field(default_factory=list)
    secondary_tokens: List[TokenStats] = field(default_factory=list)

    # Raw LLM outputs and parsed responses captured during the round
    primary_llm_output: List[str] = field(default_factory=list)
    secondary_llm_output: List[str] = field(default_factory=list)
    primary_parsed_response: str | None = None
    secondary_parsed_response: str | None = None

    # ------------- convenience helpers ------------------
    def add_move(self, move: str) -> None:
        self.moves.append(move)

    def set_apple(self, pos: list[int]) -> None:
        self.apple_position = pos

    def add_llm_output(self, text: str, is_primary: bool = True) -> None:
        """Store raw LLM output for this round."""
        if is_primary:
            self.primary_llm_output.append(text)
        else:
            self.secondary_llm_output.append(text)

    def add_parsed_response(self, response: str | dict, is_primary: bool = True) -> None:
        """Store the parsed response for this round (keeps latest)."""
        if is_primary:
            self.primary_parsed_response = response
        else:
            self.secondary_parsed_response = response

    def add_moves(self, moves: list[str]):
        """Append a list of planned moves (string directions)."""
        if moves:
            self.planned_moves.extend(moves)

    # Helper used by RoundManager.flush_buffer()
    def is_empty(self) -> bool:
        """Return True if nothing noteworthy has been recorded yet."""
        return not (self.moves or self.primary_llm_output or self.secondary_llm_output or
                    self.primary_parsed_response or self.secondary_parsed_response or
                    self.planned_moves)

    def flush(self) -> RoundData:
        """Return an immutable RoundData and clear internal lists."""
        rd = RoundData(
            apple_position=self.apple_position,
            moves=self.moves.copy(),
            planned_moves=self.planned_moves.copy(),
            primary_response_times=self.primary_times.copy(),
            secondary_response_times=self.secondary_times.copy(),
            primary_token_stats=[t.asdict() for t in self.primary_tokens],
            secondary_token_stats=[t.asdict() for t in self.secondary_tokens],
        )
        # reset for reuse (optional)
        self.moves.clear()
        self.planned_moves.clear()
        self.primary_times.clear()
        self.secondary_times.clear()
        self.primary_tokens.clear()
        self.secondary_tokens.clear()
        self.apple_position = None
        return rd


@dataclass
class GameStatistics:
    """Collects step/token/time stats for a single game session."""

    time_stats: TimeStats = field(default_factory=lambda: TimeStats(start_time=time.time()))
    step_stats: StepStats = field(default_factory=StepStats)

    # Response times -------------------------------------------
    primary_response_times: list[float] = field(default_factory=list)
    secondary_response_times: list[float] = field(default_factory=list)

    # Token stats ----------------------------------------------
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

    # -------------------------------
    # Timers
    # -------------------------------
    def record_llm_communication_start(self):
        self.last_action_time = time.perf_counter()

    def record_llm_communication_end(self):
        if self.last_action_time is not None:
            self.time_stats.add_llm_comm(time.perf_counter() - self.last_action_time)
            self.last_action_time = None

    # -------------------------------
    # Response-time accumulators
    # -------------------------------
    def record_primary_response_time(self, duration: float):
        self.primary_response_times.append(duration)

    def record_secondary_response_time(self, duration: float):
        self.secondary_response_times.append(duration)

    # -------------------------------
    # Token-usage accumulators
    # -------------------------------
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