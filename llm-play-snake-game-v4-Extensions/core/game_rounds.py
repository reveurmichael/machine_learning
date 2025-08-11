from __future__ import annotations

from typing import Dict, List, Sequence, Optional, Any

try:
    import numpy as np
    NDArray = np.ndarray  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Safeguard for docs or constrained envs
    NDArray = object

from core.game_stats import RoundBuffer

# Export names for easy import elsewhere
__all__ = [
    "BaseRoundManager",
    "RoundManager",
]


class BaseRoundManager:
    """Collect and persist **per-round** data – shared across *all* tasks (0-5).

    Why *rounds* are first-class:
        • **Task-0** (LLM planning) – one LLM plan → one round.
        • **Task-1** (heuristic) – one heuristic path-finder invocation → one round.
        • **Task-2** (ML policy) – one greedy rollout / sub-episode → one round.
        • **Task-3** (RL) – one curriculum "phase" → one round.
        • **Task-4/5** (hybrid or meta-learning) – still benefit from grouping
          a *plan* and its execution window.

    Hence the abstraction is here to stay for every planned milestone; the
    class sits in *core* and is intentionally **LLM-agnostic** so downstream
    tasks can extend it with domain-specific fields without modifying the
    shared logic.

    A *round* groups:
        1. A *plan* (ordered list of proposed moves).
        2. The actual moves executed until either a new plan is requested or
           the game ends.
        3. Optional per-round statistics (time, tokens, rewards, …).
    """

    def __init__(self) -> None:
        self.round_count: int = 1
        self.rounds_data: Dict[int, dict] = {}
        self.round_buffer: RoundBuffer = RoundBuffer(number=1)

    # ---------------------
    # Public API
    # ---------------------

    def start_new_round(self, apple_position: Optional[Sequence[int] | NDArray]) -> None:
        """Flush the current buffer, bump the counter and initialise a new round.

        This is now the **single** entry-point for beginning a new round
        which means callers (GameManager, tests, etc.) never need to deal
        with the implementation details (flush, counter, seeding).
        """
        # Persist pending data before flipping the page
        self.flush_buffer()

        # Create new buffer
        self.round_count += 1
        self.round_buffer = RoundBuffer(number=self.round_count)

        # Seed with the current apple so JSON never ends up with nulls
        self.round_buffer.set_apple(self._to_list_or_none(apple_position))

    def record_apple_position(self, position: Sequence[int] | NDArray) -> None:
        """Record a freshly spawned apple position."""
        pos_list = self._to_list_or_none(position)
        self.round_buffer.set_apple(pos_list)

        round_data = self._get_or_create_round_data(self.round_count)
        round_data["apple_position"] = pos_list

    def record_planned_moves(self, moves: List[str]) -> None:
        """Store the latest plan, replacing any previous entries for this round.

        The LLM may resend the *same* plan multiple times while we are still
        executing it (e.g. due to retries).  Overwriting avoids exponential
        duplication of the list observed in JSON outputs.
        """
        if moves:
            # Reset to the fresh plan instead of extending
            self.round_buffer.planned_moves = list(moves)

    def sync_round_data(self) -> None:
        """Synchronize the in-progress round buffer with the persistent mapping.

        This keeps the JSON structure used by replays up-to-date after every
        tick.  We purposefully mirror the **flat array** style (`moves` list)
        adopted elsewhere so historical tooling (replay engine, dashboards)
        remains fully compatible.
        """
        if not self.round_buffer:
            return

        current_round_dict = self._get_or_create_round_data(self.round_buffer.number)
        current_round_dict.update({
            "round": self.round_buffer.number,
            "apple_position": self.round_buffer.apple_position,
        })
        current_round_dict["planned_moves"] = list(self.round_buffer.planned_moves)
        if self.round_buffer.moves:
            current_round_dict.setdefault("moves", []).extend(self.round_buffer.moves)
            # After persisting, reset the buffer so only *new* moves accumulate
            self.round_buffer.moves.clear()

    def flush_buffer(self) -> None:
        """Flushes the round buffer.

        The helper exists so callers never have to worry about whether the
        current buffer holds data or not – they simply call *flush* and this
        method takes care of the conditional write.
        """
        if self.round_buffer and not self.round_buffer.is_empty():
            self.sync_round_data()
            self.round_buffer = None  # type: ignore[assignment]

    # ---------------------
    # Extension-friendly public helpers
    # ---------------------

    def record_round_game_state(self, state: dict) -> None:
        """Attach an arbitrary game_state payload to the current round.

        This provides a stable public API so extensions never need to touch
        private methods when storing immutable pre-/post-move snapshots.
        """
        if not state:
            return
        number = self.round_buffer.number if self.round_buffer else self.round_count
        round_data = self._get_or_create_round_data(number)
        round_data["game_state"] = state

    def get_current_round_number(self) -> int:
        """Return the current round number from the buffer if available."""
        return self.round_buffer.number if self.round_buffer else self.round_count

    def _get_or_create_round_data(self, round_num: int) -> dict:
        """Get or create round data dictionary."""
        return self.rounds_data.setdefault(round_num, {"round": round_num})

    def get_ordered_rounds_data(self) -> Dict[int, dict]:
        """Returns the rounds_data dictionary with keys sorted numerically."""
        sorted_keys = sorted(self.rounds_data.keys())
        return {key: self.rounds_data[key] for key in sorted_keys}

    # ---------------------
    # Internals
    # ---------------------

    @staticmethod
    def _to_list_or_none(pos: Optional[Sequence[int] | NDArray]) -> Optional[list[int]]:
        """Return *pos* as a plain Python list or ``None``.

        Accepts native lists/tuples as well as NumPy arrays to spare callers
        from handling that conversion in every call-site.
        """
        if pos is None:
            return None
        if hasattr(pos, "tolist"):
            return pos.tolist()  # type: ignore[return-value]
        return list(pos)  # type: ignore[arg-type]


class RoundManager(BaseRoundManager):
    """LLM-enhanced round manager.

    Currently it only adds a stub method expected by GameData; actual LLM
    bookkeeping can be layered on later without affecting the generic base.
    """

    # ---------------------
    # LLM-specific helpers (optional)
    # ---------------------
    def record_parsed_llm_response(self, response: Any, is_primary: bool) -> None:  # noqa: D401 – simple proxy
        """Placeholder; extend with structured logging if needed."""
        # For now, we simply attach the raw response to the round buffer for
        # potential debug use.  Downstream analytics can ignore or utilise it.
        if not hasattr(self, "round_buffer") or not self.round_buffer:
            return

        key = "primary_llm_responses" if is_primary else "secondary_llm_responses"
        setattr(self.round_buffer, key, getattr(self.round_buffer, key, []))
        getattr(self.round_buffer, key).append(response)
    