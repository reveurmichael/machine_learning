from __future__ import annotations

from typing import Dict, List, Sequence, Optional

try:
    import numpy as np
    NDArray = np.ndarray  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Safeguard for docs or constrained envs
    NDArray = object

from core.game_stats import RoundBuffer

class RoundManager:
    """Collect and persist per-round data throughout a game."""

    def __init__(self) -> None:
        self.round_count: int = 1
        self.rounds_data: Dict[int, dict] = {}
        self.round_buffer: RoundBuffer = RoundBuffer(number=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_new_round(self, apple_position: Optional[Sequence[int] | NDArray]) -> None:
        """Flush the current buffer, bump the counter and initialise a
        fresh :class:`RoundBuffer` seeded with the given *apple_position*.

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
        """Synchronize the in-progress round buffer with the persistent `rounds_data` mapping."""
        if not self.round_buffer:
            return

        current_round_dict = self._get_or_create_round_data(self.round_buffer.number)
        current_round_dict.update({
            "round": self.round_buffer.number,
            "apple_position": self.round_buffer.apple_position,
        })
        # Planned moves should reflect the *latest* plan only.  Overwrite
        # instead of extending so repeated syncs during the same round don't
        # duplicate identical plans.
        current_round_dict["planned_moves"] = list(self.round_buffer.planned_moves)
        
        # Append executed moves in order, preserving duplicates to faithfully
        # mirror the actual gameplay sequence.  This is essential for accurate
        # replays and per-round step counts.
        current_round_dict.setdefault("moves", []).extend(self.round_buffer.moves)

    def flush_buffer(self) -> None:
        """Flushes the round buffer."""
        if self.round_buffer and not self.round_buffer.is_empty():
            self.sync_round_data()
            self.round_buffer = None

    def _get_or_create_round_data(self, round_num: int) -> dict:
        """Get or create round data dictionary."""
        return self.rounds_data.setdefault(round_num, {"round": round_num})

    def get_ordered_rounds_data(self) -> Dict[int, dict]:
        """Returns the rounds_data dictionary with keys sorted numerically."""
        sorted_keys = sorted(self.rounds_data.keys())
        return {key: self.rounds_data[key] for key in sorted_keys}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list_or_none(pos: Optional[Sequence[int] | NDArray]) -> Optional[list[int]]:
        """Return *pos* as a plain Python list or ``None``.

        Accepts native lists/tuples as well as NumPy arrays to spare callers
        from handling that conversion in every call-site.
        """
        if pos is None:
            return None
        # NumPy array – convert
        if hasattr(pos, "tolist"):
            return pos.tolist()  # type: ignore[return-value]
        # Fallback – assume Sequence[int]
        return list(pos)  # type: ignore[arg-type]
    