from typing import Dict, List

from core.game_stats import RoundBuffer

class RoundManager:
    """Collect and persist per-round data throughout a game."""

    def __init__(self) -> None:
        self.round_count: int = 1
        self.rounds_data: Dict[int, dict] = {}
        self.round_buffer: RoundBuffer = RoundBuffer(number=1)

    def start_new_round(self, apple_position) -> None:
        """Start a new round of moves."""
        self.flush_buffer()
        self.round_count += 1
        self.round_buffer = RoundBuffer(number=self.round_count)
        self.round_buffer.set_apple(list(apple_position) if apple_position else None)

    def record_apple_position(self, position) -> None:
        """Record an apple position."""
        x, y = position
        self.round_buffer.set_apple([x, y])
        round_data = self._get_or_create_round_data(self.round_count)
        round_data["apple_position"] = [x, y]

    def record_llm_output(self, llm_output: str, is_primary: bool) -> None:
        """Records the raw output from an LLM for the current round."""
        self.round_buffer.add_llm_output(llm_output, is_primary)
        round_data = self._get_or_create_round_data(self.round_count)
        key = "primary_llm_output" if is_primary else "secondary_llm_output"
        round_data.setdefault(key, []).append(llm_output)

    def record_parsed_llm_response(self, response, is_primary: bool) -> None:
        """Records the parsed response from an LLM for the current round."""
        self.round_buffer.add_parsed_response(response, is_primary)
        round_data = self._get_or_create_round_data(self.round_count)
        key = "primary_parsed_response" if is_primary else "secondary_parsed_response"
        round_data[key] = response

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
            "primary_llm_output": self.round_buffer.primary_llm_output,
            "primary_parsed_response": self.round_buffer.primary_parsed_response,
            "secondary_llm_output": self.round_buffer.secondary_llm_output,
            "secondary_parsed_response": self.round_buffer.secondary_parsed_response,
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
    