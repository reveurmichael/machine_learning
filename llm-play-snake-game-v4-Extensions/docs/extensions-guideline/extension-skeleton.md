### Extension Skeleton (Using Streamlined Core)

This skeleton shows how to write a clean, newly shipped extension manager leveraging the core helpers. No legacy patterns, no private API access.

```python
# my_extension/game_manager.py
from __future__ import annotations
import argparse
from datetime import datetime
from typing import Any, Dict

from core.game_manager import BaseGameManager
from .game_logic import MyExtensionGameLogic  # your logic subclass of BaseGameLogic

class MyExtensionGameManager(BaseGameManager):
    GAME_LOGIC_CLS = MyExtensionGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any | None = None) -> None:
        super().__init__(args)
        self.agent = agent
        self.session_start = datetime.now()
        self.game_scores: list[int] = []
        self.game_steps: list[int] = []

    def initialize(self) -> None:
        # 1) logging dir
        self.setup_logging(base_dir="logs/extensions/my-ext", task_name="my_extension")
        # 2) game with grid size from --grid_size handled by BaseGameManager
        self.setup_game()
        # 3) optionally set an agent on the logic
        if self.agent and hasattr(self.game, "set_agent"):
            self.game.set_agent(self.agent)

    def run(self) -> None:
        # Simple loop – reuse the core loop helpers and reset utilities
        for _ in range(self.args.max_games):
            self.game.reset()

            while not self.game.game_over:
                # Plan or pull a move (extension-specific)
                move = self.agent.get_move(self.game) if self.agent else "NO_PATH_FOUND"
                # Persist pre-move snapshot if needed
                if hasattr(self.game.game_state, "round_manager"):
                    self.game.game_state.round_manager.record_round_game_state(
                        self.game.get_state_snapshot()
                    )
                # Execute
                self.game.make_move(move)

            # Post-game bookkeeping
            self.game_count += 1
            self.total_score += self.game.score
            self.total_steps += self.game.steps
            self.game_scores.append(self.game.score)
            self.game_steps.append(self.game.steps)

            # Save canonical game_N.json (planning rounds already handled by core)
            if hasattr(self.game, "game_state") and hasattr(self.game.game_state, "save_game_summary"):
                self.game.game_state.save_game_summary(
                    self.get_game_json_path(self.game_count),
                    metadata={}
                )
            else:
                # Or use dict-based writer
                minimal: Dict[str, Any] = {
                    "score": self.game.score,
                    "steps": self.game.steps,
                    "game_over": True,
                    "metadata": {"game_number": self.game_count},
                }
                self.save_game_json_dict(minimal, game_number=self.game_count)

            # Reset for next game
            self.reset_for_next_game()

        # Save a simple summary
        summary = {
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "game_steps": self.game_steps,
        }
        self.save_simple_session_summary(summary)
```

Key core helpers used:
- `setup_game()` automatically respects `--grid_size`.
- `round_manager.record_round_game_state(state)` to store snapshots without private access.
- `get_game_json_path(n)`, `save_game_json_dict`, `save_simple_session_summary` for canonical I/O.
- `reset_for_next_game()` to standardize cleanup.

Best practices:
- Keep per-extension logic in its `game_logic` and `agent` modules.
- Avoid direct file I/O or touching private attributes; always prefer core helpers.
- Use `on_before_move`, `on_after_move`, `on_game_end` hooks if you need extra side-effects.