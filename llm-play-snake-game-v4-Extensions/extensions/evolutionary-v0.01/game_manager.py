"""Evolutionary GameManager – v0.01
================================

Minimal session manager that wires the tiny GA planner into the base engine.
The structure mirrors *heuristics-v0.01* to keep learning curve flat.

Design decisions:
• Always headless (`--no-gui`) – evolutionary training typically runs in
  batches; visualisation is left for future versions.
• Reuses common logging helpers to stay compliant with the extension log
  layout rule.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional, List

from colorama import Fore

from core.game_manager import BaseGameManager

from game_logic import EGA_GameLogic
from agent_ga import GAAgent

from extensions.common.path_utils import setup_extension_paths
from extensions.common import EXTENSIONS_LOGS_DIR

setup_extension_paths()


class EvolutionaryGameManager(BaseGameManager):
    """Session manager for the GA agent."""

    GAME_LOGIC_CLS = EGA_GameLogic

    def __init__(self, args: argparse.Namespace) -> None:
        # Force headless to keep v0.01 simple & fast
        args.no_gui = True
        super().__init__(args)

        self.clock = None  # type: ignore[assignment]
        self.agent: Optional[GAAgent] = None
        self.log_dir: Optional[str] = None
        self.total_score: int = 0
        self.game_steps: List[int] = []
        self.game_rounds: List[int] = []

    # ------------------------------------------------------------------
    # BaseGameManager hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:  # noqa: D401  (imperative mood)
        """Initialise GA session – agent + logging."""
        self._setup_logging()
        self.agent = GAAgent()
        self.setup_game()

        assert isinstance(self.game, EGA_GameLogic)  # type guard for linters
        self.game.set_agent(self.agent)

        print(Fore.GREEN + "🧬 GA Agent initialised")
        print(Fore.CYAN + f"📂 Logs: {self.log_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"evolutionary-ga_{timestamp}"
        # Ensure directory exists under extensions logs
        self.log_dir = os.path.join(EXTENSIONS_LOGS_DIR, experiment_folder)
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop – keep identical structure to heuristic v0.01 for comparison
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D401
        print(Fore.GREEN + "🚀 Starting GA session…")
        try:
            while self.game_count < self.args.max_games and self.running:
                self._run_single_game()
            self._save_session_summary()
        except KeyboardInterrupt:
            print("\nInterrupted – saving partial results…")
            self._save_session_summary()

    def _run_single_game(self) -> None:
        self.game_count += 1
        print(Fore.BLUE + f"\n🎮 Game {self.game_count}")

        self.setup_game()
        assert isinstance(self.game, EGA_GameLogic)
        if self.agent:
            self.game.set_agent(self.agent)

        self.game_active = True
        while self.game_active and self.game.game_state.steps < self.args.max_steps:
            self.start_new_round("GA planning")
            move = self.game.get_next_planned_move()
            game_continues, _ = self.game.make_move(move)
            if not game_continues:
                self.game_active = False
        self._finalise_game()

    def _finalise_game(self) -> None:
        self.total_score += self.game.game_state.score
        self.game_scores.append(self.game.game_state.score)
        self.game_steps.append(self.game.game_state.steps)
        self.game_rounds.append(self.round_count)

        game_data = {
            "algorithm": "GA",
            "score": self.game.game_state.score,
            "steps": self.game.game_state.steps,
            "round_count": self.round_count,
            "detailed_history": {},  # v0.01 keeps it minimal
        }
        if self.log_dir:
            path = os.path.join(self.log_dir, f"game_{self.game_count}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(game_data, fh, indent=2)

        print(Fore.CYAN + f"🏆 Score: {self.game.game_state.score}")

    def _save_session_summary(self) -> None:
        summary = {
            "algorithm": "GA",
            "total_games": self.game_count,
            "total_score": self.total_score,
            "average_score": self.total_score / max(1, self.game_count),
        }
        if self.log_dir:
            with open(os.path.join(self.log_dir, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        print(Fore.GREEN + "✅ Summary saved") 