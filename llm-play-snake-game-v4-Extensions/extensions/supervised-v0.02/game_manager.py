from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, List

from core.game_manager import BaseGameManager
from .game_logic import SupervisedV02GameLogic
from .agents import create as create_agent, DEFAULT_ALGORITHM
from utils.print_utils import print_success

class SupervisedV02GameManager(BaseGameManager):
    GAME_LOGIC_CLS = SupervisedV02GameLogic

    def __init__(self, args: argparse.Namespace, agent: Any | None = None) -> None:
        super().__init__(args)
        self.agent = agent
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.session_start = datetime.now()
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []

    def initialize(self) -> None:
        self.setup_logging(base_dir="logs/extensions/supervised-v0.02", task_name="supervised_v0_02")
        self.setup_game()
        if self.agent is None:
            self.agent = create_agent(self.algorithm_name)

    def run(self) -> None:
        print_success("✅ Starting Supervised v0.02 session…")
        for _ in range(1, self.args.max_games + 1):
            self.game.reset()
            while not self.game.game_over:
                if hasattr(self.game.game_state, "round_manager"):
                    self.game.game_state.round_manager.record_round_game_state(
                        self.game.get_state_snapshot()
                    )
                move = self.agent.get_move(self.game) or "NO_PATH_FOUND"
                self.game.make_move(move)

            # Accumulate
            self.game_count += 1
            self.total_score += self.game.score
            self.total_steps += self.game.steps
            self.game_scores.append(self.game.score)
            self.game_steps.append(self.game.steps)

            # Save primary game JSON
            self.save_current_game_json(metadata={"algorithm": self.algorithm_name})

            # Save sidecar features JSON for training pipelines
            if hasattr(self.game, "move_features"):
                features_path = Path(self.get_game_json_path(self.game_count)).with_name(
                    f"game_{self.game_count}_features.json"
                )
                import json
                features_path.write_text(json.dumps(self.game.move_features, indent=2), encoding="utf-8")

            self.reset_for_next_game()

        summary = {
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "game_steps": self.game_steps,
            "algorithm": self.algorithm_name,
        }
        self.save_simple_session_summary(summary)
        print_success("✅ Supervised v0.02 session complete!")