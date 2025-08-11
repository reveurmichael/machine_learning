from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict

from core.game_manager import BaseGameManager
from .game_logic import SupervisedGameLogic
from .agents import create as create_agent, DEFAULT_ALGORITHM
from utils.print_utils import print_success

class SupervisedV03GameManager(BaseGameManager):
    GAME_LOGIC_CLS = SupervisedGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any | None = None) -> None:
        super().__init__(args)
        self.agent = agent
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.session_start = datetime.now()
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []

    def initialize(self) -> None:
        self.setup_logging(base_dir="logs/extensions/supervised-v0.03", task_name="supervised_v0_03")
        self.setup_game()
        if self.agent is None:
            self.agent = create_agent(self.algorithm_name)

    def run(self) -> None:
        print_success("✅ Starting Supervised v0.03 session…")
        all_metrics: List[Dict[str, float]] = []
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

            # Save per-game
            self.save_current_game_json(metadata={"algorithm": self.algorithm_name})
            if hasattr(self.game, "move_features"):
                self.write_json_in_logdir(
                    f"game_{self.game_count}_features.json",
                    self.game.move_features,  # type: ignore[arg-type]
                )

            # Collect metrics per game
            per_game_metrics = getattr(self.game, "compute_metrics", lambda: {"apples_per_step": 0.0})()
            all_metrics.append(per_game_metrics)

            self.reset_for_next_game()

        # Save summary and metrics sidecar
        summary = {
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "game_steps": self.game_steps,
            "algorithm": self.algorithm_name,
        }
        self.save_simple_session_summary(summary)

        self.write_json_in_logdir(
            "metrics.json",
            {
                "per_game": all_metrics,
                "avg_apples_per_step": sum(m.get("apples_per_step", 0.0) for m in all_metrics) / max(1, len(all_metrics)),
                "avg_steps_per_game": self.total_steps / max(1, self.game_count),
            },
        )

        print_success("✅ Supervised v0.03 session complete!")