from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, List

from core.game_manager import BaseGameManager
from .game_logic import SupervisedGameLogic
from .agents import create as create_agent, DEFAULT_ALGORITHM
from utils.print_utils import print_info, print_success

class SupervisedGameManager(BaseGameManager):
    GAME_LOGIC_CLS = SupervisedGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any | None = None) -> None:
        super().__init__(args)
        self.agent = agent
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.session_start = datetime.now()
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []

    def initialize(self) -> None:
        self.setup_logging(base_dir="logs/extensions/supervised-v0.01", task_name="supervised_v0_01")
        self.setup_game()
        if self.agent is None:
            self.agent = create_agent(self.algorithm_name)
        if hasattr(self.game, "set_algorithm_name"):
            self.game.set_algorithm_name(self.algorithm_name)

    def run(self) -> None:
        print_success("✅ Starting Supervised v0.01 session…")
        for game_id in range(1, self.args.max_games + 1):
            self.game.reset()
            while not self.game.game_over:
                # Persist pre-move snapshot
                if hasattr(self.game.game_state, "round_manager"):
                    self.game.game_state.round_manager.record_round_game_state(
                        self.game.get_state_snapshot()
                    )
                # Single-step decision
                move = self.agent.get_move(self.game) or "NO_PATH_FOUND"
                self.game.make_move(move)

            # Save and accumulate
            self.game_count += 1
            self.total_score += self.game.score
            self.total_steps += self.game.steps
            self.game_scores.append(self.game.score)
            self.game_steps.append(self.game.steps)
            self.save_current_game_json(metadata={"algorithm": self.algorithm_name})
            self.reset_for_next_game()
        # Session summary
        summary = {
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "game_steps": self.game_steps,
            "algorithm": self.algorithm_name,
        }
        self.save_simple_session_summary(summary)
        print_success("✅ Supervised v0.01 session complete!")