"""
Heuristic Game Manager - Simple session management for BFS
=========================================================

Minimal extension of BaseGameManager for BFS algorithm.
"""

from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import Optional, List
import json

from colorama import Fore
from core.game_manager import BaseGameManager
from game_logic import HeuristicGameLogic
from agent_bfs import BFSAgent


class HeuristicGameManager(BaseGameManager):
    """Simple session manager for BFS algorithm."""

    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.agent: Optional[BFSAgent] = None
        self.log_dir: Optional[str] = None
        self.game_steps: List[int] = []  # Track steps per game for efficiency metrics
        self.game_rounds: List[int] = []  # Track rounds per game for round analysis

    def initialize(self) -> None:
        """Initialize BFS game manager."""
        self._setup_logging()
        self.agent = BFSAgent()
        self.setup_game()

        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)

        print(Fore.GREEN + "🤖 BFS Agent initialized")
        print(Fore.CYAN + f"📂 Logs: {self.log_dir}")

    def _setup_logging(self) -> None:
        """Setup log directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/heuristics-bfs_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self) -> None:
        """Run BFS game session."""
        try:
            print(Fore.GREEN + "🚀 Starting BFS session...")
            print(Fore.CYAN + f"📊 Target games: {self.args.max_games}")

            while self.game_count < self.args.max_games and self.running:
                self._run_single_game()

            self._save_session_summary()
            print(Fore.GREEN + f"✅ Session completed! Total score: {self.total_score}")

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n⚠️  Interrupted")
            self._save_session_summary()
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
            raise

    def _run_single_game(self) -> None:
        """Run a single game."""
        self.game_count += 1
        print(Fore.BLUE + f"\n🎮 Game {self.game_count}")

        self.setup_game()
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)

        self.game_active = True
        self.consecutive_no_path_found = 0

        while self.game_active and self.game.game_state.steps < self.args.max_steps:
            self.start_new_round("BFS pathfinding")
            planned_move = self.game.get_next_planned_move()

            if planned_move == "NO_PATH_FOUND":
                self.consecutive_no_path_found += 1
                if self.consecutive_no_path_found >= 5:
                    print(Fore.RED + "❌ Too many pathfinding failures")
                    break
            else:
                self.consecutive_no_path_found = 0
                game_continues, apple_eaten = self.game.make_move(planned_move)
                if not game_continues:
                    self.game_active = False

        self._finalize_game()

    def _finalize_game(self) -> None:
        """Save game results."""
        self.total_score += self.game.game_state.score
        self.game_scores.append(self.game.game_state.score)
        self.game_steps.append(self.game.game_state.steps)  # Track steps for efficiency metrics
        self.game_rounds.append(self.round_count)  # Track rounds for round analysis

        # Simple game file save
        try:
            game_data = {
                "algorithm": "BFS",
                "score": self.game.game_state.score,
                "steps": self.game.game_state.steps,
                "round_count": self.round_count,
                "snake_length": len(self.game.snake_positions),
                "game_end_reason": getattr(self.game.game_state, 'game_end_reason', 'MAX_STEPS'),
                "detailed_history": {
                    "apple_positions": getattr(self.game.game_state, 'apple_positions', []),
                    "moves": getattr(self.game.game_state, 'moves', []),
                    "rounds_data": getattr(self.game.game_state.round_manager, 'rounds_data', {}) if hasattr(self.game.game_state, 'round_manager') else {}
                },
                "metadata": {
                    "timestamp": getattr(self.game.game_state, 'timestamp', ''),
                    "game_number": self.game_count,
                    "round_count": self.round_count
                }
            }

            game_filepath = os.path.join(self.log_dir, f"game_{self.game_count}.json")
            with open(game_filepath, 'w') as f:
                json.dump(game_data, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Save error: {e}")

        print(Fore.CYAN + f"📊 Score: {self.game.game_state.score}, Steps: {self.game.game_state.steps}")

    def _save_session_summary(self) -> None:
        """Save simple session summary."""
        summary = {
            "algorithm": "BFS",
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_rounds": sum(self.game_rounds),
            "scores": self.game_scores,
            "round_counts": self.game_rounds,
            "average_score": self.total_score / max(self.game_count, 1),
            "total_steps": sum(self.game_steps),
            "score_per_step": self.total_score / max(sum(self.game_steps), 1),
            "score_per_round": self.total_score / max(sum(self.game_rounds), 1)
        }

        print(Fore.MAGENTA + f"🧠 Algorithm: {summary['algorithm']}")
        print(Fore.CYAN + f"🎮 Total games: {summary['total_games']}")
        print(Fore.CYAN + f"🔄 Total rounds: {summary['total_rounds']}")
        print(Fore.CYAN + f"🏆 Total score: {summary['total_score']}")
        print(Fore.YELLOW + f"📈 Scores: {summary['scores']}")
        print(Fore.YELLOW + f"🔢 Round counts: {summary['round_counts']}")
        print(Fore.MAGENTA + f"📊 Average score: {summary['average_score']:.1f}")
        print(Fore.GREEN + f"⚡ Score per step: {summary['score_per_step']:.3f}")
        print(Fore.GREEN + f"🎯 Score per round: {summary['score_per_round']:.3f}")
        with open(os.path.join(self.log_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str) 