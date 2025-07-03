from __future__ import annotations
"""
Heuristic Game Manager 
--------------------

Session management for multi-algorithm heuristic agents.

Evolution from v0.01: This module demonstrates how to extend the simple
proof-of-concept to support multiple algorithms using factory patterns.
Shows natural software progression while maintaining the same base architecture.

Design Philosophy:
- Extends BaseGameManager (inherits all generic session management)
- Uses HeuristicGameLogic for game mechanics
- Factory pattern for algorithm selection (v0.02 enhancement)
- No LLM dependencies (no token stats, no continuation mode)
- Simplified logging (no Task-0 replay compatibility as requested)

Evolution from v0.03: Adds language-rich move explanations and JSONL dataset generation while retaining multi-algorithm flexibility.
"""

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
import time
from datetime import datetime
from typing import Optional, Union, List
import json
import os

# Import from project root using absolute imports
from utils.print_utils import print_info, print_warning, print_error, print_success
from core.game_manager import BaseGameManager
from core.game_agents import BaseAgent
from extensions.common import EXTENSIONS_LOGS_DIR, HEURISTICS_LOG_PREFIX

# Import heuristic-specific components using relative imports
from game_logic import HeuristicGameLogic
from agents import create_agent, get_available_algorithms, DEFAULT_ALGORITHM

# Type alias for any heuristic agent (from agents package)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agents import BFSAgent, BFSSafeGreedyAgent, BFSHamiltonianAgent
    from agents import DFSAgent, AStarAgent, AStarHamiltonianAgent, HamiltonianAgent

HeuristicAgent = Union[
    'BFSAgent', 'BFSSafeGreedyAgent', 'BFSHamiltonianAgent',
    'DFSAgent', 'AStarAgent', 'AStarHamiltonianAgent', 'HamiltonianAgent'
]

class HeuristicGameManager(BaseGameManager):
    """
    Multi-algorithm session manager for heuristics v0.04.
    
    Evolution from v0.01:
    - Factory pattern for algorithm selection (was: hardcoded BFS)
    - Support for 7 different heuristic algorithms
    - Improved error handling and verbose mode
    - Simplified logging without Task-0 replay compatibility
    
    Design Patterns:
    - Template Method: Inherits base session management structure
    - Factory Pattern: Uses HeuristicGameLogic for game logic
    - Strategy Pattern: Pluggable heuristic algorithms (v0.02 enhancement)
    - Abstract Factory: Algorithm creation based on configuration
    """

    # Use heuristic-specific game logic
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize multi-algorithm heuristic game manager.
        
        Args:
            args: Command line arguments with algorithm selection
        """
        super().__init__(args)

        # Algorithm configuration
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.verbose = getattr(args, "verbose", False)

        # Agent and logging
        self.agent: Optional[BaseAgent] = None
        self.log_dir: Optional[str] = None
        self.session_start_time = datetime.now()
        self.game_steps: List[int] = []  # Track steps per game for efficiency metrics
        self.game_rounds: List[int] = []  # Track rounds per game for round analysis

    def initialize(self) -> None:
        """Initialize the multi-algorithm heuristic game manager."""
        # Setup logging directory
        self._setup_logging()

        # Create and configure heuristic agent using factory pattern
        self._setup_agent()

        # Initialize base components
        self.setup_game()

        # Configure game with agent
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)

        print_success(f"🤖 Heuristics v0.04 initialized with {self.algorithm_name} algorithm")
        if self.verbose and self.agent:
            print_info(f"🔍 Agent: {self.agent}")
        print_info(f"📂 Logs: {self.log_dir}")

    def _setup_logging(self) -> None:
        """Setup logging directory for **extension mode**.

        CRITICAL: All heuristic extensions write their outputs under:

            ROOT/logs/extensions/datasets/grid-size-N/<extension>_v<version>_<timestamp>/

        This follows the standardized dataset folder structure defined in
        docs/extensions-guideline/datasets-folder.md for consistency across
        all extensions and grid sizes.

        Directory pattern:
            logs/extensions/datasets/grid-size-{N}/heuristics-v0.04_v0.04_{timestamp}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_size = getattr(self.args, "grid_size", 10)
        
        # Follow standardized dataset folder structure
        # Reference: docs/extensions-guideline/datasets-folder.md
        dataset_folder = f"heuristics_v0.04_{timestamp}"
        base_dir = os.path.join(EXTENSIONS_LOGS_DIR, "datasets", f"grid-size-{grid_size}", dataset_folder)

        # Algorithm-specific subdirectory (all files for one run live here)
        self.log_dir = os.path.join(base_dir, self.algorithm_name.lower())

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_agent(self) -> None:
        """
        Factory method to create appropriate agent based on algorithm selection.
        
        Evolution from v0.01: Was hardcoded to BFSAgent(), now supports 7 algorithms
        using the agents package factory pattern.
        """
        # Use agents package factory method
        self.agent = create_agent(self.algorithm_name)
        
        if not self.agent:
            available_algorithms = get_available_algorithms()
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}. Available: {available_algorithms}")

        if self.verbose:
            print_info(f"🏭 Created {self.agent.__class__.__name__} for {self.algorithm_name}")

    def run(self) -> None:
        """
        Run the multi-algorithm heuristic game session.
        
        Evolution from v0.01: Same structure, enhanced with verbose output
        and better algorithm-specific messaging.
        """
        try:
            print_success("🚀 Starting heuristics v0.04 session...")
            print_info(f"📊 Target games: {self.args.max_games}")
            print_info(f"🧠 Algorithm: {self.algorithm_name}")

            if self.verbose and self.agent:
                print_info(f"🔍 Agent details: {getattr(self.agent, 'description', 'No description available')}")

            # Main game loop
            while self.game_count < self.args.max_games and self.running:
                self._run_single_game()

            # Save session summary
            self._save_session_summary()

            print_success("✅ Heuristics v0.04 session completed!")
            print_info(f"📊 Games played: {self.game_count}")
            print_info(f"🏆 Total score: {self.total_score}")
            if self.game_count > 0:
                avg_score = self.total_score / self.game_count
                print_info(f"📈 Average score: {avg_score:.1f}")

        except KeyboardInterrupt:
            print_warning("\n⚠️  Session interrupted by user")
            self._save_session_summary()
        except Exception as e:
            print_error(f"❌ Session error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise

    def _run_single_game(self) -> None:
        """
        Run a single game with selected heuristic algorithm.
        
        Evolution from v0.01: Same structure, enhanced with verbose output
        and algorithm-specific error handling.
        """
        self.game_count += 1
        self.round_count = 1

        if self.verbose:
            print_info(f"\n🎮 Starting Game {self.game_count} with {self.algorithm_name}")
        else:
            print_info(f"\n🎮 Game {self.game_count}")

        # Reset game state
        self.setup_game()
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)

        # Reset game manager state for new game
        self.game_active = True
        self.need_new_plan = True
        self.consecutive_no_path_found = 0

        # Manually save the state for the very first round (round 1)
        if self.game.game_state.round_manager.round_count == 1:
            self.game.game_state.round_manager.rounds_data[1]['game_state'] = self.game.get_state_snapshot()

        game_start_time = time.time()

        # Game loop
        while self.game_active and self.game.game_state.steps < self.args.max_steps:
            # Start new round for planning
            self.start_new_round(f"{self.algorithm_name} pathfinding")

            # Get next move from heuristic agent
            planned_move = self.game.get_next_planned_move()

            # Execute move
            if planned_move == "NO_PATH_FOUND":
                self.consecutive_no_path_found += 1
                if self.verbose:
                    print_warning(f"⚠️  No path found (attempt {self.consecutive_no_path_found})")

                if self.consecutive_no_path_found >= 5:
                    print_error(f"❌ Too many consecutive pathfinding failures with {self.algorithm_name}")
                    self.game_active = False
                    break
            else:
                self.consecutive_no_path_found = 0

                # Make the move
                game_continues, apple_eaten = self.game.make_move(planned_move)

                # Show apple eaten in verbose mode
                if apple_eaten and self.verbose:
                    current_score = self.game.game_state.score
                    print_success(f"🍎 Apple eaten! Score: {current_score}")
                elif apple_eaten:
                    current_score = self.game.game_state.score
                    print_success(f"🍎 Score: {current_score}")

                if not game_continues:
                    self.game_active = False

        # Record game completion
        game_duration = time.time() - game_start_time
        self._finalize_game(game_duration)

    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and save simplified results (no Task-0 compatibility)."""
        # Update session stats
        self.total_score += self.game.game_state.score

        self.game_scores.append(self.game.game_state.score)
        self.game_steps.append(self.game.game_state.steps)  # Track steps for efficiency metrics
        self.game_rounds.append(self.round_count)  # Track rounds for round analysis

        # Use HeuristicGameData.generate_game_summary for v0.04 explanation support
        if hasattr(self.game.game_state, 'generate_game_summary'):
            # Use proper game data generation (includes v0.04 explanations)
            game_data = self.game.game_state.generate_game_summary(
                primary_provider=self.algorithm_name.lower(),
                primary_model=self.algorithm_name,
                metadata={
                    "game_number": self.game_count,
                    "round_count": self.round_count,
                    "duration_seconds": round(game_duration, 2)
                }
            )
        else:
            # Fallback to manual construction for older versions
            game_data = {
                "algorithm": self.algorithm_name,
                "score": self.game.game_state.score,
                "steps": self.game.game_state.steps,
                "round_count": self.round_count,
                "snake_length": len(self.game.snake_positions),
                "duration_seconds": round(game_duration, 2),
                "game_end_reason": getattr(self.game.game_state, 'game_end_reason', 'UNKNOWN'),
                "detailed_history": {
                    "apple_positions": getattr(self.game.game_state, 'apple_positions', []),
                    "moves": getattr(self.game.game_state, 'moves', []),
                    "rounds_data": getattr(self.game.game_state.round_manager, 'rounds_data', {}) if hasattr(self.game.game_state, 'round_manager') else {},
                },
                "metadata": {
                    "timestamp": getattr(self.game.game_state, 'timestamp', ''),
                    "game_number": self.game_count,
                    "round_count": self.round_count
                }
            }

        # self.log_dir 已经是算法目录
        game_filepath = os.path.join(self.log_dir, f"game_{self.game_count}.json")
        with open(game_filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, default=str)  # Handle numpy types

        # Show results
        if self.verbose:
            print_info(f"📊 Game {self.game_count} completed:")
            print_info(f"   Algorithm: {self.algorithm_name}")
            print_info(f"   Score: {self.game.game_state.score}")
            print_info(f"   Steps: {self.game.game_state.steps}")
            print_info(f"   Duration: {game_duration:.2f}s")
        else:
            print_info(f"📊 Score: {self.game.game_state.score}, Steps: {self.game.game_state.steps}")

    def _save_session_summary(self) -> None:
        """Save simplified session summary (no Task-0 compatibility as requested)."""
        summary_data = {
            "heuristics_version": "v0.04",
            "timestamp": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": self.algorithm_name,
            "total_games": self.game_count,
            "total_score": self.total_score,
            "total_rounds": sum(self.game_rounds),
            "scores": self.game_scores,
            "round_counts": self.game_rounds,
            "total_steps": sum(self.game_steps),
            "statistics": {
                "average_score": self.total_score / max(self.game_count, 1),
                "max_score": max(self.game_scores) if self.game_scores else 0,
                "min_score": min(self.game_scores) if self.game_scores else 0,
                "score_per_step": self.total_score / max(sum(self.game_steps), 1),
                "score_per_round": self.total_score / max(sum(self.game_rounds), 1)
            }
        }
        print_info(f"🧠 Algorithm: {summary_data['algorithm']}")
        print_info(f"🎮 Total games: {summary_data['total_games']}")
        print_info(f"🔄 Total rounds: {summary_data['total_rounds']}")
        print_info(f"🏆 Total score: {summary_data['total_score']}")
        print_info(f"📈 Scores: {summary_data['scores']}")
        print_info(f"🔢 Round counts: {summary_data['round_counts']}")
        print_info(f"📊 Average score: {summary_data['statistics']['average_score']:.1f}")
        print_info(f"⚡ Score per step: {summary_data['statistics']['score_per_step']:.3f}")
        print_info(f"🎯 Score per round: {summary_data['statistics']['score_per_round']:.3f}")

        # self.log_dir 已经是算法目录
        summary_filepath = os.path.join(self.log_dir, "summary.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)  # Handle numpy types

        if self.verbose:
            print_success(f"💾 Session summary saved to: {summary_filepath}") 

    def start_new_round(self, round_type: str | None = None) -> None:
        """
        Starts a new round, extending the base method to save the full game state.

        Design Pattern: Template Method (Enhancement)
        Purpose: Overrides the base method to add heuristic-specific data (the full game state)
                 to each round's record, ensuring data-rich logs for dataset generation.
        Educational Value: Demonstrates how to cleanly extend a base class's behavior
                         without modifying the original, a core tenet of the OCP.
        """
        super().start_new_round(round_type)
        # After the base method creates the round entry, add the full game state snapshot.
        if self.game and self.game.game_state and self.game.game_state.round_manager:
            current_round_number = self.game.game_state.round_manager.round_count
            self.game.game_state.round_manager.rounds_data[current_round_number]['game_state'] = self.game.get_state_snapshot()
    