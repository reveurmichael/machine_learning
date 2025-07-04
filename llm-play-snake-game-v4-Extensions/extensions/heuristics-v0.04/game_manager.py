from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

v0.04 Enhancement: Supports incremental JSONL/CSV dataset updates after each game
to provide real-time dataset growth visibility.

Design Patterns:
- Template Method: Inherits base session management structure
- Factory Pattern: Uses HeuristicGameLogic for game logic
- Strategy Pattern: Pluggable heuristic algorithms
- Observer Pattern: Game state changes trigger dataset updates
"""

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
import time
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
import json
import os
from colorama import Fore
import numpy as np

# Import from project root using absolute imports
from utils.print_utils import print_info, print_warning, print_error, print_success
from core.game_manager import BaseGameManager
from core.game_agents import BaseAgent
from extensions.common import EXTENSIONS_LOGS_DIR
from config.game_constants import END_REASON_MAP

# Import heuristic-specific components using relative imports
from game_logic import HeuristicGameLogic
from agents import create_agent, get_available_algorithms, DEFAULT_ALGORITHM

# Import dataset generation utilities for automatic updates
from extensions.common.utils.dataset_generator_core import DatasetGenerator
from extensions.common.utils.dataset_utils import save_csv_dataset
from extensions.common.utils.csv_schema import create_csv_row
from extensions.common.config.dataset_formats import CSV_BASIC_COLUMNS
from extensions.common.utils.dataset_format_utils import extract_dataset_records
from extensions.common.utils.jsonl_utils import append_jsonl_records
import pandas as pd

# Type alias for any heuristic agent (from agents package)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agents import BFSAgent, BFSSafeGreedyAgent, BFSHamiltonianAgent
    from agents import DFSAgent, AStarAgent, AStarHamiltonianAgent, HamiltonianAgent

HeuristicAgent = Union[
    'BFSAgent', 'BFSSafeGreedyAgent', 'BFSHamiltonianAgent',
    'DFSAgent', 'AStarAgent', 'AStarHamiltonianAgent', 'HamiltonianAgent'
]

def _to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj

class HeuristicGameManager(BaseGameManager):
    """
    Multi-algorithm session manager for heuristics v0.04.
    
    Evolution from v0.01:
    - Factory pattern for algorithm selection (was: hardcoded BFS)
    - Support for 7 different heuristic algorithms
    - Improved error handling and verbose mode
    - Simplified logging without Task-0 replay compatibility
    
    v0.04 Enhancement:
    - Automatic JSONL/CSV/summary.json updates after each game
    - Real-time dataset growth visibility
    - No optional parameters - updates always happen
    
    Design Patterns:
    - Template Method: Inherits base session management structure
    - Factory Pattern: Uses HeuristicGameLogic for game logic
    - Strategy Pattern: Pluggable heuristic algorithms (v0.02 enhancement)
    - Abstract Factory: Algorithm creation based on configuration
    - Observer Pattern: Game state changes trigger dataset updates
    """

    # Use heuristic-specific game logic
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize heuristic game manager with automatic dataset update capabilities."""
        super().__init__(args)

        # Heuristic-specific attributes
        self.algorithm_name: str = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.agent: Optional[Any] = None
        self.verbose: bool = getattr(args, "verbose", False)

        # Session statistics for summary
        self.total_score: int = 0
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []
        self.game_rounds: List[int] = []
        self.session_start_time: datetime = datetime.now()
        
        # Dataset update tracking (always enabled)
        self.dataset_generator: Optional[DatasetGenerator] = None
        
        print_info(f"[HeuristicGameManager] Initialized for {self.algorithm_name}")

    def initialize(self) -> None:
        """Initialize the game manager with automatic dataset update capabilities."""
        # Setup logging directory
        self._setup_logging()

        # Setup agent
        self._setup_agent()

        # Initialize dataset generator for automatic updates
        self._setup_dataset_generator()
        
        # Setup base game components
        self.setup_game()

        # Configure game with agent
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)
            # Ensure grid_size is set correctly
            if hasattr(self.game.game_state, 'grid_size'):
                self.game.game_state.grid_size = self.args.grid_size

        print_info(f"[HeuristicGameManager] Initialization complete for {self.algorithm_name}")

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

    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for automatic updates."""
        self.dataset_generator = DatasetGenerator(self.algorithm_name, Path(self.log_dir))
        print_info(f"[HeuristicGameManager] Dataset generator initialized for automatic updates")

    def run(self) -> None:
        """Run the heuristic game session with automatic dataset updates."""
        print_success("✅ 🚀 Starting heuristics v0.04 session...")
        print_info(f"📊 Target games: {self.args.max_games}")
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info("")
        
        # Run games
        for game_id in range(1, self.args.max_games + 1):
            print_info(f"🎮 Game {game_id}")
            # Run single game
            game_duration = self._run_single_game()
            # Finalize game and update datasets
            self._finalize_game(game_duration)
            # Display results
            self._display_game_results(game_duration)
            # Update session statistics
            self._update_session_stats(game_duration)
            # Check if we should continue
            if game_id < self.args.max_games:
                print_info("")  # Spacer between games
        
        # Save session summary
        self._save_session_summary()
        print_success("✅ ✅ Heuristics v0.04 session completed!")
        print_info(f"🎮 Games played: {len(self.game_scores)}")
        print_info(f"🏆 Total score: {self.total_score}")
        print_info(f"📈 Average score: {self.total_score / len(self.game_scores) if self.game_scores else 0:.1f}")
        print_success("✅ Heuristics v0.04 execution completed successfully!")
        if hasattr(self, 'log_dir') and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")

    def _run_single_game(self) -> float:
        """Run a single game and return its duration."""
        start_time = time.time()
        
        # Initialize game
        self.game.reset()

        # Record initial game state for Round 1
        if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
            self.game.game_state.round_manager.record_game_state(self.game.get_state_snapshot())

        # Game loop
        while not self.game.game_over:
            # Start new round for each move (heuristics plan one move at a time)
            self.start_new_round(f"{self.algorithm_name} pathfinding")

            # Record game state in round manager BEFORE getting the move
            # This ensures the agent uses the same game state for explanations as the dataset generator
            recorded_game_state = None
            if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
                recorded_game_state = self.game.get_state_snapshot()
                self.game.game_state.round_manager.record_game_state(recorded_game_state)

            # Get move from agent using the recorded game state for SSOT compliance
            if recorded_game_state and hasattr(self.game, 'get_next_planned_move_with_state'):
                move = self.game.get_next_planned_move_with_state(recorded_game_state)
            else:
                move = self.game.get_next_planned_move()
                
            if move == "NO_PATH_FOUND":
                # Record game state for the final round before ending
                if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
                    self.game.game_state.round_manager.record_game_state(self.game.get_state_snapshot())
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break
            
            # Apply move
            self.game.make_move(move)

            # Sync round data after move execution to ensure all data is persisted
            if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
                self.game.game_state.round_manager.sync_round_data()
            
            # Update display if GUI is enabled
            if hasattr(self.game, 'update_display'):
                self.game.update_display()

        # Calculate duration
        game_duration = time.time() - start_time
        
        return game_duration

    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and update datasets automatically."""
        # Increment game count before saving (matches Task-0 behavior)
        self.game_count += 1
        
        # Generate game data with explanations and metrics
        game_data = self._generate_game_data(game_duration)
        
        # Save game data
        self._save_game_data(game_data)
        
        # Update datasets automatically
        self._update_datasets_incrementally([game_data])

    def _determine_game_end_reason(self) -> str:
        """Determine why the game ended and return a canonical key from END_REASON_MAP."""
        if hasattr(self.game.game_state, 'game_end_reason'):
            raw_reason = self.game.game_state.game_end_reason
        else:
            # Fallback logic
            if self.game.game_state.steps >= self.game.game_state.max_steps:
                raw_reason = "MAX_STEPS_REACHED"
            elif self.game.game_state.score >= self.game.game_state.max_score:
                raw_reason = "MAX_STEPS_REACHED"
            else:
                raw_reason = "SELF"
        
        if raw_reason not in END_REASON_MAP:
            print_warning(f"[GameManager] Unknown end reason '{raw_reason}', defaulting to 'SELF'.")
            return "SELF"
        return raw_reason

    def _update_session_stats(self, game_duration: float) -> None:
        """Update session statistics."""
        self.total_score += self.game.game_state.score
        self.game_scores.append(self.game.game_state.score)
        self.game_steps.append(self.game.game_state.steps)
        self.game_rounds.append(self.round_count)

    def _generate_game_data(self, game_duration: float) -> Dict[str, Any]:
        """Generate game data for logging and dataset generation."""
        # Use the game data's generate_game_summary method for Task-0 compatible game files
        # This ensures clean data without game_state in rounds_data and without move_metrics
        game_summary = self.game.game_state.generate_game_summary()
        
        # Add algorithm name and duration for heuristics
        game_summary["algorithm"] = self.algorithm_name
        game_summary["duration_seconds"] = round(game_duration, 2)
        
        # Add explanations and metrics for dataset generation (v0.04 enhancement)
        game_summary["move_explanations"] = getattr(self.game.game_state, 'move_explanations', [])
        game_summary["move_metrics"] = getattr(self.game.game_state, 'move_metrics', [])
        
        return game_summary
    
    def _save_game_data(self, game_data: Dict[str, Any]) -> None:
        """Save individual game data."""
        # Use game_count to match Task-0 numbering (games start at 1, not 0)
        # game_count is incremented before this method is called, so it's already correct
        game_file = os.path.join(self.log_dir, f"game_{self.game_count}.json")
        print_info(f"[DEBUG] game_count: {self.game_count}, filename: game_{self.game_count}.json")
        with open(game_file, 'w') as f:
            json.dump(_to_serializable(game_data), f, indent=2)

    def _display_game_results(self, game_duration: float) -> None:
        """Display game results."""
        print_info(f"📊 Score: {self.game.game_state.score}, Steps: {self.game.game_state.steps}")

    def _save_session_summary(self) -> None:
        """Save session summary."""
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        summary = {
            "session_timestamp": self.session_start_time.strftime("%Y%m%d_%H%M%S"),
            "algorithm": self.algorithm_name,
            "total_games": len(self.game_scores),
            "total_score": self.total_score,
            "average_score": self.total_score / len(self.game_scores) if self.game_scores else 0,
            "total_steps": sum(self.game_steps),
            "total_rounds": sum(self.game_rounds),
            "session_duration_seconds": round(session_duration, 2),
            "score_per_step": self.total_score / sum(self.game_steps) if self.game_steps else 0,
            "score_per_round": self.total_score / sum(self.game_rounds) if self.game_rounds else 0,
            "game_scores": self.game_scores,
            "game_steps": self.game_steps,
            "round_counts": self.game_rounds,
            "configuration": {
                "grid_size": getattr(self.args, "grid_size", 10),
                "max_games": getattr(self.args, "max_games", 1),
                "verbose": getattr(self.args, "verbose", False),
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Display summary
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info(f"🎮 Total games: {len(self.game_scores)}")
        print_info(f"🔄 Total rounds: {sum(self.game_rounds)}")
        print_info(f"🏆 Total score: {self.total_score}")
        print_info(f"📈 Scores: {self.game_scores}")
        print_info(f"🔢 Round counts: {self.game_rounds}")
        print_info(f"📊 Average score: {summary['average_score']:.1f}")
        print_info(f"⚡ Score per step: {summary['score_per_step']:.3f}")
        print_info(f"🎯 Score per round: {summary['score_per_round']:.3f}")

    def start_new_round(self, round_type: str | None = None) -> None:
        """Start a new round with automatic dataset updates."""
        super().start_new_round(round_type)

    def flush_buffer_with_game_state(self):
        """Flush buffer with current game state for dataset generation."""
        super().flush_buffer_with_game_state()

    def finish_round(self, reason: str = "") -> None:
        """Finish round with automatic dataset updates."""
        super().finish_round(reason)

    def increment_round(self, reason: str = "") -> None:
        """Increment round with automatic dataset updates."""
        super().increment_round(reason)

    def _update_datasets_incrementally(self, games_data: List[Dict[str, Any]]) -> None:
        """Update datasets incrementally after each game."""
        if not self.dataset_generator:
            return
            
        print_info(f"[HeuristicGameManager] Starting dataset update with {len(games_data)} games")
        
        # Extract dataset records from all games
        all_jsonl_records = []
        all_csv_records = []
        
        for game_data in games_data:
            jsonl_records, csv_records = extract_dataset_records(game_data, self.algorithm_name)
            all_jsonl_records.extend(jsonl_records)
            all_csv_records.extend(csv_records)
        
        print_info(f"[HeuristicGameManager] After extraction - JSONL records: {len(all_jsonl_records)}, CSV records: {len(all_csv_records)}")
        
        # Save updated datasets
        self._save_updated_datasets(all_jsonl_records, all_csv_records)

    def _save_updated_datasets(self, jsonl_records: List[Dict[str, Any]], csv_records: List[Dict[str, Any]]) -> None:
        """Save updated datasets."""
        if not self.dataset_generator:
            return
        
        print_info(f"[HeuristicGameManager] Saving datasets - JSONL: {len(jsonl_records)} records, CSV: {len(csv_records)} records")
        
        # Save JSONL dataset
        if jsonl_records:
            jsonl_path = os.path.join(self.log_dir, f"{self.algorithm_name.lower()}_dataset.jsonl")
            append_jsonl_records(jsonl_path, jsonl_records, overwrite=True)
            print_success(f"✅ [HeuristicGameManager] Updated JSONL dataset: {len(jsonl_records)} records -> {jsonl_path}")
        
        # Save CSV dataset
        if csv_records:
            csv_path = os.path.join(self.log_dir, f"{self.algorithm_name.lower()}_dataset.csv")
            df = pd.DataFrame(csv_records)
            df.to_csv(csv_path, index=False)
            print_success(f"✅ [HeuristicGameManager] Updated CSV dataset: {len(csv_records)} records -> {csv_path}")

    def setup_game(self) -> None:
        """Create game logic and optional GUI interface with correct grid size."""
        # Get grid size from command line arguments
        grid_size = getattr(self.args, "grid_size", 10)
        
        # Use the specified game logic class with correct grid size
        self.game = self.GAME_LOGIC_CLS(grid_size=grid_size, use_gui=self.use_gui)

        # Attach GUI if visual mode is requested
        if self.use_gui:
            # Lazy import keeps headless extensions free of pygame.
            from gui.game_gui import GameGUI  # noqa: WPS433 – intentional local import
            gui = GameGUI()
            # Ensure GUI pixel scaling matches the *actual* game grid size
            if hasattr(self.game, "grid_size"):
                gui.resize(self.game.grid_size)  # auto-adjust cell size & grid lines
            self.game.set_gui(gui)
