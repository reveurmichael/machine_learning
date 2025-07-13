# TODO: In the attached "extensions/heuristics-v0.04/game_manager.py", I have a bunch of TODOs. Please go through them and fix them. Basically, you might want to make writting game_manager.py for extensions much easier (not only for heuristics-v0.04, but also for other extensions). However, keep in mind that writing jsonl files is specific to heuristics-v0.04. Hence, state_management.py (PRE/POST move states) is specific to heuristics-v0.04. For this time, you are allowed to adjust Task0 codebase. But, don't change any functionality of Task0 and heuristics-v0.04. Attached md files can be useful for you, though some of them are outdated. You might want to update core.md file after you are finished.

```python
from __future__ import annotations
import sys
import os
from pathlib import Path

# Fix UTF-8 encoding issues on Windows
# This ensures that all subprocesses and file operations use UTF-8
# All file operations (CSV, JSONL, JSON) in v0.04 use UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
Heuristic Game Manager 
----------------

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
from typing import Optional, List, Dict, Any
import json
import os
import copy

# Import from project root using absolute imports
from utils.print_utils import print_info, print_warning, print_success, print_error
from core.game_manager import BaseGameManager
from extensions.common import EXTENSIONS_LOGS_DIR
from config.game_constants import END_REASON_MAP

# Import heuristic-specific components using relative imports
from game_logic import HeuristicGameLogic
from agents import get_available_algorithms, DEFAULT_ALGORITHM

# Import dataset generation utilities for automatic updates
from dataset_generator import DatasetGenerator

# Import BFSAgent for SSOT utilities
from extensions.common.utils.game_state_utils import to_serializable
from heuristics_utils import (
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    bfs_pathfind,
)

# Import state management for robust pre/post state separation
from state_management import StateManager, validate_explanation_head_consistency

# Type alias for any heuristic agent (from agents package)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# JSON serialization moved to BFSAgent for SSOT compliance


class HeuristicGameManager(BaseGameManager):
    """
    Minimal, SOLID, and DRY session manager for heuristics v0.04.
    Inherits all generic session, logging, and dataset logic from BaseGameManager.
    Only implements heuristics-v0.04-specific logic (JSONL, PRE/POST state management).
    """
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any) -> None:
        """Initialize heuristic game manager with automatic dataset update capabilities.
        
        Args:
            args: Command line arguments namespace
            agent: Required agent instance (SSOT enforcement)
        """
        super().__init__(args, agent)

        # Heuristic-specific attributes
        self.algorithm_name: str = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        # Shared agent instance (SSOT). Must be provided.
        self.agent: Any = agent
        self.verbose: bool = getattr(args, "verbose", False)

        # Session statistics for summary
        self.total_score: int = 0
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []
        self.game_rounds: List[int] = []
        self.session_start_time: datetime = datetime.now()

        # Dataset update tracking (always enabled)
        self.dataset_generator: Optional[DatasetGenerator] = None

        # Initialize limits manager for consistent limit tracking
        from core.game_limits_manager import create_limits_manager
        self.limits_manager = create_limits_manager(args)

        print_info(f"[HeuristicGameManager] Initialized for {self.algorithm_name}")

    def initialize(self) -> None:
        """Initialize the game manager with automatic dataset update capabilities."""
        # Use base class initialization for logging, agent, limits, etc.
        super().initialize()
        
        # Setup logging directory for heuristics
        self._setup_logging()
        
        # Setup agent validation
        self._setup_agent()
        
        # Heuristics-v0.04-specific dataset generator (JSONL, PRE/POST)
        if self.log_dir:
            self._setup_dataset_generator()

    def _setup_logging(self) -> None:
        """Setup logging directory for **extension mode**.

        CRITICAL: All heuristic extensions write their outputs under:

            ROOT/logs/extensions/datasets/grid-size-N/<extension>_v<version>_<timestamp>/

        This follows the standardized dataset folder structure defined in
        docs/extensions-guideline/datasets-folder.md for consistency across
        all extensions and grid sizes.

        Directory pattern:
            logs/extensions/datasets/grid-size-{N}/heuristics_v0.04_{timestamp}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_size = getattr(self.args, "grid_size", 10)

        # Follow standardized dataset folder structure
        # Reference: docs/extensions-guideline/datasets-folder.md
        dataset_folder = f"heuristics_v0.04_{timestamp}"
        base_dir = os.path.join(
            EXTENSIONS_LOGS_DIR, "datasets", f"grid-size-{grid_size}", dataset_folder
        )

        # Algorithm-specific subdirectory (all files for one run live here)
        self.log_dir = os.path.join(base_dir, self.algorithm_name.lower())

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_agent(self) -> None:
        """
        Factory method to create appropriate agent based on algorithm selection.
        """
        try:
            # Require agent to be provided (SSOT enforcement)
            if self.agent is None:
                raise RuntimeError(
                    f"Agent is required for HeuristicGameManager. Algorithm '{self.algorithm_name}' needs an agent instance."
                )

            # Validate that the provided agent matches requested algorithm
            provided_name = getattr(self.agent, "algorithm_name", None)
            if provided_name and provided_name.upper() != self.algorithm_name.upper():
                raise RuntimeError(
                    f"Provided agent algorithm '{provided_name}' does not match requested '{self.algorithm_name}'."
                )

            if not self.agent:
                available_algorithms = get_available_algorithms()
                raise ValueError(
                    f"Unknown algorithm: {self.algorithm_name}. Available: {available_algorithms}"
                )

            if self.verbose:
                print_info(
                    f"🏭 Using {self.agent.__class__.__name__} for {self.algorithm_name}"
                )
        except Exception:
            raise

    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for automatic updates."""
        # Pass current agent instance to allow agent-level control over prompt/completion formatting
        self.dataset_generator = DatasetGenerator(
            self.algorithm_name, Path(self.log_dir), agent=self.agent
        )

        # Open CSV and JSONL files for writing
        self.dataset_generator._open_csv()
        self.dataset_generator._open_jsonl()

        print_info(
            "[HeuristicGameManager] Dataset generator initialized for automatic updates"
        )

    def run(self) -> None:
        """Run the heuristic game session with automatic dataset updates."""
        print_success("✅ 🚀 Starting heuristics v0.04 session...")
        print_info(f"📊 Target games: {self.args.max_games}")
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info("")

        # Run games
        for game_id in range(1, self.args.max_games + 1):
            print_info(f"🎮 Game {game_id}")
            # Run single game using heuristics-specific logic
            game_duration = self._run_single_game()
            # Finalize game and update datasets
            self._finalize_game(game_duration)
            # Update session statistics
            self._update_session_stats(game_duration)
            # Check if we should continue
            if game_id < self.args.max_games:
                print_info("")  # Spacer between games

        # Save session summary
        self._save_session_summary()

        # Close dataset generator files
        if self.dataset_generator:
            if self.dataset_generator._csv_writer:
                self.dataset_generator._csv_writer[1].close()
                print_success("CSV dataset saved")
            if self.dataset_generator._jsonl_fh:
                self.dataset_generator._jsonl_fh.close()
                print_success("JSONL dataset saved")

        print_success("✅ ✅ Heuristics v0.04 session completed!")
        print_info(f"🎮 Games played: {len(self.game_scores)}")
        print_info(f"🏆 Total score: {self.total_score}")
        print_info(
            f"📈 Average score: {self.total_score / len(self.game_scores) if self.game_scores else 0:.1f}"
        )
        print_success("✅ Heuristics v0.04 execution completed successfully!")
        if hasattr(self, "log_dir") and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")

    def _run_single_game(self) -> float:
        """Run a single heuristics game with agent integration.
        
        Returns:
            Duration of the game in seconds
        """
        start_time = time.time()
        
        # Initialize game
        self.game.reset()
        
        # Game loop with agent integration
        steps = 0
        max_steps = self.args.max_steps
        
        while not getattr(self.game, 'game_over', False):
            steps += 1
            
            # Get move from game logic (which handles agent interaction)
            move = self.game.get_next_planned_move()
            
            # Handle NO_PATH_FOUND case
            if move == "NO_PATH_FOUND":
                # End game when no path is found
                if hasattr(self.game, 'game_state'):
                    self.game.game_state.record_game_end("NO_PATH_FOUND")
                break
            
            # Apply move
            self.game.make_move(move)
            
            # Check max steps using limits manager
            if not self.limits_manager.check_step_limit_with_adapter(self):
                if hasattr(self.game, 'game_state'):
                    self.game.game_state.record_game_end("MAX_STEPS_REACHED")
                break
        
        # Calculate duration
        game_duration = time.time() - start_time
        
        return game_duration

    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and update datasets automatically.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Use base class method for common finalization
        super().finalize_game(game_duration)
        
        # Update datasets automatically (heuristics-specific)
        self._update_datasets_incrementally([self.generate_game_data(game_duration)])
        
        # Display game results using the correct method
        self.display_game_results(game_duration)

    def _update_session_stats(self, game_duration: float) -> None:
        """Update session statistics.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Use base class method for common stats
        super().update_session_stats(game_duration)
        
        # Update heuristics-specific stats
        self._update_task_specific_stats(game_duration)

    def _save_session_summary(self) -> None:
        """Save session summary."""
        # Use base class method and extend with heuristics-specific data
        super().save_session_summary()
        
        # Add heuristics-specific summary data
        if hasattr(self, 'session_start_time'):
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
            
            # Extend summary with heuristics-specific information
            summary_file = os.path.join(self.log_dir, "summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                
                # Add heuristics-specific fields
                summary["algorithm"] = self.algorithm_name
                summary["session_duration_seconds"] = round(session_duration, 2)
                summary["score_per_step"] = (
                    self.total_score / sum(self.game_steps) if self.game_steps else 0
                )
                summary["score_per_round"] = (
                    self.total_score / sum(self.game_rounds) if self.game_rounds else 0
                )
                
                # Save updated summary
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                
                # Display heuristics-specific summary
                print_info(f"🧠 Algorithm: {self.algorithm_name}")
                print_info(f"📊 Average score: {summary['average_score']:.1f}")
                print_info(f"⚡ Score per step: {summary['score_per_step']:.3f}")
                print_info(f"🎯 Score per round: {summary['score_per_round']:.3f}")

    def _update_datasets_incrementally(self, games_data: List[Dict[str, Any]]) -> None:
        """Update datasets incrementally after each game."""
        if not self.dataset_generator:
            print_warning("[HeuristicGameManager] No dataset generator available")
            return

        for game_data in games_data:
            # game_count is already incremented
            game_data["game_number"] = self.game_count
            self.dataset_generator._process_single_game(game_data)

    def setup_game(self) -> None:
        """Create game logic and optional GUI interface with correct grid size."""
        # Get grid size from command line arguments
        grid_size = getattr(self.args, "grid_size", 10)

        # Use the specified game logic class with correct grid size
        self.game = self.GAME_LOGIC_CLS(grid_size=grid_size, use_gui=self.use_gui)

        # Set the agent on the game logic (required for move planning)
        if hasattr(self.game, 'set_agent'):
            self.game.set_agent(self.agent)

        # Attach GUI if visual mode is requested
        if self.use_gui:
            # Lazy import keeps headless extensions free of pygame.
            from gui.game_gui import GameGUI  # noqa: WPS433 – intentional local import

            gui = GameGUI()
            # Ensure GUI pixel scaling matches the *actual* game grid size
            if hasattr(self.game, "grid_size"):
                gui.resize(self.game.grid_size)  # auto-adjust cell size & grid lines
            self.game.set_gui(gui)

    def _add_task_specific_game_data(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Add heuristics-specific data to game data.
        
        Args:
            game_data: Game data dictionary to modify
            game_duration: Duration of the game in seconds
        """
        # Add heuristics-specific fields
        game_data.update({
            "algorithm": self.algorithm_name,
            "agent_type": type(self.agent).__name__,
            "grid_size": self.args.grid_size,
            "max_steps": self.args.max_steps,
        })

    def _prepare_game_data_for_saving(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare game data for saving with heuristics-specific formatting.
        
        Args:
            game_data: Game data dictionary to prepare
            
        Returns:
            Prepared game data dictionary
        """
        # Add heuristics-specific metadata
        game_data["heuristics_version"] = "v0.04"
        game_data["timestamp"] = datetime.now().isoformat()
        return game_data

    def _display_task_specific_results(self, game_duration: float) -> None:
        """Display heuristics-specific results.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        print_info(f"[HeuristicGameManager] Game completed in {game_duration:.2f}s")
        print_info(f"[HeuristicGameManager] Algorithm: {self.algorithm_name}")

    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add heuristics-specific summary data.
        
        Args:
            summary: Summary dictionary to modify
        """
        summary.update({
            "algorithm": self.algorithm_name,
            "heuristics_version": "v0.04",
            "agent_type": type(self.agent).__name__,
        })

    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display heuristics-specific summary information.
        
        Args:
            summary: Session summary dictionary
        """
        print_info(f"🧠 Algorithm: {summary['algorithm']}")
        print_info(f"🤖 Agent: {summary['agent_type']}")

    def _update_task_specific_stats(self, game_duration: float) -> None:
        """Update heuristics-specific statistics.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Track algorithm performance
        if not hasattr(self, 'algorithm_stats'):
            self.algorithm_stats = {
                'total_games': 0,
                'total_score': 0,
                'total_steps': 0,
                'total_duration': 0.0,
            }
        
        self.algorithm_stats['total_games'] += 1
        self.algorithm_stats['total_score'] += getattr(self.game, 'score', 0)
        self.algorithm_stats['total_steps'] += getattr(self.game, 'steps', 0)
        self.algorithm_stats['total_duration'] += game_duration
```

