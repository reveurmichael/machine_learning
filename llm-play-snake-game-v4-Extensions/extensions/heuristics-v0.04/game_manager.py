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
from agents import create, get_available_algorithms, DEFAULT_ALGORITHM

# Import dataset generation utilities for automatic updates
from dataset_generator import DatasetGenerator

# Import BFSAgent for SSOT utilities
from agents.agent_bfs import BFSAgent

# Import state management for robust pre/post state separation
from state_management import StateManager, validate_explanation_head_consistency

# Type alias for any heuristic agent (from agents package)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agents import BFSAgent

# JSON serialization moved to BFSAgent for SSOT compliance

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
        """
        
        try:
            # Use agents package canonical factory method
            self.agent = create(self.algorithm_name)
            
            if not self.agent:
                available_algorithms = get_available_algorithms()
                raise ValueError(f"Unknown algorithm: {self.algorithm_name}. Available: {available_algorithms}")

            if self.verbose:
                print_info(f"🏭 Created {self.agent.__class__.__name__} for {self.algorithm_name}")
        except Exception:
            raise

    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for automatic updates."""
        self.dataset_generator = DatasetGenerator(self.algorithm_name, Path(self.log_dir))
        
        # Open CSV and JSONL files for writing
        self.dataset_generator._open_csv()
        self.dataset_generator._open_jsonl()
        
        print_info("[HeuristicGameManager] Dataset generator initialized for automatic updates")

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
        print_info(f"📈 Average score: {self.total_score / len(self.game_scores) if self.game_scores else 0:.1f}")
        print_success("✅ Heuristics v0.04 execution completed successfully!")
        if hasattr(self, 'log_dir') and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")

    def _run_single_game(self) -> float:
        """Run a single game and return its duration using robust state management."""
        start_time = time.time()
        
        # Initialize game
        self.game.reset()

        # Initialize state manager for robust pre/post state separation
        state_manager = StateManager()

        # Create initial pre-move state for round 1
        initial_raw_state = self.game.get_state_snapshot()
        initial_pre_state = state_manager.create_pre_move_state(initial_raw_state)
        
        # Fail-fast: Validate initial game state
        if not initial_pre_state.get_snake_positions():
            print_error(f"[FAIL-FAST] Initial game state has no snake positions: {initial_raw_state}")
            raise RuntimeError("[SSOT] Initial game state has no snake positions - game reset failed")
        
        # Store initial pre-move state in round data
        if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
            self.game.game_state.round_manager.round_buffer.number = 1
            round_data = self.game.game_state.round_manager._get_or_create_round_data(1)
            round_data['game_state'] = dict(initial_pre_state.game_state)  # Convert back to dict for storage
            self.game.game_state.round_manager.sync_round_data()
            
            # Fail-fast: Verify round 1 was recorded
            rounds_keys = list(self.game.game_state.round_manager.rounds_data.keys())
            if 1 not in rounds_keys and '1' not in rounds_keys:
                raise RuntimeError(f"[SSOT] Round 1 not recorded after setup. Available rounds: {rounds_keys}")
        else:
            raise RuntimeError("[SSOT] Round manager missing after game reset. Cannot record round 1 pre-move state.")

        # Game loop with robust state management
        steps = 0
        while not self.game.game_over:
            steps += 1
            
            # Sync previous round's data before starting a new round
            if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager and self.round_count > 0:
                self.game.game_state.round_manager.sync_round_data()

            # Start new round for each move
            self.start_new_round(f"{self.algorithm_name} pathfinding")

            # --- ROBUST PRE-MOVE STATE MANAGEMENT ---
            # Create immutable pre-move state from current game state
            raw_pre_state = self.game.get_state_snapshot()
            pre_state = state_manager.create_pre_move_state(raw_pre_state)
            
            # Store pre-move state in round data
            if hasattr(self.game.game_state, "round_manager") and self.game.game_state.round_manager:
                round_num = self.game.game_state.round_manager.round_buffer.number
                round_data = self.game.game_state.round_manager._get_or_create_round_data(round_num)
                round_data['game_state'] = dict(pre_state.game_state)  # Convert back to dict for storage

            # --- AGENT DECISION MAKING WITH IMMUTABLE STATE ---
            # Extract state dict for agent compatibility (safe because original was deep-copied)
            agent_state_dict = dict(pre_state.game_state)
            
            # Fail-fast: Ensure game logic has required method
            if not hasattr(self.game, 'get_next_planned_move_with_state'):
                raise RuntimeError("[SSOT] Game logic missing get_next_planned_move_with_state method - required for SSOT compliance")
            
            # Get move and explanation using immutable pre-move state
            move, explanation = self.game.get_next_planned_move_with_state(agent_state_dict, return_explanation=True)
            
            # --- FAIL-FAST: VALIDATE EXPLANATION HEAD CONSISTENCY ---
            if not validate_explanation_head_consistency(pre_state, explanation):
                import json as _json
                print_error("[SSOT] FAIL-FAST: Explanation head position mismatch")
                print_error(f"[SSOT] Pre-move state: {_json.dumps(dict(pre_state.game_state))}")
                print_error(f"[SSOT] Explanation: {_json.dumps(explanation)}")
                raise RuntimeError("[SSOT] FAIL-FAST: Explanation head position does not match pre-move state")

            # --- FAIL-FAST SSOT VALIDATION ---
            # Validate move against pre-move state using centralized utilities
            head = pre_state.get_head_position()
            body_positions = pre_state.get_snake_positions()
            manhattan_distance = BFSAgent.calculate_manhattan_distance(dict(pre_state.game_state))
            valid_moves = BFSAgent.calculate_valid_moves_ssot(dict(pre_state.game_state))
            
            if move == "NO_PATH_FOUND":
                if valid_moves:
                    print_error(f"[SSOT VIOLATION] Agent returned 'NO_PATH_FOUND' but valid moves exist: {valid_moves} for head {head}")
                    raise RuntimeError(f"SSOT violation: agent returned 'NO_PATH_FOUND' but valid moves exist: {valid_moves}")
                # Record final game state
                if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
                    round_num = self.game.game_state.round_manager.round_buffer.number
                    round_data = self.game.game_state.round_manager._get_or_create_round_data(round_num)
                    round_data['game_state'] = copy.deepcopy(self.game.get_state_snapshot())
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break
                
            if move not in valid_moves:
                print_error(f"[SSOT VIOLATION] Agent chose '{move}' but valid moves are {valid_moves} for head {head}")
                print_error("[SSOT VIOLATION] This indicates a bug in the agent or state management")
                print_error(f"[SSOT VIOLATION] Game state: head={head}, snake={body_positions}, grid_size={pre_state.get_grid_size()}")
                raise RuntimeError(f"SSOT violation: agent move '{move}' not in valid moves {valid_moves}")

            # --- APPLY MOVE AND CREATE POST-MOVE STATE ---
            # Apply move to game logic
            self.game.make_move(move)
            
            # Create post-move state from game state after move
            raw_post_state = self.game.get_state_snapshot()
            post_state = state_manager.create_post_move_state(pre_state, move, raw_post_state)
            
            # --- POST-MOVE VALIDATION ---
            # Check if there are any valid moves left after move
            post_valid_moves = BFSAgent.calculate_valid_moves_ssot(dict(post_state.game_state))
            if not post_valid_moves:
                print_error("[DEBUG] No valid moves left after move. Ending game as TRAPPED/NO_PATH_FOUND.")
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break
                
            # Check if apple is reachable from new post-move head position
            post_head = post_state.get_head_position()
            post_apple = post_state.get_apple_position()
            post_snake_positions = post_state.get_snake_positions()
            obstacles = set(tuple(p) for p in post_snake_positions[:-1])
            
            # Simple BFS pathfinding implementation
            path_to_apple = BFSAgent._bfs_pathfind(post_head, post_apple, obstacles, post_state.get_grid_size())
            if path_to_apple is None:
                print_error("Apple unreachable after move. Ending game as NO_PATH_FOUND.")
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break

            # Update display if GUI is enabled
            if hasattr(self.game, 'update_display'):
                self.game.update_display()
                
            # Check max steps after move execution
            if steps >= self.args.max_steps:
                print_error("[DEBUG] Max steps reached. Current game state:")
                print_error(f"[DEBUG] Head: {self.game.head_position}")
                print_error(f"[DEBUG] Snake: {self.game.snake_positions}")
                print_error(f"[DEBUG] Apple: {self.game.apple_position}")
                print_error(f"[DEBUG] Score: {self.game.game_state.score}")
                print_error(f"[DEBUG] Steps: {self.game.game_state.steps}")
                print_error(f"[DEBUG] Game over: {self.game.game_over}")
                print_error(f"[DEBUG] Game end reason: {getattr(self.game.game_state, 'game_end_reason', 'None')}")
                
                # Record the final move before ending the game
                if hasattr(self.game.game_state, 'round_manager') and self.game.game_state.round_manager:
                    round_num = self.game.game_state.round_manager.round_buffer.number
                    round_data = self.game.game_state.round_manager._get_or_create_round_data(round_num)
                    round_data['game_state'] = copy.deepcopy(self.game.get_state_snapshot())
                
                self.game.game_state.record_game_end("MAX_STEPS_REACHED")
                break

        # Calculate duration
        game_duration = time.time() - start_time

        # --- FAIL-FAST: Ensure final step is recorded ---
        final_steps = self.game.game_state.steps
        final_rounds = self.game.game_state.round_manager.round_count if hasattr(self.game.game_state, 'round_manager') else 0
        if final_steps != steps:
            print_error(f"[SSOT] FAIL-FAST: Step count mismatch! Game state shows {final_steps} steps but loop executed {steps} steps")
            raise RuntimeError(f"[SSOT] Step count mismatch: game_state.steps={final_steps}, loop_steps={steps}")

        # --- FAIL-FAST: Ensure explanations, metrics, and moves are aligned ---
        explanations = getattr(self.game.game_state, 'move_explanations', [])
        metrics = getattr(self.game.game_state, 'move_metrics', [])
        dataset_game_states = self.game.game_state.generate_game_summary().get('dataset_game_states', {})
        # Only count pre-move states for rounds 2..N+1
        n_states = len([k for k in dataset_game_states.keys() if str(k).isdigit() and int(k) > 1])
        n_expl = len(explanations)
        n_metrics = len(metrics)
        if not (n_expl == n_metrics == n_states):
            print_error("[SSOT] FAIL-FAST: Misalignment detected after game!")
            print_error(f"[SSOT] Explanations: {n_expl}, Metrics: {n_metrics}, Pre-move states (rounds 2+): {n_states}")
            print_error(f"[SSOT] dataset_game_states keys: {list(dataset_game_states.keys())}")
            raise RuntimeError(f"[SSOT] Misalignment: explanations={n_expl}, metrics={n_metrics}, pre-move states (rounds 2+): {n_states}")

        return game_duration

    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and update datasets automatically."""
        # Increment game count before saving (matches Task-0 behavior)
        self.game_count += 1
        
        # Set game number in game state (matches Task-0 behavior)
        self.game.game_state.game_number = self.game_count
        
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
        with open(game_file, 'w', encoding='utf-8') as f:
            json.dump(BFSAgent.to_serializable(game_data), f, indent=2)

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
        with open(summary_file, 'w', encoding='utf-8') as f:
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
        
        for game_data in games_data:
            # game_count is already incremented
            game_data['game_number'] = self.game_count
            self.dataset_generator._process_single_game(game_data)

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
