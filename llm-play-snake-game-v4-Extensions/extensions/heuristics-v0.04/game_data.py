"""
Heuristic Game Data - Data tracking for heuristic algorithms
----------------

This module extends BaseGameData to add heuristic-specific tracking
while maintaining compatibility with the base game data structure.

Design Philosophy:
- Extends BaseGameData (inherits all generic game state)
- Adds heuristic-specific metrics (algorithm name, path calculations)
- Maintains same JSON output format as Task-0 for compatibility
- Uses BaseGameStatistics instead of LLM-specific GameStatistics
- Custom TimeStats without LLM pollution
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from core.game_data import BaseGameData
from core.game_stats_manager import NumPyJSONEncoder
from game_rounds import HeuristicRoundManager


@dataclass
class HeuristicTimeStats:
    """Time statistics for heuristic algorithms without LLM pollution.
    
    This class provides the same timing functionality as TimeStats but
    excludes LLM-specific fields like llm_communication_time.
    """
    start_time: float
    
    def record_end_time(self) -> None:
        """Record the end time."""
        self.end_time = time.time()
    
    def asdict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        end = getattr(self, 'end_time', None) or time.time()
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": end - self.start_time,
        }


class HeuristicGameData(BaseGameData):
    """
    Game data tracking for heuristic algorithms.
    
    Extends BaseGameData with heuristic-specific attributes while
    maintaining the same core functionality and JSON output format.
    
    Design Patterns:
    - Template Method: Inherits base data management structure
    - Strategy Pattern: Different heuristic algorithms share same data structure
    """
    
    def __init__(self) -> None:
        """Initialize heuristic game data tracking."""
        super().__init__()
        
        # Override round_manager to use heuristic-specific version
        self.round_manager = HeuristicRoundManager()
        
        # Override time_stats to use heuristic-specific version without LLM pollution
        self.stats.time_stats = HeuristicTimeStats(start_time=time.time())
        
        # Heuristic-specific tracking
        self.algorithm_name: str = "BFS"  # Default algorithm
        self.path_calculations: int = 0   # Number of pathfinding calls
        self.successful_paths: int = 0
        self.failed_paths: int = 0
        
        # Track search performance
        self.total_search_time: float = 0.0
        self.nodes_explored: int = 0
        
        # Grid size (will be set by game logic)
        self.grid_size: int = 10  # Default, will be overridden
        
        # v0.04 Enhancement: Store move explanations for JSONL dataset generation
        self.last_move_explanation: str = ""
        self.move_explanations: list[str] = []  # Store all explanations for this game
        # v0.04 Enhancement: Store structured metrics per move
        self.move_metrics: list[dict] = []
        
    def reset(self) -> None:
        """Reset game data for new game."""
        super().reset()
        
        # Ensure we use heuristic-specific round manager
        self.round_manager = HeuristicRoundManager()
        
        # Ensure we use heuristic-specific time stats without LLM pollution
        self.stats.time_stats = HeuristicTimeStats(start_time=time.time())
        
        # Reset heuristic-specific counters
        self.path_calculations = 0
        self.successful_paths = 0
        self.failed_paths = 0
        self.total_search_time = 0.0
        self.nodes_explored = 0
        
        # Reset v0.04 explanation tracking
        self.last_move_explanation = ""
        self.move_explanations = []
        self.move_metrics = []
        # (No SSOT validation or round 0 recording here; handled in game logic)
    
    def record_pathfinding_attempt(self, success: bool, search_time: float = 0.0, nodes_explored: int = 0) -> None:
        """
        Record a pathfinding attempt for statistics.
        
        Args:
            success: Whether pathfinding was successful
            search_time: Time taken for search (seconds)
            nodes_explored: Number of nodes explored during search
        """
        self.path_calculations += 1
        self.total_search_time += search_time
        self.nodes_explored += nodes_explored
        
        if success:
            self.successful_paths += 1
        else:
            self.failed_paths += 1
    
    def record_move_explanation(self, explanation: str) -> None:
        """
        Record move explanation for JSONL dataset generation.
        
        v0.04 Enhancement: Store explanations for each move to enable
        rich dataset creation for LLM fine-tuning.
        
        Args:
            explanation: Natural language explanation of the move reasoning
        """
        self.move_explanations.append(explanation)
        self.last_move_explanation = explanation
    
    def record_move_metrics(self, metrics: dict) -> None:
        """Record structured metrics for the current move."""
        self.move_metrics.append(metrics)
    
    def get_heuristic_stats(self) -> Dict[str, Any]:
        """
        Get heuristic-specific statistics.
        
        Returns:
            Dictionary containing heuristic performance metrics
        """
        success_rate = (self.successful_paths / max(self.path_calculations, 1)) * 100
        avg_search_time = self.total_search_time / max(self.path_calculations, 1)
        avg_nodes_per_search = self.nodes_explored / max(self.path_calculations, 1)
        
        return {
            "algorithm_name": self.algorithm_name,
            "path_calculations": self.path_calculations,
            "successful_paths": self.successful_paths,
            "failed_paths": self.failed_paths,
            "success_rate_percent": round(success_rate, 2),
            "total_search_time": round(self.total_search_time, 4),
            "average_search_time": round(avg_search_time, 4),
            "total_nodes_explored": self.nodes_explored,
            "average_nodes_per_search": round(avg_nodes_per_search, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert game data to dictionary for JSON serialization.
        
        Extends the base to_dict method with heuristic-specific data.
        
        Returns:
            Dictionary representation of game data
        """
        # Get base dictionary
        data = super().to_dict()
        
        # Add heuristic-specific data
        data["heuristic_info"] = {
            "algorithm": self.algorithm_name,
            "path_calculations": self.path_calculations,
            "successful_paths": self.successful_paths,
            "failed_paths": self.failed_paths
        }
        
        # Add heuristic stats
        data["heuristic_stats"] = self.get_heuristic_stats()
        
        # # v0.04 Enhancement: Add move explanations and metrics for dataset generation
        # data["move_explanations"] = self.move_explanations
        # data["move_metrics"] = self.move_metrics
        
        return data

    def generate_game_summary(
        self,
        primary_provider: str = "bfs",
        primary_model: Optional[str] = None,
        parser_provider: Optional[str] = None,
        parser_model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Produce a Task-0 compatible *game_N.json* dictionary.

        The structure purposefully mirrors the canonical `GameData` output so
        that replay tooling and dashboards work *unchanged* for heuristic
        experiments.  Fields that are strictly LLM-specific (token stats,
        prompt/response timings, etc.) are **omitted** here while retaining
        identical keys for the shared data blocks (time_stats, step_stats,
        detailed_history, …).
        
        Changes for Task-0 compatibility:
        - Removed heuristic_info->primary_provider and primary_model 
        - Removed time_stats->llm_communication_time
        - Removed detailed_history->rounds_data->N->game_state
        - Removed detailed_history->move_explanations
        - Added grid_size field
        - Fixed step_stats to show correct values
        - Ensured planned_moves matches moves in rounds_data
        - Fixed game_over and game_end_reason to be accurate
        
        v0.04 Enhancement: Added dataset_game_states for dataset generation
        while keeping game files clean for Task-0 compatibility.
        """
        # Clean rounds data: remove game_state and ensure planned_moves matches moves
        cleaned_rounds_data = {}
        dataset_game_states = {}  # Store game states for dataset generation
        
        # Clean rounds data: remove game_state and ensure planned_moves matches moves
        ordered_rounds = self.round_manager.get_ordered_rounds_data()
        for round_key, round_data in ordered_rounds.items():
            cleaned_round = {
                "round": round_data.get("round", int(round_key)),
                "apple_position": round_data.get("apple_position", [0, 0])
            }
            
            # Add moves if present
            if "moves" in round_data:
                cleaned_round["moves"] = round_data["moves"]
            
            # Always include planned_moves for consistency with Task-0 pipeline logic
            # This represents the planning phase, while moves represents the execution phase
            if "planned_moves" in round_data:
                cleaned_round["planned_moves"] = round_data["planned_moves"]
            
            cleaned_rounds_data[round_key] = cleaned_round
            
            # Store game state for dataset generation (separate from Task-0 compatible data)
            if "game_state" in round_data:
                dataset_game_states[round_key] = round_data["game_state"]
        
        # SSOT: Fail fast if round 1 is missing from dataset_game_states
        if 1 not in dataset_game_states and '1' not in dataset_game_states:
            raise RuntimeError(f"[SSOT] Round 1 missing from dataset_game_states. Available: {list(dataset_game_states.keys())}")
        
        # Game state (single termination point ensures consistency)
        game_over = self.game_over
        game_end_reason = self.game_end_reason
        
        # Time stats (heuristics never have llm_communication_time)
        time_stats_clean = self.stats.time_stats.asdict()
        
        # For heuristics: use round-by-round moves from rounds data
        moves_from_rounds = []
        
        # Build moves from rounds, using the already-built dataset_game_states
        # DO NOT overwrite dataset_game_states - it contains the correct pre-move states!
        ordered_rounds = self.round_manager.get_ordered_rounds_data()
        for round_key in sorted(ordered_rounds.keys()):
            if int(round_key) == 0:
                continue
            round_data = ordered_rounds[round_key]
            if 'moves' in round_data and round_data['moves']:
                # For heuristics, there should be exactly one move per round
                move = round_data['moves'][0]
                moves_from_rounds.append(move)
        
        # Ensure all game states have the game_number for consistency
        for state in dataset_game_states.values():
            if isinstance(state, dict):
                state['game_number'] = self.game_number
        
        summary = {
            # Core outcome data
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": game_over,
            "game_end_reason": game_end_reason,
            "round_count": self.round_manager.round_count,
            # Heuristic-specific data
            "grid_size": self.grid_size,
            # Statistics
            "time_stats": time_stats_clean,
            "step_stats": self.stats.step_stats.asdict(),
            # Metadata
            "metadata": {
                "timestamp": self.timestamp,
                "game_number": self.game_number,
                "round_count": self.round_manager.round_count,
                **kwargs.get("metadata", {}),
            },
            # Replay data (cleaned for Task-0 compatibility)
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": moves_from_rounds,
                "rounds_data": cleaned_rounds_data,
            },
        }
        if dataset_game_states:
            summary["dataset_game_states"] = dataset_game_states
        
        return summary

    # ----------------
    # Serialisation helper – mirrors core.GameData.save_game_summary
    # ----------------

    def save_game_summary(self, filepath: str, **kwargs):  # type: ignore[override]
        """Write *game_N.json* using the local `generate_game_summary()`.
        Always writes with UTF-8 encoding for full Unicode compatibility.
        """
        # Ensure the final round buffer is persisted.
        if hasattr(self, "round_manager") and self.round_manager:
            self.round_manager.flush_buffer()

        summary_dict = self.generate_game_summary(**kwargs)

        import os
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, cls=NumPyJSONEncoder, indent=2)

        return summary_dict 

    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics for heuristics.
        
        POST-EXECUTION: This method is called AFTER the move has been executed.
        The move parameter is the direction that was just executed, and apple_eaten
        indicates whether the snake ate an apple during this move.
        
        This method ensures step_stats are correctly updated for heuristic algorithms.
        The base class doesn't update step_stats, so we need to do it here.
        
        Args:
            move: The move direction that was just executed (POST-MOVE)
            apple_eaten: Whether an apple was eaten during this move (POST-MOVE)
        """
        # Call base class method which handles basic move recording
        super().record_move(move, apple_eaten)
        
        # Update step statistics based on move type
        # POST-EXECUTION: These stats reflect the move that was just executed
        if move == "INVALID_REVERSAL":
            self.stats.step_stats.invalid_reversals += 1
        elif move == "NO_PATH_FOUND":
            self.stats.step_stats.no_path_found += 1
        else:
            # Valid move (UP, DOWN, LEFT, RIGHT)
            self.stats.step_stats.valid += 1

    def record_game_end(self, reason: str) -> None:
        """Record the end of a game with proper heuristic timing.
        
        POST-EXECUTION: This method is called when the game ends, after all moves
        have been executed. The reason parameter indicates why the game ended.
        
        Args:
            reason: The reason the game ended (from END_REASON_MAP) (POST-GAME)
        """
        if not self.game_over:
            self.stats.time_stats.record_end_time()
        super().record_game_end(reason) 