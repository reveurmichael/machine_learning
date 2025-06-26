"""
Heuristic Game Data - Data tracking for heuristic algorithms
--------------------

This module extends BaseGameData to add heuristic-specific tracking
while maintaining compatibility with the base game data structure.

Design Philosophy:
- Extends BaseGameData (inherits all generic game state)
- Adds heuristic-specific metrics (algorithm name, path calculations)
- Maintains same JSON output format as Task-0 for compatibility
- Uses BaseGameStatistics instead of LLM-specific GameStatistics
"""

from __future__ import annotations

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from typing import Dict, Any, Optional

from core.game_data import BaseGameData
from core.game_stats_manager import NumPyJSONEncoder


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
        
        # Heuristic-specific tracking
        self.algorithm_name: str = "BFS"  # Default algorithm
        self.path_calculations: int = 0   # Number of pathfinding calls
        self.average_path_length: float = 0.0
        self.successful_paths: int = 0
        self.failed_paths: int = 0
        
        # Track search performance
        self.total_search_time: float = 0.0
        self.nodes_explored: int = 0
        
        # v0.04 Enhancement: Store move explanations for JSONL dataset generation
        self.last_move_explanation: str = ""
        self.move_explanations: list[str] = []  # Store all explanations for this game
        
    def reset(self) -> None:
        """Reset game data for new game."""
        super().reset()
        
        # Reset heuristic-specific counters
        self.path_calculations = 0
        self.average_path_length = 0.0
        self.successful_paths = 0
        self.failed_paths = 0
        self.total_search_time = 0.0
        self.nodes_explored = 0
        
        # Reset v0.04 explanation tracking
        self.last_move_explanation = ""
        self.move_explanations = []
    
    def record_pathfinding_attempt(self, success: bool, path_length: int = 0, 
                                 search_time: float = 0.0, nodes_explored: int = 0) -> None:
        """
        Record a pathfinding attempt for statistics.
        
        Args:
            success: Whether pathfinding was successful
            path_length: Length of found path (if successful)
            search_time: Time taken for search (seconds)
            nodes_explored: Number of nodes explored during search
        """
        self.path_calculations += 1
        self.total_search_time += search_time
        self.nodes_explored += nodes_explored
        
        if success:
            self.successful_paths += 1
            # Update average path length
            if self.successful_paths > 0:
                total_length = (self.average_path_length * (self.successful_paths - 1)) + path_length
                self.average_path_length = total_length / self.successful_paths
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
            "average_path_length": round(self.average_path_length, 2),
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
        """
        summary = {
            # Outcome ---------------------
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": self.game_over,
            "game_end_reason": self.game_end_reason,
            "round_count": self.round_manager.round_count,
            # Heuristic provenance -------------
            "heuristic_info": {
                "algorithm": self.algorithm_name,
                "primary_provider": primary_provider,
                "primary_model": primary_model,
            },
            # Timings / generic stats ----------
            "time_stats": self.stats.time_stats.asdict(),
            "step_stats": self.stats.step_stats.asdict(),
            # Misc metadata --------------------
            "metadata": {
                "timestamp": self.timestamp,
                "game_number": self.game_number,
                **kwargs.get("metadata", {}),
            },
            # Replay data ---------------------
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": self.moves,
                "rounds_data": self.round_manager.get_ordered_rounds_data(),
                # v0.04 Enhancement: Include move explanations for JSONL dataset generation
                "move_explanations": self.move_explanations,
            },
        }
        return summary

    # ---------------------
    # Serialisation helper – mirrors core.GameData.save_game_summary
    # ---------------------

    def save_game_summary(self, filepath: str, **kwargs):  # type: ignore[override]
        """Write *game_N.json* using the local `generate_game_summary()`."""
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