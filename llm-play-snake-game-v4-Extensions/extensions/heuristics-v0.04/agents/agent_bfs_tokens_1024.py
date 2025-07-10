from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 1024 Token Agent - Moderately detailed BFS pathfinding for Snake Game v0.04
----------------

This module implements a token-limited BFS agent (1024 tokens) that inherits
from the standard BFS agent but generates moderately detailed explanations.

Design Patterns:
- Inheritance: Extends BFSAgent with token-limited explanations
- Strategy Pattern: Same BFS pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import List, Tuple

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions
)
from heuristics_utils import count_obstacles_in_path


class BFS1024TokenAgent(BFSAgent):
    """
    BFS Agent with 1024-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses all pathfinding logic)
    - Overrides _generate_move_explanation() for moderate detail
    - Maintains identical algorithm behavior with medium explanations
    
    Token Limit: ~1024 tokens (moderately detailed explanations)
    """

    def __init__(self):
        """Initialize BFS 1024-token agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-1024"

    def _generate_move_explanation(self, game_state: dict, path: List[Tuple[int, int]], 
                                 direction: str, valid_moves: List[str],
                                 manhattan_distance: int, remaining_free_cells: int) -> dict:
        """
        Generate moderately detailed explanation for the chosen move (1024 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        board_fill_ratio = snake_length / (grid_size * grid_size)
        obstacles_avoided = count_obstacles_in_path(path, set(tuple(p) for p in body_positions))
        
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Moderate detail explanation
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "SITUATION ASSESSMENT:",
            f"• Head position: {tuple(head_pos)}",
            f"• Apple position: {tuple(apple_pos)}",
            f"• Snake length: {snake_length} segments",
            f"• Grid: {grid_size}×{grid_size}, board fill: {board_fill_ratio:.1%} ({space_pressure} pressure)",
            f"• Free cells: {remaining_free_cells}",
            "",
            "PATHFINDING RESULTS:",
            f"• Manhattan distance: {manhattan_distance} steps",
            f"• BFS path length: {path_length} steps",
            f"• Path efficiency: {efficiency_ratio:.2f} ({'optimal' if is_optimal else 'suboptimal'})",
            f"• Valid moves available: {valid_moves} ({len(valid_moves)} options)",
            f"• Obstacles near path: {obstacles_avoided}",
            "",
            "MOVE DECISION:",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",
            f"• Rationale: {'Optimal BFS path' if is_optimal else 'Best available BFS path'} to apple",
            "",
            "STRATEGIC ANALYSIS:",
            f"Moving {direction} follows the BFS-computed shortest path from {tuple(head_pos)} to {tuple(apple_pos)}. " +
            f"This advances optimally toward the apple while maintaining {remaining_free_cells - 1} free cells " +
            f"for future maneuvering. Path validated as safe with {space_pressure} board pressure."
        ]

        # Metrics matching parent format
        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),
                "path_length": int(path_length),
                "obstacles_near_path": int(obstacles_avoided),
                "remaining_free_cells": int(remaining_free_cells),
                "valid_moves": valid_moves,
                "final_chosen_direction": direction,
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": int(snake_length),
                "grid_size": int(grid_size),
            },
            "explanation_steps": explanation_parts,
        }

        return explanation_dict
