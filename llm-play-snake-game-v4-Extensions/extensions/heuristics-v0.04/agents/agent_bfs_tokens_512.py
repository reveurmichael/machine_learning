from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 512 Token Agent - Concise BFS pathfinding for Snake Game v0.04
----------------

This module implements a token-limited BFS agent (512 tokens) that inherits
from the standard BFS agent but generates very concise explanations.

Design Patterns:
- Inheritance: Extends BFSAgent with token-limited explanations
- Strategy Pattern: Same BFS pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import List, Tuple, Dict, Any

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions
)


class BFS512TokenAgent(BFSAgent):
    """
    BFS Agent with 512-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses all pathfinding logic)
    - Overrides _generate_move_explanation() for concise output
    - Maintains identical algorithm behavior with shorter explanations
    
    Token Limit: ~512 tokens (very concise explanations)
    """

    def __init__(self):
        """Initialize BFS 512-token agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-512"

    def _generate_move_explanation(self, game_state: dict, path: List[Tuple[int, int]], 
                                 direction: str, valid_moves: List[str],
                                 manhattan_distance: int, remaining_free_cells: int) -> dict:
        """
        Generate concise explanation for the chosen move (512 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate basic metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        # Concise explanation
        explanation_parts = [
            "BFS Analysis:",
            f"Head: {tuple(head_pos)} → Apple: {tuple(apple_pos)}",
            f"Path found: {path_length} steps (optimal: {path_length == manhattan_distance})",
            f"Valid moves: {valid_moves}",
            f"Moving {direction} to {next_pos}",
            "",
            f"Rationale: BFS shortest path to apple. Move {direction} advances optimally toward target.",
            f"Safety: Validated move on computed path. {remaining_free_cells} free cells remain."
        ]

        # Metrics matching parent format
        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),
                "path_length": int(path_length),
                "obstacles_near_path": self._count_obstacles_in_path(path, set(tuple(p) for p in body_positions)),
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


