from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 2048 Token Agent - Detailed BFS pathfinding for Snake Game v0.04
----------------

This module implements a token-limited BFS agent (2048 tokens) that inherits
from the standard BFS agent but generates detailed explanations.

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


class BFS2048TokenAgent(BFSAgent):
    """
    BFS Agent with 2048-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses all pathfinding logic)
    - Overrides _generate_move_explanation() for detailed output
    - Maintains identical algorithm behavior with detailed explanations
    
    Token Limit: ~2048 tokens (detailed explanations)
    """

    def __init__(self):
        """Initialize BFS 2048-token agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-2048"

    def _generate_move_explanation(self, game_state: dict, path: List[Tuple[int, int]], 
                                 direction: str, valid_moves: List[str],
                                 manhattan_distance: int, remaining_free_cells: int) -> dict:
        """
        Generate detailed explanation for the chosen move (2048 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = BFSAgent.extract_body_positions(game_state)
        
        # Calculate detailed metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        obstacles_avoided = self._count_obstacles_in_path(path, set(tuple(p) for p in body_positions))
        
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"
        
        # Detailed explanation (condensed from full version)
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {tuple(head_pos)}",
            f"• Target apple position: {tuple(apple_pos)}",
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",
            f"• Snake length: {snake_length} segments",
            f"• Grid dimensions: {grid_size}×{grid_size} ({grid_size * grid_size} total cells)",
            f"• Board occupation: {snake_length}/{grid_size * grid_size} cells ({board_fill_ratio:.1%}) - {space_pressure} space pressure",
            f"• Free cells remaining: {remaining_free_cells}",
            "",
            "PHASE 2: MOVE VALIDATION",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• Rejected moves: {list(set(['UP', 'DOWN', 'LEFT', 'RIGHT']) - set(valid_moves))}",
            "• Validation criteria: no wall collisions, no body collisions, within grid bounds",
            "",
            "PHASE 3: BFS PATHFINDING EXECUTION",
            f"• Algorithm: Breadth-First Search from {tuple(head_pos)} to {tuple(apple_pos)}",
            f"• Search space: {grid_size * grid_size - snake_length} accessible cells",
            f"• Obstacles to navigate: {snake_length - 1} body segments",
            f"• Manhattan distance baseline: {manhattan_distance} steps (theoretical minimum)",
            "",
            "PHASE 4: PATH ANALYSIS RESULTS",
            f"• Shortest path found: {path_length} steps",
            f"• Path efficiency: {efficiency_str}",
            f"• Path optimality: {'OPTIMAL - no detours required' if is_optimal else 'SUB-OPTIMAL - includes ' + str(detour_steps) + ' detour step(s)'}",
            f"• Obstacles near path: {obstacles_avoided} body segments in adjacent cells",
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path[:min(4, len(path))]])}{'...' if len(path) > 4 else ''}",
            "",
            "PHASE 5: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",
            "• Rationale: First step of shortest path to apple",
            "• Risk assessment: LOW (validated safe move on optimal path)",
            "",
            "PHASE 6: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1}",
            "• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",
            "• Risk mitigation: BFS guarantees shortest path, minimizing exposure time",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} follows the BFS-computed {'optimal' if is_optimal else 'best available'} path " +
            f"to the apple at {tuple(apple_pos)}. This move advances from {tuple(head_pos)} to {next_pos}, " +
            f"maintaining {'perfect trajectory efficiency' if is_optimal else f'good efficiency despite {detour_steps} detour(s)'}. " +
            f"Decision validated against {len(valid_moves)} valid options with {space_pressure} board pressure."
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
