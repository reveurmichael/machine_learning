from __future__ import annotations
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy 512 Token Agent - Concise BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a token-limited BFS-SAFE-GREEDY agent (512 tokens) that inherits
from the standard BFS-SAFE-GREEDY agent but generates very concise explanations.

Design Patterns:
- Inheritance: Extends BFSSafeGreedyAgent with token-limited explanations
- Strategy Pattern: Same safe-greedy pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports

# Import extension-specific components using relative imports
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions, extract_apple_position
)

if TYPE_CHECKING:
    pass


class BFSSafeGreedy512TokenAgent(BFSSafeGreedyAgent):
    """
    BFS Safe Greedy Agent with 512-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSSafeGreedyAgent (reuses all pathfinding and safety logic)
    - Overrides explanation generation methods for concise output
    - Maintains identical algorithm behavior with shorter explanations
    
    Token Limit: ~512 tokens (very concise explanations)
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy 512-token agent, extending base BFS Safe Greedy."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-SAFE-GREEDY-512"

    def _generate_safe_apple_explanation(self, game_state: dict, path: List[List[int]], 
                                        direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int,
                                        metrics: dict) -> dict:
        """
        Generate concise explanation for apple path move (512 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = extract_apple_position(game_state)
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate basic metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        next_pos = path[1] if len(path) > 1 else path[0]
        
        # Concise explanation
        explanation_parts = [
            "BFS-SAFE-GREEDY Analysis:",
            f"Head: {tuple(head_pos)} → Apple: {tuple(apple_pos)}",
            f"Path found: {path_length} steps (optimal: {path_length == manhattan_distance})",
            f"Valid moves: {valid_moves}",
            "Safety check: SAFE (tail reachable)",
            f"Moving {direction} to {tuple(next_pos)}",
            "",
            "Rationale: BFS shortest path to apple with safety validation. " +
            f"Move {direction} advances optimally while ensuring tail remains reachable.",
            f"Strategy: Safe-greedy approach. {remaining_free_cells} free cells remain."
        ]

        # Use metrics passed from parent
        explanation_dict = {
            "strategy_phase": "SAFE_APPLE_PATH",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

        return explanation_dict

    def _generate_tail_chase_explanation(self, game_state: dict, tail: List[int], path: List[List[int]], 
                                       direction: str, valid_moves: List[str], manhattan_distance: int, 
                                       remaining_free_cells: int, metrics: dict) -> dict:
        """
        Generate concise explanation for tail chase move (512 tokens max).
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = extract_apple_position(game_state)
        
        # Calculate basic metrics
        path_length = len(path) - 1
        
        # Concise explanation
        explanation_parts = [
            "BFS-SAFE-GREEDY Tail Chase:",
            f"Head: {tuple(head_pos)} → Tail: {tuple(tail)}",
            "Apple path unsafe, using tail chase strategy",
            f"Path to tail: {path_length} steps",
            f"Moving {direction} (always safe)",
            "",
            "Rationale: Apple path blocked/unsafe. Tail chasing ensures survival " +
            "while maintaining board position for future opportunities."
        ]

        # Use metrics passed from parent
        explanation_dict = {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

        return explanation_dict

