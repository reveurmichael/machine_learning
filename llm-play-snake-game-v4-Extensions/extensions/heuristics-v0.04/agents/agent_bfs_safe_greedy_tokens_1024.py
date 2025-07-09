from __future__ import annotations
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy 1024 Token Agent - Moderate BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a token-limited BFS-SAFE-GREEDY agent (1024 tokens) that inherits
from the standard BFS-SAFE-GREEDY agent but generates moderately detailed explanations.

Design Patterns:
- Inheritance: Extends BFSSafeGreedyAgent with token-limited explanations
- Strategy Pattern: Same safe-greedy pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import TYPE_CHECKING, Dict, Any

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from utils.moves_utils import position_to_direction

# Import extension-specific components using relative imports
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from .agent_bfs import BFSAgent

if TYPE_CHECKING:
    pass


class BFSSafeGreedy1024TokenAgent(BFSSafeGreedyAgent):
    """
    BFS Safe Greedy Agent with 1024-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSSafeGreedyAgent (reuses all pathfinding and safety logic)
    - Overrides explanation generation methods for moderate detail
    - Maintains identical algorithm behavior with medium explanations
    
    Token Limit: ~1024 tokens (moderately detailed explanations)
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy 1024-token agent, extending base BFS Safe Greedy."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-SAFE-GREEDY-1024"

    def _generate_safe_apple_explanation(self, game_state: dict, path: List[List[int]], 
                                        direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int,
                                        metrics: dict) -> dict:
        """
        Generate moderately detailed explanation for safe apple path move (1024 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = BFSAgent.extract_apple_position(game_state)
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = BFSAgent.extract_body_positions(game_state)
        
        # Calculate metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        board_fill_ratio = snake_length / (grid_size * grid_size)
        next_pos = path[1] if len(path) > 1 else path[0]
        
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Moderate detail explanation
        explanation_parts = [
            "=== BFS-SAFE-GREEDY PATHFINDING ANALYSIS ===",
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
            "",
            "SAFETY VALIDATION:",
            f"• Safety check: PASSED (tail reachable after move)",
            f"• Next position: {tuple(next_pos)}",
            f"• Risk assessment: LOW (move validated as safe)",
            "",
            "MOVE DECISION:",
            f"• Chosen direction: {direction}",
            f"• Strategy: Safe-greedy (safety-validated apple pursuit)",
            f"• Rationale: {'Optimal safe path' if is_optimal else 'Best available safe path'} to apple",
            "",
            "STRATEGIC ANALYSIS:",
            f"Moving {direction} follows the BFS-computed shortest safe path from {tuple(head_pos)} to {tuple(apple_pos)}. " +
            "Safety validation confirms tail remains reachable after this move, ensuring the snake won't become trapped. " +
            f"This advances optimally toward the apple while maintaining {remaining_free_cells - 1} free cells " +
            f"for future maneuvering with {space_pressure} board pressure."
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
        Generate moderately detailed explanation for tail chase move (1024 tokens max).
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = BFSAgent.extract_apple_position(game_state)
        grid_size = game_state.get('grid_size', 10)
        
        # Calculate metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Moderate detail explanation
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: TAIL CHASE STRATEGY ===",
            "",
            "PRIMARY STRATEGY FAILURE:",
            f"• Apple pathfinding to {tuple(apple_pos)}: FAILED",
            "• Failure reason: Safety validation rejected apple path",
            "• Risk detected: Potential self-trapping if pursuing apple",
            "",
            "SECONDARY STRATEGY - TAIL CHASING:",
            f"• Target: Snake tail at {tuple(tail)}",
            "• Strategy rationale: Tail chasing is always safe (tail moves away)",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• BFS pathfinding from {tuple(head_pos)} to {tuple(tail)}",
            f"• Tail chase path found: {path_length} steps",
            "",
            "SAFETY ANALYSIS:",
            "• Safety guarantee: ABSOLUTE (tail moves as snake advances)",
            "• Self-collision risk: ZERO (impossible to catch moving tail)",
            "• Space preservation: Maintains current board position",
            f"• Board pressure: {space_pressure} ({board_fill_ratio:.1%} occupation)",
            "",
            "MOVE DECISION:",
            f"• Chosen direction: {direction}",
            f"• Next position: Following tail at distance {path_length}",
            f"• Apple distance: {manhattan_distance} steps (for future reference)",
            f"• Space management: Preserving {remaining_free_cells} free cells",
            "",
            "STRATEGIC POSITIONING:",
            "BFS-Safe-Greedy activated tail chase strategy after determining apple pursuit was unsafe. " +
            f"Moving {direction} toward tail provides guaranteed safety while maintaining board position. " +
            "This defensive strategy preserves survival until safer apple pursuit opportunities emerge, " +
            "demonstrating the algorithm's adaptive safety-first approach."
        ]

        # Use metrics passed from parent
        explanation_dict = {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

        return explanation_dict
