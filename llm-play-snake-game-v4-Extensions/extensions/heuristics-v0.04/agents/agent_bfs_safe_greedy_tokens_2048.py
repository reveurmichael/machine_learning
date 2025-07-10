from __future__ import annotations
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy 2048 Token Agent - Detailed BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a token-limited BFS-SAFE-GREEDY agent (2048 tokens) that inherits
from the standard BFS-SAFE-GREEDY agent but generates detailed explanations.

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
from heuristics_utils import count_obstacles_in_path

if TYPE_CHECKING:
    pass


class BFSSafeGreedy2048TokenAgent(BFSSafeGreedyAgent):
    """
    BFS Safe Greedy Agent with 2048-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSSafeGreedyAgent (reuses all pathfinding and safety logic)
    - Overrides explanation generation methods for detailed output
    - Maintains identical algorithm behavior with detailed explanations
    
    Token Limit: ~2048 tokens (detailed explanations)
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy 2048-token agent, extending base BFS Safe Greedy."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-SAFE-GREEDY-2048"

    def _generate_safe_apple_explanation(self, game_state: dict, path: List[List[int]], 
                                        direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int,
                                        metrics: dict) -> dict:
        """
        Generate detailed explanation for safe apple path move (2048 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = extract_apple_position(game_state)
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate detailed metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        obstacles_avoided = count_obstacles_in_path(path, set(tuple(p) for p in body_positions))
        next_pos = path[1] if len(path) > 1 else path[0]
        
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"
        
        # Detailed explanation (condensed from full version)
        explanation_parts = [
            "=== BFS-SAFE-GREEDY PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {tuple(head_pos)}",
            f"• Target apple position: {tuple(apple_pos)}",
            f"• Snake body positions: {[tuple(p) for p in body_positions[:5]]}{'...' if len(body_positions) > 5 else ''}",
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
            "PHASE 5: SAFETY VALIDATION",
            "• Safety check: PASSED (tail reachable after move)",
            f"• Next position: {tuple(next_pos)}",
            "• Risk assessment: LOW (move validated as safe)",
            "• Safety guarantee: Tail remains accessible for escape route",
            "",
            "PHASE 6: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            "• Strategy: Safe-greedy (safety-validated apple pursuit)",
            f"• Rationale: {'Optimal safe path' if is_optimal else 'Best available safe path'} to apple",
            "• Expected outcome: Advance 1 step closer to apple along safe route",
            "",
            "PHASE 7: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1}",
            "• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",
            "• Risk mitigation: Safety validation prevents trapping scenarios",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} follows the BFS-computed {'optimal' if is_optimal else 'best available'} safe path " +
            f"to the apple at {tuple(apple_pos)}. This move advances from {tuple(head_pos)} to {tuple(next_pos)}, " +
            f"maintaining {'perfect trajectory efficiency' if is_optimal else f'good efficiency despite {detour_steps} detour(s)'} " +
            f"while ensuring tail remains reachable. Decision validated against {len(valid_moves)} valid options " +
            f"with {space_pressure} board pressure, demonstrating safe-greedy algorithm's balanced approach."
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
        Generate detailed explanation for tail chase move (2048 tokens max).
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = extract_apple_position(game_state)
        grid_size = game_state.get('grid_size', 10)
        
        # Calculate detailed metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Detailed explanation (condensed from full version)
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: TAIL CHASE STRATEGY ===",
            "",
            "PHASE 1: PRIMARY STRATEGY FAILURE ANALYSIS",
            f"• Primary strategy attempted: Apple pathfinding to {tuple(apple_pos)}",
            "• Primary strategy result: FAILED (unsafe or no path found)",
            "• Failure reason: Safety validation rejected apple path",
            "• Risk detected: Potential self-trapping if pursuing apple",
            "• Algorithm response: Activate secondary strategy",
            "",
            "PHASE 2: SECONDARY STRATEGY - TAIL CHASING",
            "• Strategy priority: SECONDARY (defensive positioning)",
            f"• Target: Snake tail at {tuple(tail)}",
            "• Rationale: Tail chasing is always safe (tail moves away)",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• BFS pathfinding from {tuple(head_pos)} to {tuple(tail)}",
            f"• Tail chase path found: {path_length} steps",
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path[:min(4, len(path))]])}{'...' if len(path) > 4 else ''}",
            "",
            "PHASE 3: TAIL CHASE SAFETY ANALYSIS",
            "• Safety guarantee: ABSOLUTE (tail moves as snake advances)",
            "• Self-collision risk: ZERO (impossible to catch moving tail)",
            "• Space preservation: Maintains current board position",
            "• Future opportunities: Keeps options open for apple pursuit",
            f"• Board pressure: {space_pressure} ({board_fill_ratio:.1%} occupation)",
            "",
            "PHASE 4: STRATEGIC POSITIONING",
            f"• Current head position: {tuple(head_pos)}",
            f"• Chosen direction: {direction}",
            f"• Next position: Following tail at distance {path_length}",
            f"• Apple distance: {manhattan_distance} steps (for future reference)",
            f"• Space management: Preserving {remaining_free_cells} free cells",
            "• Positioning benefit: Maintains mobility while avoiding risks",
            "",
            "PHASE 5: SAFE-GREEDY DEFENSIVE LOGIC",
            "• Algorithm strength: Never pursues risky apple paths",
            "• Fallback reliability: Tail chasing provides guaranteed safe moves",
            "• vs Standard BFS: Would attempt unsafe apple path",
            "• Adaptive behavior: Switches strategies based on safety assessment",
            "",
            "=== CONCLUSION ===",
            "BFS-Safe-Greedy activated tail chase strategy after determining apple pursuit was unsafe. " +
            f"Moving {direction} toward tail at {tuple(tail)} provides guaranteed safety while maintaining " +
            "board position. This defensive strategy preserves the snake's survival until safer " +
            "apple pursuit opportunities emerge, demonstrating the algorithm's adaptive safety-first approach."
        ]

        # Use metrics passed from parent
        explanation_dict = {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

        return explanation_dict
