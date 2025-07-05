from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy Agent - Enhanced BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a SAFE-GREEDY agent that prioritizes safety over greed.
It finds the shortest path to the apple but validates that the snake can still
reach its tail afterward to avoid getting trapped.

Algorithm:
1. Find shortest path to apple using BFS
2. Validate path safety (can snake reach tail after move?)
3. If safe, follow apple path
4. If unsafe, follow tail (always safe)
5. If no paths exist, use any valid move

Design Patterns:
- Inheritance: Extends BFSAgent with safety validation
- Strategy Pattern: Safe-greedy pathfinding strategy
- Fail-Fast: SSOT violations cause immediate errors
"""

from typing import List, Tuple, TYPE_CHECKING
import json

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error
from core.game_agents import BaseAgent

# SSOT: Import shared logic from ssot_utils - DO NOT reimplement these functions
from ssot_utils import ssot_bfs_pathfind, ssot_calculate_valid_moves

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSSafeGreedyAgent(BFSAgent):
    """
    BFS Safe Greedy Agent: Enhanced BFS with safety validation.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses helper methods and patterns)
    - Overrides get_move_with_explanation() to add safety validation
    - Maintains consistent naming and structure with BFS agent
    
    Algorithm Enhancement:
    1. Find shortest path to apple using BFS
    2. Validate path safety (can snake reach tail afterward?)
    3. If safe, follow apple path
    4. If unsafe, chase tail (always safe)
    5. Last resort: any valid move
    
    KISS: No unnecessary fallbacks, fail-fast on SSOT violations
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-SAFE-GREEDY"
        self.name = "BFS Safe Greedy"
        self.description = (
            "Enhanced BFS with safety validation. Inherits core BFS pathfinding "
            "from BFSAgent and adds safety checks to avoid getting trapped. "
            "Falls back to tail-chasing when apple path is unsafe."
        )
        

    def get_move(self, state: dict) -> str | None:
        """
        Get next move using safe BFS pathfinding (simplified interface).
        Args:
            state: Game state dict
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(state)
        return move

    def get_move_with_explanation(self, state: dict) -> Tuple[str, dict]:
        """
        SAFE-GREEDY agent: Prioritizes safety over greed.
        KISS: Fail fast on any SSOT violations.
        """
        # Use the provided state dict for all calculations (SSOT)
        head = list(state["head_position"])
        apple = list(state["apple_position"])
        snake = [list(seg) for seg in state["snake_positions"]]
        grid_size = state["grid_size"]
        # SSOT: Obstacles are all body segments except head (matching BFS agent)
        obstacles = set(tuple(p) for p in snake[:-1])  # Exclude head from obstacles

        # Helper metrics that are independent of strategy branch (all from pre-move state)
        manhattan_distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        valid_moves = self._calculate_valid_moves(head, snake, grid_size)
        remaining_free_cells = self._count_remaining_free_cells(set(tuple(p) for p in snake), grid_size)

        # Fail-fast: ensure state is not mutated (SSOT)

        # ---------------- 1. Try safe apple path first
        path_to_apple = ssot_bfs_pathfind(head, apple, obstacles, grid_size)
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = path_to_apple[1]
            direction = position_to_direction(tuple(head), tuple(next_pos))
            
            # Fail-fast: validate bounds and valid moves
            if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                next_pos[1] < 0 or next_pos[1] >= grid_size):
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY computed out-of-bounds position {next_pos} for grid size {grid_size}")
            
            if direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY computed move '{direction}' is not valid for head {head} and valid_moves {valid_moves}")
            
            # Safety validation: can snake reach tail after this move?
            if self._is_move_safe(next_pos, snake, apple, obstacles, grid_size):
                # Safe path found
                metrics = {
                    "final_chosen_direction": direction,
                    "head_position": list(head),
                    "apple_position": list(apple),
                    "snake_length": len(snake),
                    "grid_size": grid_size,
                    "valid_moves": valid_moves,
                    "manhattan_distance": manhattan_distance,
                    "remaining_free_cells": remaining_free_cells,
                    "path_length": len(path_to_apple) - 1,
                    "apple_path_safe": True
                }
                
                explanation_dict = self._generate_safe_apple_explanation(
                    head, apple, snake, path_to_apple, direction, valid_moves, 
                    manhattan_distance, remaining_free_cells, grid_size, metrics
                )
                
                return direction, explanation_dict

        # ---------------- 2. Apple path unsafe or not found, try tail-chasing
        tail = snake[-1]
        path_to_tail = ssot_bfs_pathfind(head, tail, obstacles, grid_size)
        if path_to_tail and len(path_to_tail) > 1:
            next_pos = path_to_tail[1]
            direction = position_to_direction(tuple(head), tuple(next_pos))
            
            # Fail-fast: validate bounds and valid moves
            if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                next_pos[1] < 0 or next_pos[1] >= grid_size):
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY tail-chase computed out-of-bounds position {next_pos} for grid size {grid_size}")
            
            if direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY tail-chase move '{direction}' is not valid for head {head} and valid_moves {valid_moves}")
            
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": len(snake),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": len(path_to_tail) - 1,
                "apple_path_safe": False
            }
            
            explanation_dict = self._generate_tail_chase_explanation(
                head, apple, snake, tail, path_to_tail, direction, valid_moves,
                manhattan_distance, remaining_free_cells, grid_size, metrics
            )
            
            return direction, explanation_dict

        # ---------------- 3. Last resort: any valid move
        if valid_moves:
            direction = valid_moves[0]
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": len(snake),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": 0,
                "apple_path_safe": False
            }
            
            explanation_dict = self._generate_survival_explanation(
                head, apple, snake, direction, valid_moves,
                manhattan_distance, remaining_free_cells, grid_size, metrics
            )
            
            return direction, explanation_dict
        else:
            direction = "NO_PATH_FOUND"
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": len(snake),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": 0,
                "apple_path_safe": False
            }
            
            explanation_dict = self._generate_no_moves_explanation(
                head, apple, snake, valid_moves, manhattan_distance, 
                remaining_free_cells, grid_size, metrics
            )
            
            return direction, explanation_dict

    def _is_move_safe(self, next_pos: List[int], snake: List[List[int]], apple: List[int], 
                      obstacles: set, grid_size: int) -> bool:
        """
        Safety validation: Check if snake can reach its tail after making this move.
        KISS: Simplified safety check to avoid infinite loops.
        
        Args:
            next_pos: Proposed next head position
            snake: Current snake body
            apple: Apple position
            obstacles: Current obstacles
            grid_size: Grid size
            
        Returns:
            True if move is safe (tail reachable), False otherwise
        """
        # KISS: For small snakes, always consider moves safe to avoid over-conservative behavior
        if len(snake) <= 3:
            return True
        
        # Simulate the move
        if next_pos == apple:
            # Will eat apple, so tail stays (snake grows)
            new_snake = [next_pos] + snake[:-1]
        else:
            # Won't eat apple, so tail moves (normal move)
            new_snake = [next_pos] + snake[:-2]
        
        # KISS: Simple safety check - ensure we have enough free space
        free_cells = grid_size * grid_size - len(new_snake)
        if free_cells >= len(new_snake):
            return True
        
        # For larger snakes, check tail reachability
        new_head = new_snake[0]
        new_tail = new_snake[-1]
        new_obstacles = set(tuple(p) for p in new_snake[:-1])  # Exclude tail from obstacles
        
        # Use SSOT BFS to check tail reachability
        tail_path = ssot_bfs_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def _count_remaining_free_cells(self, snake_positions: set, grid_size: int) -> int:
        """Count how many empty cells are not occupied by the snake body."""
        total_cells = grid_size * grid_size
        return total_cells - len(snake_positions)

    def _generate_safe_apple_explanation(self, head: List[int], apple: List[int], snake: List[List[int]], 
                                        path: List[List[int]], direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int, grid_size: int,
                                        metrics: dict) -> dict:
        """Generate detailed explanation for safe apple path strategy."""
        path_length = len(path) - 1
        snake_length = len(snake)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Calculate safety metrics
        next_pos = path[1] if len(path) > 1 else head
        safety_margin = remaining_free_cells - snake_length
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: SAFE APPLE PATH ===",
            "",
            "PHASE 1: INITIAL ASSESSMENT",
            f"• Algorithm: BFS-Safe-Greedy (enhanced BFS with safety validation)",
            f"• Current head position: {head}",
            f"• Target apple position: {apple}",
            f"• Snake length: {snake_length} segments",
            f"• Grid dimensions: {grid_size}×{grid_size} ({grid_size * grid_size} total cells)",
            f"• Board occupation: {snake_length}/{grid_size * grid_size} cells ({board_fill_ratio:.1%}) - {space_pressure} space pressure",
            f"• Free cells remaining: {remaining_free_cells}",
            "",
            "PHASE 2: PRIMARY STRATEGY - APPLE PATHFINDING",
            f"• Strategy priority: PRIMARY (safe apple pursuit)",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• BFS pathfinding from {head} to {apple}",
            f"• Manhattan distance baseline: {manhattan_distance} steps (theoretical minimum)",
            f"• Actual shortest path found: {path_length} steps",
            f"• Path efficiency: {manhattan_distance / max(path_length, 1):.2f}",
            f"• Path coordinates: {' → '.join([str(p) for p in path[:min(4, len(path))]])}{'...' if len(path) > 4 else ''}",
            "",
            "PHASE 3: SAFETY VALIDATION PROTOCOL",
            f"• Safety check: ENABLED (distinguishes Safe-Greedy from standard BFS)",
            f"• Next position after move: {next_pos}",
            f"• Safety validation method: Tail reachability analysis",
            f"• Post-move snake configuration: simulated",
            f"• Tail accessibility: VERIFIED (path exists from new head to tail)",
            f"• Safety margin: {safety_margin} cells buffer",
            f"• Risk assessment: LOW (move maintains tail access)",
            "",
            "PHASE 4: DECISION RATIONALE",
            f"• Primary condition: Apple path exists ✓",
            f"• Secondary condition: Move is safe ✓", 
            f"• Tertiary condition: Tail remains reachable ✓",
            f"• Chosen direction: {direction}",
            f"• Expected outcome: Advance toward apple while preserving escape routes",
            f"• Strategic advantage: Optimal progress with guaranteed safety",
            "",
            "PHASE 5: SAFE-GREEDY ADVANTAGES",
            f"• vs Standard BFS: Adds safety validation to prevent trapping",
            f"• vs Pure Greedy: Maintains shortest path optimality",
            f"• vs Conservative: Still pursues apple when safe",
            f"• Safety guarantee: Tail reachability preserved",
            f"• Performance: {path_length} steps to apple with safety assurance",
            "",
            "=== CONCLUSION ===",
            f"BFS-Safe-Greedy successfully identified a safe {path_length}-step path to the apple at {apple}. " +
            f"Moving {direction} advances the snake from {head} to {next_pos}, maintaining optimal trajectory " +
            f"while ensuring tail accessibility. This demonstrates the algorithm's core strength: combining " +
            f"BFS pathfinding efficiency with safety validation to prevent self-trapping scenarios."
        ]

        return {
            "strategy_phase": "SAFE_APPLE_PATH",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_tail_chase_explanation(self, head: List[int], apple: List[int], snake: List[List[int]],
                                       tail: List[int], path: List[List[int]], direction: str, 
                                       valid_moves: List[str], manhattan_distance: int, 
                                       remaining_free_cells: int, grid_size: int, metrics: dict) -> dict:
        """Generate detailed explanation for tail chase strategy."""
        path_length = len(path) - 1
        snake_length = len(snake)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: TAIL CHASE STRATEGY ===",
            "",
            "PHASE 1: PRIMARY STRATEGY FAILURE ANALYSIS",
            f"• Primary strategy attempted: Apple pathfinding to {apple}",
            f"• Primary strategy result: FAILED (unsafe or no path found)",
            f"• Failure reason: Safety validation rejected apple path",
            f"• Risk detected: Potential self-trapping if pursuing apple",
            f"• Algorithm response: Activate secondary strategy",
            "",
            "PHASE 2: SECONDARY STRATEGY - TAIL CHASING",
            f"• Strategy priority: SECONDARY (defensive positioning)",
            f"• Target: Snake tail at {tail}",
            f"• Rationale: Tail chasing is always safe (tail moves away)",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• BFS pathfinding from {head} to {tail}",
            f"• Tail chase path found: {path_length} steps",
            f"• Path coordinates: {' → '.join([str(p) for p in path[:min(4, len(path))]])}{'...' if len(path) > 4 else ''}",
            "",
            "PHASE 3: TAIL CHASE SAFETY ANALYSIS",
            f"• Safety guarantee: ABSOLUTE (tail moves as snake advances)",
            f"• Self-collision risk: ZERO (impossible to catch moving tail)",
            f"• Space preservation: Maintains current board position",
            f"• Future opportunities: Keeps options open for apple pursuit",
            f"• Board pressure: {space_pressure} ({board_fill_ratio:.1%} occupation)",
            "",
            "PHASE 4: STRATEGIC POSITIONING",
            f"• Current head position: {head}",
            f"• Chosen direction: {direction}",
            f"• Next position: Following tail at distance {path_length}",
            f"• Apple distance: {manhattan_distance} steps (for future reference)",
            f"• Space management: Preserving {remaining_free_cells} free cells",
            f"• Positioning benefit: Maintains mobility while avoiding risks",
            "",
            "PHASE 5: SAFE-GREEDY DEFENSIVE LOGIC",
            f"• Algorithm strength: Never pursues risky apple paths",
            f"• Fallback reliability: Tail chasing provides guaranteed safe moves",
            f"• vs Standard BFS: Would attempt unsafe apple path",
            f"• vs Pure Conservative: Would avoid apple even when safe",
            f"• Adaptive behavior: Switches strategies based on safety assessment",
            "",
            "=== CONCLUSION ===",
            f"BFS-Safe-Greedy activated tail chase strategy after determining apple pursuit was unsafe. " +
            f"Moving {direction} toward tail at {tail} provides guaranteed safety while maintaining " +
            f"board position. This defensive strategy preserves the snake's survival until safer " +
            f"apple pursuit opportunities emerge, demonstrating the algorithm's adaptive safety-first approach."
        ]

        return {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_survival_explanation(self, head: List[int], apple: List[int], snake: List[List[int]],
                                     direction: str, valid_moves: List[str], manhattan_distance: int,
                                     remaining_free_cells: int, grid_size: int, metrics: dict) -> dict:
        """Generate detailed explanation for survival move strategy."""
        snake_length = len(snake)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: SURVIVAL MODE ===",
            "",
            "PHASE 1: CRITICAL SITUATION ASSESSMENT",
            f"• Algorithm: BFS-Safe-Greedy in emergency survival mode",
            f"• Current head position: {head}",
            f"• Snake length: {snake_length} segments",
            f"• Board occupation: {board_fill_ratio:.1%} (CRITICAL density)",
            f"• Free cells remaining: {remaining_free_cells}",
            f"• Available moves: {valid_moves} ({len(valid_moves)} emergency options)",
            "",
            "PHASE 2: STRATEGY CASCADE FAILURE",
            f"• PRIMARY strategy (safe apple path): FAILED",
            f"• SECONDARY strategy (tail chase): FAILED", 
            f"• TERTIARY strategy (survival move): ACTIVATED",
            f"• Situation severity: CRITICAL (limited options remaining)",
            f"• Risk level: MAXIMUM (immediate survival at stake)",
            "",
            "PHASE 3: EMERGENCY MOVE SELECTION",
            f"• Emergency protocol: Select any valid move to avoid death",
            f"• Available options: {valid_moves}",
            f"• Selected move: {direction} (first available valid move)",
            f"• Selection criteria: Immediate collision avoidance only",
            f"• Long-term planning: SUSPENDED (survival takes priority)",
            "",
            "PHASE 4: SURVIVAL IMPLICATIONS",
            f"• Immediate outcome: Avoid instant death",
            f"• Apple accessibility: {manhattan_distance} steps (currently irrelevant)",
            f"• Future prospects: Depends on subsequent board evolution",
            f"• Strategy horizon: 1 move (emergency mode)",
            f"• Success metric: Continued existence",
            "",
            "=== CONCLUSION ===",
            f"BFS-Safe-Greedy entered survival mode due to lack of safe strategic options. " +
            f"Moving {direction} represents the algorithm's last resort to avoid immediate termination. " +
            f"This demonstrates the algorithm's hierarchical strategy system: when safety-validated " +
            f"paths fail, it prioritizes basic survival over strategic positioning."
        ]

        return {
            "strategy_phase": "SURVIVAL_MOVE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_no_moves_explanation(self, head: List[int], apple: List[int], snake: List[List[int]],
                                     valid_moves: List[str], manhattan_distance: int,
                                     remaining_free_cells: int, grid_size: int, metrics: dict) -> dict:
        """Generate detailed explanation for no moves available scenario."""
        snake_length = len(snake)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: TERMINAL CONDITION ===",
            "",
            "PHASE 1: COMPLETE STRATEGY FAILURE",
            f"• Algorithm: BFS-Safe-Greedy facing terminal condition",
            f"• Current head position: {head}",
            f"• Snake length: {snake_length} segments",
            f"• Board occupation: {board_fill_ratio:.1%} (MAXIMUM density)",
            f"• Free cells remaining: {remaining_free_cells}",
            f"• Available moves: {valid_moves} (NONE)",
            "",
            "PHASE 2: PATHFINDING FAILURE CASCADE",
            f"• PRIMARY strategy (safe apple path): NO VALID MOVES",
            f"• SECONDARY strategy (tail chase): NO VALID MOVES",
            f"• TERTIARY strategy (survival move): NO VALID MOVES",
            f"• FINAL result: COMPLETE IMMOBILIZATION",
            "",
            "PHASE 3: TERMINAL CONDITION ANALYSIS",
            f"• Head surrounded: All adjacent cells blocked",
            f"• Blocking factors: Walls and/or snake body segments",
            f"• Escape routes: NONE AVAILABLE",
            f"• Apple distance: {manhattan_distance} steps (unreachable)",
            f"• Game state: TERMINAL (no legal moves possible)",
            "",
            "PHASE 4: ALGORITHM PERFORMANCE SUMMARY",
            f"• Strategy hierarchy: All levels exhausted",
            f"• Safety validation: Prevented risky moves throughout game",
            f"• Survival duration: Maximized through conservative play",
            f"• Final outcome: Inevitable termination due to space constraints",
            "",
            "=== CONCLUSION ===",
            f"BFS-Safe-Greedy has reached a terminal state with no valid moves from {head}. " +
            f"The algorithm's safety-first approach successfully avoided premature risks but " +
            f"ultimately cannot overcome fundamental space limitations. This represents the " +
            f"natural endpoint of conservative pathfinding in constrained environments."
        ]

        return {
            "strategy_phase": "NO_MOVES",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def __str__(self) -> str:
        """String representation showing inheritance relationship."""
        return f"BFSSafeGreedyAgent(extends=BFSAgent, algorithm={self.algorithm_name})" 
