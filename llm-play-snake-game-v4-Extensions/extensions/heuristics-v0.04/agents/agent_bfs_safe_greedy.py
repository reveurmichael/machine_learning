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
                
                explanation_dict = {
                    "strategy_phase": "SAFE_APPLE_PATH",
                    "metrics": metrics,
                    "explanation_steps": [
                        f"Found safe path to apple with length {len(path_to_apple) - 1}",
                        f"Safety check passed: tail reachable after move '{direction}'",
                        f"Moving {direction} toward apple at {apple}"
                    ]
                }
                
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
            
            explanation_dict = {
                "strategy_phase": "TAIL_CHASE",
                "metrics": metrics,
                "explanation_steps": [
                    f"Apple path unsafe or not found",
                    f"Following tail-chase strategy: {direction}",
                    f"Tail-chasing is always safe"
                ]
            }
            
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
            
            explanation_dict = {
                "strategy_phase": "SURVIVAL_MOVE",
                "metrics": metrics,
                "explanation_steps": [
                    f"No safe paths found to apple or tail",
                    f"Using survival move: '{direction}' to avoid immediate death"
                ]
            }
            
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
            
            explanation_dict = {
                "strategy_phase": "NO_MOVES",
                "metrics": metrics,
                "explanation_steps": [
                    f"No valid moves available from {head}",
                    f"Snake is trapped"
                ]
            }
            
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

    def __str__(self) -> str:
        """String representation showing inheritance relationship."""
        return f"BFSSafeGreedyAgent(extends=BFSAgent, algorithm={self.algorithm_name})" 
