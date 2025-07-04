from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy Agent - Enhanced BFS with Safety Validation for Snake Game v0.04
---------------------------------------------------------------------------------

This module implements an enhanced BFS agent that adds safety validation
to prevent the snake from getting trapped in dead-ends.

v0.04 Enhancement: Generates natural language explanations for each move
to create rich datasets for LLM fine-tuning.

Strategy:
1. Use BFS to find shortest path to apple
2. Validate that following this path won't trap the snake
3. If path is unsafe, use greedy movement toward apple
4. Generate detailed explanations for strategy selection

Design Patterns:
- Strategy Pattern: BFS with safety validation strategy
- Template Method: Extends basic BFS with safety checks
- Protocol Pattern: Implements BaseAgent interface for compatibility
"""

from typing import List, Tuple, TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSSafeGreedyAgent(BFSAgent):
    """
    BFS Safe Greedy Agent: Enhanced BFS with safety validation.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses basic BFS pathfinding)
    - Overrides get_move() to add safety validation
    - Extends with tail-chasing fallback behavior
    - Demonstrates evolution from basic to enhanced algorithm
    
    Algorithm Enhancement:
    1. Find shortest path to apple using inherited BFS
    2. Validate path safety (can snake reach tail afterward?)
    3. If safe, follow apple path
    4. If unsafe, chase tail instead (always safe)
    5. Last resort: any non-crashing move
    
    This shows how software evolves: start with working solution (BFS),
    then enhance with additional features (safety validation).
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

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using safe BFS pathfinding (simplified interface).
        
        Enhancement over parent BFS:
        - Adds safety validation before following apple path
        - Implements tail-chasing fallback strategy
        - Provides last-resort non-crashing move selection
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move
        
    def get_move_with_explanation(self, game: "HeuristicGameLogic") -> Tuple[str, dict]:
        """
        Compute next move *and* a structured explanation object.
        
        SSOT Compliance: This method now generates explanations solely from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # -------------------------------------------------- Common prelude
        # Get positions from game state snapshot to ensure consistency with dataset generator
        game_state = game.get_state_snapshot()
        head = tuple(game_state["head_position"])
        apple = tuple(game_state["apple_position"])
        snake = [tuple(seg) for seg in game_state["snake_positions"]]
        grid_size = game_state["grid_size"]
        obstacles = set(snake[:-1])  # Tail can vacate → not an obstacle

        # Helper metrics that are independent of strategy branch
        manhattan_distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        valid_moves = self._get_valid_moves(head, set(snake), grid_size)
        remaining_free_cells = self._count_remaining_free_cells(set(snake), grid_size)

        # ------------------------------------------------ Apple path first
        path_to_apple = self._bfs_pathfind(head, apple, obstacles, grid_size)
        apple_path_safe = False
        if path_to_apple and len(path_to_apple) > 1:
            apple_path_safe = self._path_is_safe(path_to_apple, snake, apple, grid_size)

        if path_to_apple and len(path_to_apple) > 1 and apple_path_safe:
            # Use the safe apple path
            next_pos = path_to_apple[1]
            direction = position_to_direction(head, next_pos)
            if direction not in valid_moves:
                explanation_dict = {
                    "strategy_phase": "INVALID_MOVE",
                    "metrics": {},
                    "explanation_steps": [f"BFS-SAFE-GREEDY computed move '{direction}' which is not valid. Returning NO_PATH_FOUND."],
                }
                return "NO_PATH_FOUND", explanation_dict
            strategy_phase = "APPLE_PATH"
            fallback_used = False
            apple_path_length = len(path_to_apple) - 1
            tail_path_length = None

            # Explanatory chain-of-thought (cot)
            explanation_steps = [
                f"Step 1: Evaluate immediate valid moves: {valid_moves}.",
                f"Step 2: Use BFS to find shortest path to apple at {apple}; path length found: {apple_path_length}.",
                "Step 3: Validate safety by checking if snake can still reach its tail after eating apple — result: safe.",
                "Step 4: Confirm strategy phase as 'APPLE_PATH' since apple path is safe.",
                f"Step 5: Select first move in safe apple path, which is '{direction}'.",
                "",  # Spacer line
                "Conclusion:",
                f"By moving '{direction}', the snake safely advances toward the apple along the shortest verified route, ensuring survival and maximizing future reward."
            ]

            metrics = {
                "manhattan_distance": int(manhattan_distance),
                "apple_path_length": int(apple_path_length),
                "tail_path_length": tail_path_length,
                "valid_moves": valid_moves,
                "apple_path_safe": apple_path_safe,
                "fallback_used": fallback_used,
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": int(len(snake)),
                "grid_size": int(grid_size),
                "remaining_free_cells": int(remaining_free_cells),
            }

            explanation_dict = {
                "strategy_phase": strategy_phase,
                "metrics": metrics,
                "explanation_steps": explanation_steps,
            }

            return direction, explanation_dict

        # ---------------------------------------- Fallback – tail chasing
        tail = snake[-1]
        path_to_tail = self._bfs_pathfind(head, tail, obstacles, grid_size)
        if path_to_tail and len(path_to_tail) > 1:
            next_pos = path_to_tail[1]
            direction = position_to_direction(head, next_pos)
            strategy_phase = "TAIL_CHASE"
            fallback_used = True
            tail_path_length = len(path_to_tail) - 1
            apple_path_length = len(path_to_apple) - 1 if path_to_apple else None

            explanation_steps = [
                f"Step 1: Evaluate immediate valid moves: {valid_moves}.",
                f"Step 2: Direct apple path deemed unsafe (apple_path_safe={apple_path_safe}).",
                "Step 3: Switch to tail-chasing fallback strategy to guarantee survival.",
                f"Step 4: BFS path to tail at {tail} has length {tail_path_length}.",
                f"Step 5: Select first move '{direction}' to move towards tail.",
                "",  # Spacer line
                "Conclusion:",
                f"By moving '{direction}', the snake prioritizes survival by following its tail, ensuring it can continue playing and wait for better opportunities."
            ]

            metrics = {
                "manhattan_distance": int(manhattan_distance),
                "apple_path_length": int(apple_path_length),
                "tail_path_length": tail_path_length,
                "valid_moves": valid_moves,
                "apple_path_safe": apple_path_safe,
                "fallback_used": fallback_used,
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": int(len(snake)),
                "grid_size": int(grid_size),
                "remaining_free_cells": int(remaining_free_cells),
            }

            explanation_dict = {
                "strategy_phase": strategy_phase,
                "metrics": metrics,
                "explanation_steps": explanation_steps,
            }

            return direction, explanation_dict

        # ----------------------------------------- Last resort scenario
        last_resort_move = self._get_safe_move(head, obstacles, grid_size)
        strategy_phase = "LAST_RESORT"
        fallback_used = True

        explanation_steps = [
            f"Step 1: Evaluate immediate valid moves: {valid_moves}.",
            "Step 2: No safe path to apple or tail could be found.",
            f"Step 3: Select any non-crashing move as last resort: '{last_resort_move}'.",
            "",  # Spacer line
            "Conclusion:",
            f"By moving '{last_resort_move}', the snake avoids immediate collision, maximizing its chance to survive and seek future opportunities."
        ]

        metrics = {
            "manhattan_distance": int(manhattan_distance),
            "apple_path_length": None,
            "tail_path_length": None,
            "valid_moves": valid_moves,
            "apple_path_safe": False,
            "fallback_used": fallback_used,
            "final_chosen_direction": last_resort_move,
            "head_position": list(head),
            "apple_position": list(apple),
            "snake_length": int(len(snake)),
            "grid_size": int(grid_size),
            "remaining_free_cells": int(remaining_free_cells),
        }

        explanation_dict = {
            "strategy_phase": strategy_phase,
            "metrics": metrics,
            "explanation_steps": explanation_steps,
        }

        return last_resort_move, explanation_dict

    def _path_is_safe(
        self,
        path: List[Tuple[int, int]],
        snake: List[Tuple[int, int]],
        apple: Tuple[int, int],
        grid_size: int
    ) -> bool:
        """
        Safety enhancement: Validate path by simulating execution.
        
        This is the key enhancement over basic BFS - we simulate
        following the path and check if the snake can still reach
        its tail afterward, avoiding getting trapped.
        
        Args:
            path: Proposed path to apple
            snake: Current snake body
            apple: Apple position
            grid_size: Size of game grid
            
        Returns:
            True if path is safe (tail reachable), False otherwise
        """
        # Simulate following the path
        virtual_snake = list(snake)
        
        for step in path[1:]:  # Skip current head position
            virtual_snake.insert(0, step)  # Move head
            
            if step == apple:
                # Apple eaten - snake grows, keep tail
                break
            
            # No apple - tail moves forward
            virtual_snake.pop()
        
        # Check if new head can reach new tail
        new_head = virtual_snake[0]
        new_tail = virtual_snake[-1]
        new_obstacles = set(virtual_snake[:-1])  # Exclude tail
        
        # Use inherited BFS to check tail reachability
        tail_path = self._bfs_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def _get_safe_move(
        self, 
        head: Tuple[int, int], 
        obstacles: set, 
        grid_size: int
    ) -> str:
        """
        Last resort: find any non-crashing move.
        
        Args:
            head: Current head position
            obstacles: Set of obstacle positions
            grid_size: Size of game grid
            
        Returns:
            Safe direction or "NO_PATH_FOUND"
        """
        for direction, (dx, dy) in DIRECTIONS.items():
            next_pos = (head[0] + dx, head[1] + dy)
            
            # Check bounds
            if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                continue
                
            # Check obstacles
            if next_pos not in obstacles:
                return direction
                
        return "NO_PATH_FOUND"

    def __str__(self) -> str:
        """String representation showing inheritance relationship."""
        return f"BFSSafeGreedyAgent(extends=BFSAgent, algorithm={self.algorithm_name})" 
