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

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

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
        
    def get_move_with_explanation(self, game: "HeuristicGameLogic") -> Tuple[str, str]:
        """
        Get next move using safe BFS pathfinding with detailed explanation.
        
        v0.04 Enhancement: Returns both move and natural language explanation
        for LLM fine-tuning dataset generation.
        
        Enhancement over parent BFS:
        - Adds safety validation before following apple path
        - Implements tail-chasing fallback strategy
        - Provides last-resort non-crashing move selection
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (direction_string, explanation_string)
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            snake = [tuple(seg) for seg in game.snake_positions]
            grid_size = game.grid_size
            obstacles = set(snake[:-1])  # Exclude tail (can vacate)

            # ---------------------
            # 1. Try shortest safe path to apple (using inherited BFS)
            # ---------------------
            path_to_apple = self._bfs_pathfind(head, apple, obstacles, grid_size)
            
            if path_to_apple and len(path_to_apple) > 1:
                # Safety enhancement: validate path before using it
                if self._path_is_safe(path_to_apple, snake, apple, grid_size):
                    next_pos = path_to_apple[1]
                    direction = position_to_direction(head, next_pos)
                    path_length = len(path_to_apple) - 1
                    explanation = (
                        f"BFS Safe Greedy found a safe shortest path of length {path_length} to apple at {apple}. "
                        f"Moving {direction} from {head} to {next_pos}. "
                        f"Path safety verified: snake can reach its tail after collecting the apple, avoiding traps."
                    )
                    return direction, explanation

            # ---------------------
            # 2. Fallback: Chase tail (always safe)
            # ---------------------
            tail = snake[-1]
            path_to_tail = self._bfs_pathfind(head, tail, obstacles, grid_size)
            
            if path_to_tail and len(path_to_tail) > 1:
                next_pos = path_to_tail[1]
                direction = position_to_direction(head, next_pos)
                tail_path_length = len(path_to_tail) - 1
                explanation = (
                    f"BFS Safe Greedy: Direct path to apple at {apple} deemed unsafe. "
                    f"Falling back to tail-chasing strategy. Moving {direction} toward tail at {tail} "
                    f"(distance: {tail_path_length}). This guarantees survival while waiting for safer apple opportunity."
                )
                return direction, explanation

            # ---------------------
            # 3. Last resort: any non-crashing move
            # ---------------------
            last_resort_move = self._get_safe_move(head, obstacles, grid_size)
            explanation = (
                f"BFS Safe Greedy: No safe path to apple at {apple} and no path to tail. "
                f"Using last resort move {last_resort_move} to avoid immediate collision. "
                f"This indicates a very constrained game state."
            )
            return last_resort_move, explanation
            
        except Exception as e:
            explanation = f"BFS Safe Greedy Agent encountered an error: {str(e)}"
            print_error(f"BFS Safe Greedy Agent error: {e}")
            return "NO_PATH_FOUND", explanation

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