"""
BFS Safe Greedy Agent - Enhanced BFS with Safety Checks
==========================================================

This module implements an enhanced BFS agent that prioritizes safety.
It finds the shortest path to the apple but ensures the snake can still
reach its tail afterward. Falls back to tail-chasing when no safe path exists.

The agent implements the SnakeAgent protocol, making it compatible with
the existing game engine infrastructure.

Design Patterns:
- Strategy Pattern: Enhanced BFS with safety checks
- Template Method: BFS with safety validation overlay
- Fallback Pattern: Tail-chasing when apple path is unsafe
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
import numpy as np

from config.game_constants import DIRECTIONS, VALID_MOVES
from utils.moves_utils import position_to_direction
from core.game_agents import SnakeAgent

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSSafeGreedyAgent:
    """
    BFS Safe Greedy Agent: Enhanced BFS with safety validation.
    
    Algorithm:
    1. Find shortest path to apple using BFS
    2. Validate path safety (can snake reach tail afterward?)
    3. If safe, follow apple path
    4. If unsafe, chase tail instead (always safe)
    5. Last resort: any non-crashing move
    
    This agent demonstrates evolution from basic BFS by adding
    strategic safety considerations.
    """

    def __init__(self) -> None:
        self.algorithm_name = "BFS-SAFE-GREEDY"
        self.name = "BFS Safe Greedy"
        self.description = (
            "Enhanced BFS with safety validation. Finds shortest path to apple "
            "but ensures snake can reach tail afterward. Falls back to tail-chasing "
            "when apple path is unsafe. Evolution from basic BFS."
        )

    # ------------------------------------------------------------------
    # Public API expected by game manager
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using safe BFS pathfinding.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        head = tuple(game.head_position)
        apple = tuple(game.apple_position)
        snake = [tuple(seg) for seg in game.snake_positions]
        size = game.grid_size

        def in_bounds(pos: Tuple[int, int]) -> bool:
            return 0 <= pos[0] < size and 0 <= pos[1] < size

        obstacles = set(snake[:-1])  # tail can vacate

        # --------------------------------------------------------------
        # 1. Try shortest safe path to apple
        # --------------------------------------------------------------
        path_to_apple = self._bfs(head, apple, obstacles, in_bounds)
        if path_to_apple and len(path_to_apple) > 1:
            if self._path_is_safe(path_to_apple, snake, apple, in_bounds):
                return position_to_direction(head, path_to_apple[1])

        # --------------------------------------------------------------
        # 2. Chase tail (always safe)
        # --------------------------------------------------------------
        tail = snake[-1]
        path_to_tail = self._bfs(head, tail, obstacles, in_bounds)
        if path_to_tail and len(path_to_tail) > 1:
            return position_to_direction(head, path_to_tail[1])

        # --------------------------------------------------------------
        # 3. Last resort: any non-crashing move
        # --------------------------------------------------------------
        for dir_name, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if in_bounds(nxt) and nxt not in obstacles:
                return dir_name
        return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _path_is_safe(
        self,
        path: List[Tuple[int, int]],
        snake: List[Tuple[int, int]],
        apple: Tuple[int, int],
        in_bounds,
    ) -> bool:
        """
        Validate path safety by simulating execution.
        
        Simulates following the path and checks if the snake can
        still reach its tail afterward (avoiding getting trapped).
        
        Args:
            path: Proposed path to apple
            snake: Current snake body
            apple: Apple position
            in_bounds: Boundary check function
            
        Returns:
            True if path is safe (tail reachable), False otherwise
        """
        virtual = list(snake)
        for step in path[1:]:
            virtual.insert(0, step)
            if step == apple:
                break  # grow – keep tail
            virtual.pop()
        new_head, new_tail = virtual[0], virtual[-1]
        return bool(self._bfs(new_head, new_tail, set(virtual[:-1]), in_bounds))

    # ------------------------------------------------------------------
    # BFS utility
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: set[Tuple[int, int]],
        in_bounds,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search pathfinding.
        
        Args:
            start: Starting position
            goal: Goal position  
            obstacles: Set of obstacle positions
            in_bounds: Boundary validation function
            
        Returns:
            Path as list of positions, or None if no path exists
        """
        if start == goal:
            return [start]
        queue: deque[List[Tuple[int, int]]] = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            pos = path[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (pos[0] + dx, pos[1] + dy)
                if not in_bounds(nxt) or nxt in obstacles or nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == goal:
                    return new_path
                visited.add(nxt)
                queue.append(new_path)
        return None

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSSafeGreedyAgent(algorithm={self.algorithm_name})" 