"""
BFS Agent - Breadth-First Search pathfinding for Snake Game
==========================================================

This module implements a simple BFS (Breadth-First Search) agent that finds
the shortest path to the apple while avoiding walls and its own body.

The agent implements the SnakeAgent protocol, making it compatible with
the existing game engine infrastructure.

Design Patterns:
- Strategy Pattern: BFS algorithm encapsulated as a strategy
- Protocol Pattern: Implements SnakeAgent interface for compatibility
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
import numpy as np

from config.game_constants import DIRECTIONS, VALID_MOVES
from core.game_agents import SnakeAgent

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSAgent:
    """Safe-Greedy BFS agent: shortest safe path to apple, else chase tail."""

    def __init__(self) -> None:
        self.algorithm_name = "BFS-SafeGreedy"
        self.name = "BFS-SafeGreedy"
        self.description = (
            "BFS shortest-path to apple with tail-reachability safety; "
            "tail-chasing fallback when no safe apple path exists."
        )

    # ------------------------------------------------------------------
    # Public API expected by game manager
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str | None:  # noqa: D401
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
                return self._direction_between(head, path_to_apple[1])

        # --------------------------------------------------------------
        # 2. Chase tail (always safe)
        # --------------------------------------------------------------
        tail = snake[-1]
        path_to_tail = self._bfs(head, tail, obstacles, in_bounds)
        if path_to_tail and len(path_to_tail) > 1:
            return self._direction_between(head, path_to_tail[1])

        # --------------------------------------------------------------
        # 3. Last-ditch: any non-crashing move
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
        """Simulate *path*; ensure head can reach tail afterward."""
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

    # ------------------------------------------------------------------
    # Direction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _direction_between(frm: Tuple[int, int], to: Tuple[int, int]) -> str:
        dx, dy = to[0] - frm[0], to[1] - frm[1]
        return {
            (1, 0): "RIGHT",
            (-1, 0): "LEFT",
            (0, 1): "UP",
            (0, -1): "DOWN",
        }.get((dx, dy), "NO_PATH_FOUND")

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})" 