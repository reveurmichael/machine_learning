from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, Set

from config.game_constants import DIRECTIONS, VALID_MOVES
from utils.moves_utils import position_to_direction
from core.game_agents import SnakeAgent


# Helper: generate simple Hamiltonian cycle (for even grids)
def generate_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    """Generate a simple Hamiltonian cycle using boustrophedon pattern."""
    cycle: List[Tuple[int, int]] = []
    for y in range(grid_size):
        row = (
            list(range(grid_size)) if y % 2 == 0 else list(range(grid_size - 1, -1, -1))
        )
        for x in row:
            cycle.append((x, y))
    return cycle


class BFSHamiltonianAgent(SnakeAgent):
    """
    BFS Hamiltonian Hybrid Agent: Enhanced BFS with Hamiltonian fallback.
    
    This agent combines the efficiency of BFS pathfinding with the safety
    guarantee of Hamiltonian cycles. It tries BFS first, then falls back
    to Hamiltonian cycle when no safe path exists.
    
    Evolution from BFS Safe Greedy: Adds Hamiltonian cycle as ultimate fallback.
    """

    def __init__(self) -> None:
        self.algorithm_name = "BFS-HAMILTONIAN"
        self.name = "BFS Hamiltonian Hybrid"
        self.description = (
            "Hybrid agent combining BFS pathfinding with Hamiltonian cycle fallback. "
            "Tries shortest safe path to apple, then tail-chasing, then intelligent "
            "Hamiltonian shortcuts, ensuring the snake never gets trapped."
        )
        self.hamiltonian: List[Tuple[int, int]] = []
        self.grid_size: Optional[int] = None

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """Get next move using BFS with Hamiltonian fallback."""
        head = tuple(game.head_position)
        apple = tuple(game.apple_position)
        snake = [tuple(seg) for seg in game.snake_positions]
        size = game.grid_size

        # Initialize Hamiltonian cycle once per grid size
        if self.grid_size != size:
            self.grid_size = size
            self.hamiltonian = generate_hamiltonian_cycle(size)

        def in_bounds(pos: Tuple[int, int]) -> bool:
            return 0 <= pos[0] < size and 0 <= pos[1] < size

        obstacles = set(snake[:-1])  # tail can vacate

        # 1. Try shortest safe path to apple
        path_to_apple = self._bfs(head, apple, obstacles, in_bounds)
        if path_to_apple and len(path_to_apple) > 1:
            if self._path_is_safe(path_to_apple, snake, apple, in_bounds):
                return position_to_direction(head, path_to_apple[1])

        # 2. Tail chase fallback
        tail = snake[-1]
        path_to_tail = self._bfs(head, tail, obstacles, in_bounds)
        if path_to_tail and len(path_to_tail) > 1:
            return position_to_direction(head, path_to_tail[1])

        # 3. Intelligent Hamiltonian shortcut: jump ahead toward apple if segment is safe
        move = self._hamiltonian_shortcut(head, apple, snake, in_bounds)
        if move:
            return move

        # 4. Hamiltonian fallback (full cycle)
        move = self._hamiltonian_move(head, obstacles)
        if move:
            return move

        # 5. Last-ditch: any non-crashing move
        for dir_name, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if in_bounds(nxt) and nxt not in obstacles:
                return dir_name

        return "NO_PATH_FOUND"

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
        """
        virtual = list(snake)
        for step in path[1:]:
            virtual.insert(0, step)
            if step == apple:
                break  # grow – keep tail
            virtual.pop()
        new_head, new_tail = virtual[0], virtual[-1]
        return bool(self._bfs(new_head, new_tail, set(virtual[:-1]), in_bounds))

    def _hamiltonian_move(
        self, head: Tuple[int, int], obstacles: Set[Tuple[int, int]]
    ) -> Optional[str]:
        """Move to next position in Hamiltonian cycle."""
        if head not in self.hamiltonian:
            return None
        idx = self.hamiltonian.index(head)
        next_cell = self.hamiltonian[(idx + 1) % len(self.hamiltonian)]
        if next_cell not in obstacles:
            return position_to_direction(head, next_cell)
        return None

    def _hamiltonian_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        snake: List[Tuple[int, int]],
        in_bounds,
    ) -> Optional[str]:
        """Try intelligent Hamiltonian shortcut toward apple."""
        # Determine indices on cycle
        if head not in self.hamiltonian or apple not in self.hamiltonian:
            return None
        idx_h = self.hamiltonian.index(head)
        idx_a = self.hamiltonian.index(apple)
        # Direction: forward cycle distance
        dist = (idx_a - idx_h) % len(self.hamiltonian)
        if dist == 0 or dist > len(self.hamiltonian) // 2:
            return None
        # Build segment from head to apple along cycle
        segment = [
            self.hamiltonian[(idx_h + i) % len(self.hamiltonian)]
            for i in range(dist + 1)
        ]
        # Simulate safety of segment
        if self._segment_is_safe(segment, snake, apple, in_bounds):
            return position_to_direction(head, segment[1])
        return None

    def _segment_is_safe(
        self,
        segment: List[Tuple[int, int]],
        snake: List[Tuple[int, int]],
        apple: Tuple[int, int],
        in_bounds,
    ) -> bool:
        """Check if following a Hamiltonian segment is safe."""
        virtual = list(snake)
        for step in segment[1:]:
            virtual.insert(0, step)
            if step == apple:
                break
            virtual.pop()
        new_head, new_tail = virtual[0], virtual[-1]
        return bool(self._bfs(new_head, new_tail, set(virtual[:-1]), in_bounds))

    @staticmethod
    def _bfs(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        in_bounds,
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding implementation."""
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
        return f"BFSHamiltonianAgent(algorithm={self.algorithm_name})"
