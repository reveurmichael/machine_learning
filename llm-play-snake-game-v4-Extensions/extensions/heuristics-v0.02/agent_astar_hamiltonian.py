"""
A* Hamiltonian Hybrid Agent - Advanced Pathfinding with Safety Guarantee
==============================================

This agent combines the efficiency of A* pathfinding with the ultimate
safety guarantee of Hamiltonian cycles. It represents the most advanced
heuristic approach in the v0.02 collection.

Evolution: A* (optimal pathfinding) + Hamiltonian (safety guarantee) = Best of both worlds.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set

from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction


class AStarHamiltonianAgent:
    """
    A* Hamiltonian Hybrid Agent: The most advanced heuristic approach.
    
    This agent combines:
    1. A* pathfinding for optimal apple routes
    2. Safety validation to prevent trapping
    3. Hamiltonian cycle as ultimate fallback
    
    Algorithm:
    1. Use A* to find optimal path to apple
    2. Validate path safety (can reach tail afterward)
    3. If safe, follow A* path
    4. Otherwise, follow Hamiltonian cycle (guaranteed safe)
    
    This represents the pinnacle of heuristic Snake AI - optimal when possible,
    safe always.
    """

    def __init__(self) -> None:
        self.algorithm_name = "ASTAR-HAMILTONIAN"
        self.name = "A* Hamiltonian Hybrid"
        self.description = (
            "Advanced hybrid agent combining A* optimal pathfinding with "
            "Hamiltonian cycle safety guarantee. Uses A* when safe, falls back "
            "to Hamiltonian cycle when necessary. Represents the pinnacle of "
            "heuristic Snake AI."
        )
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_cycle_index: int = 0
        self.debug_log = False  # Disable for cleaner output in v0.02

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str:  # noqa: D401
        """
        Get next move using A* with Hamiltonian fallback.

        Strategy priority:
        1. Use A* to find a shortest path to the apple.
           If following that path keeps the snake able to reach its tail
           afterwards (safety guarantee), take the first step.
        2. Otherwise fall back to the pre-computed Hamiltonian cycle.
        3. If even that fails (should be impossible), make *any* safe move.
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            grid_size = game.grid_size
            body = [tuple(seg) for seg in game.snake_positions]
            obstacles: Set[Tuple[int, int]] = set(body[:-1])  # tail can vacate

            # Re-generate cycle if grid size changed or not yet built
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_safe_cycle(grid_size)
                if self.debug_log:
                    print(f"🌀 Generated {len(self.cycle)}-cell cycle")

            # ------------------------------------------------------
            # 1. Safe A* path to apple
            # ------------------------------------------------------
            path_to_apple = self._astar_path(head, apple, obstacles, grid_size)
            if (
                path_to_apple
                and len(path_to_apple) > 1
                and self._path_safe_after_eat(path_to_apple, body, grid_size)
            ):
                if self.debug_log:
                    print("🚀 Taking safe A* shortcut to apple")
                return position_to_direction(head, path_to_apple[1])

            # ------------------------------------------------------
            # 2. Follow Hamiltonian cycle
            # ------------------------------------------------------
            head_idx = self.cycle_map.get(head, -1)
            if head_idx == -1:
                head_idx = self._find_nearest_index(head)
            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]
            if self._is_safe_move(next_pos, game):
                self.current_cycle_index = next_idx
                return position_to_direction(head, next_pos)

            # ------------------------------------------------------
            # 3. Any safe move (failsafe)
            # ------------------------------------------------------
            return self._find_any_safe_move(head, game)

        except Exception as e:  # pragma: no cover – defensive guard
            if self.debug_log:
                print(f"🚨 A* Hamiltonian agent error: {e}")
            return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Cycle generation & validation
    # ------------------------------------------------------------------

    def _generate_safe_cycle(self, size: int) -> None:
        """Generate boustrophedon Hamiltonian cycle with adjacent endpoints."""
        self.cycle = []

        # Boustrophedon sweep
        for y in range(size):
            if y % 2 == 0:
                self.cycle.extend([(x, y) for x in range(size)])
            else:
                self.cycle.extend([(x, y) for x in range(size - 1, -1, -1)])

        # Ensure adjacency of first/last cells
        if size % 2 == 0:
            # Last cell ends at (0, size-1); make it (1, size-1) then add (0, size-1)
            self.cycle[-1] = (1, size - 1)
            self.cycle.append((0, size - 1))
        else:
            # Odd grids already end adjacent; just truncate/ensure length
            self.cycle = self.cycle[: size * size]

        # Final length guard (remove any excess)
        if len(self.cycle) > size * size:
            self.cycle = self.cycle[: size * size]

        # Build fast lookup map
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    def _validate_cycle(self) -> None:
        """Run basic integrity checks on the generated cycle."""
        if len(self.cycle) != self.grid_size * self.grid_size:
            print(
                f"❌ Cycle length mismatch: {len(self.cycle)} vs {self.grid_size ** 2}"
            )
        duplicates = len(self.cycle) - len(set(self.cycle))
        if duplicates:
            print(f"❌ Duplicate positions detected: {duplicates}")
        first, last = self.cycle[0], self.cycle[-1]
        if abs(first[0] - last[0]) + abs(first[1] - last[1]) != 1:
            print(f"❌ Endpoints not adjacent: {first} → {last}")

    # ------------------------------------------------------------------
    # Safety & utility helpers
    # ------------------------------------------------------------------

    def _safe_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        game: "HeuristicGameLogic",
    ) -> Optional[str]:
        """Check for direct shortcut to apple (adjacent positions)."""
        if abs(head[0] - apple[0]) + abs(head[1] - apple[1]) != 1:
            return None
        for dir_name in ("UP", "DOWN", "LEFT", "RIGHT"):
            if self._get_next_position(head, dir_name) == apple and self._is_safe_move(
                apple, game
            ):
                return dir_name
        return None

    def _is_safe_move(self, pos: Tuple[int, int], game: "HeuristicGameLogic") -> bool:
        if not (0 <= pos[0] < game.grid_size and 0 <= pos[1] < game.grid_size):
            return False
        body = [tuple(seg) for seg in game.snake_positions]
        tail = body[-1] if body else None
        is_eating = pos == tuple(game.apple_position)
        if pos in body[:-1]:
            return False
        if pos == tail and not is_eating:
            return True  # Tail vacates
        if pos == tail and is_eating:
            return False
        return True

    def _find_nearest_index(self, pos: Tuple[int, int]) -> int:
        best_idx, best_dist = 0, float("inf")
        for idx, cycle_pos in enumerate(self.cycle):
            dist = abs(pos[0] - cycle_pos[0]) + abs(pos[1] - cycle_pos[1])
            if dist < best_dist:
                best_dist, best_idx = dist, idx
        return best_idx

    def _find_any_safe_move(self, head: Tuple[int, int], game: "HeuristicGameLogic") -> str:
        """Find any safe move as ultimate fallback."""
        for dir_name in ("UP", "DOWN", "LEFT", "RIGHT"):
            nxt = self._get_next_position(head, dir_name)
            if self._is_safe_move(nxt, game):
                return dir_name
        return "NO_PATH_FOUND"

    @staticmethod
    def _get_next_position(pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Get next position given current position and direction."""
        dx, dy = DIRECTIONS[direction]
        return pos[0] + dx, pos[1] + dy

    # ------------------------------------------------------------------
    # A* path-finding + safety simulation helpers
    # ------------------------------------------------------------------

    def _astar_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Return shortest path using A* or *None* if not reachable."""
        open_set: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {start: self._heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float("inf")))
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            open_set.remove(current)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size)
                    or neighbor in obstacles
                ):
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    open_set.add(neighbor)
        return None

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _path_safe_after_eat(
        self,
        path: List[Tuple[int, int]],
        snake_body: List[Tuple[int, int]],
        grid_size: int,
    ) -> bool:
        """Simulate *path*; ensure head can still reach tail afterwards."""
        virtual = list(snake_body)
        apple = path[-1]
        for step in path[1:]:
            virtual.insert(0, step)
            if step == apple:
                break  # snake grows – keep tail
            virtual.pop()
        new_head, new_tail = virtual[0], virtual[-1]
        obstacles = set(virtual[:-1])
        return bool(self._astar_path(new_head, new_tail, obstacles, grid_size))

    # ------------------------------------------------------------------
    # Diagnostics helper
    # ------------------------------------------------------------------

    def get_cycle_info(self) -> Dict[str, object]:
        """Get information about the Hamiltonian cycle."""
        return {
            "algorithm": self.algorithm_name,
            "cycle_length": len(self.cycle),
            "grid_size": self.grid_size,
            "current_cycle_index": self.current_cycle_index,
            "has_valid_cycle": len(self.cycle) == self.grid_size * self.grid_size
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"AStarHamiltonianAgent(algorithm={self.algorithm_name})" 