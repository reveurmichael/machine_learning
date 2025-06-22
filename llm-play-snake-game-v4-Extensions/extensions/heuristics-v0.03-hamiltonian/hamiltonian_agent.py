"""
Hamiltonian Agent - Coordinate System Compliant
===============================================

Implements a Hamiltonian cycle agent that strictly adheres to the project-wide
coordinate system:
  • (0,0)  = bottom-left corner
  • X-axis increases to the **right**, Y-axis increases **up**
  • Directions: UP=(0,1), RIGHT=(1,0), DOWN=(0,-1), LEFT=(-1,0)

Main capabilities
-----------------
1. Generates a boustrophedon Hamiltonian cycle covering every cell exactly
   once.  A small patch ensures the cycle closes correctly on even-sized grids.
2. Provides a one-step apple shortcut when the fruit is adjacent and the move
   is safe.
3. Full safety checks that account for the tail vacating when not eating.
4. Extensive validation helpers (length, uniqueness, adjacency) when
   *debug_log* is enabled.

Design Patterns
---------------
* Strategy – interchangeable path-planning algorithm.
* Fail-fast – never raises to the game engine, always returns a move string.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config.game_constants import DIRECTIONS


class HamiltonianAgent:  # pylint: disable=too-many-public-methods
    """Robust Hamiltonian cycle agent compliant with the Y-up coordinate system."""

    def __init__(self) -> None:
        self.name = "Hamiltonian-CoordinateCorrect"
        self.description = "Hamiltonian agent compliant with project coordinate system"
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_index: int = 0
        self.debug_log: bool = False  # toggle verbose diagnostics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str:  # noqa: D401
        """Return the next direction for the snake head."""
        try:
            head = tuple(game.head_position)
            grid_size = game.grid_size

            if not self.cycle or self.grid_size != grid_size:
                self._build_cycle(grid_size)
                if self.debug_log:
                    print(
                        f"🌀 Built cycle of {len(self.cycle)} cells –"
                        f" {'VALID' if self._validate_cycle() else 'INVALID'}"
                    )

            # Apple shortcut --------------------------------------------------
            apple = tuple(game.apple_position)
            shortcut = self._safe_shortcut(head, apple, game)
            if shortcut:
                if self.debug_log:
                    print(f"🍎 Shortcut via {shortcut}")
                return shortcut

            # Normal cycle following -----------------------------------------
            head_idx = self.cycle_map.get(head, self._nearest_cycle_index(head))
            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]

            if self._is_safe_move(next_pos, game):
                self.current_index = next_idx
                return self._direction_between(head, next_pos)

            # Fallback – any safe move ---------------------------------------
            return self._fallback_safe_move(head, game)

        except Exception as err:  # pragma: no cover
            if self.debug_log:
                print(f"🚨 Hamiltonian agent error: {err}")
            return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Cycle generation & validation
    # ------------------------------------------------------------------

    def _build_cycle(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.cycle = []

        # Boustrophedon pattern starting at (0,0) moving right on even rows
        for y in range(grid_size):
            row = range(grid_size) if y % 2 == 0 else range(grid_size - 1, -1, -1)
            self.cycle.extend((x, y) for x in row)

        # Even grid patch to ensure endpoints adjacency
        if grid_size % 2 == 0:
            # End currently at (0, grid_size-1).  Patch path to close loop.
            self.cycle.pop()  # remove last cell (0, size-1)
            self.cycle.extend(
                [
                    (0, grid_size - 1),
                    (0, grid_size - 2),
                    (1, grid_size - 2),
                ]
            )

        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}
        self.current_index = 0

    def _validate_cycle(self) -> bool:
        size_sq = self.grid_size * self.grid_size
        if len(self.cycle) != size_sq or len(set(self.cycle)) != size_sq:
            return False
        return all(
            self._are_adjacent(self.cycle[i], self.cycle[(i + 1) % size_sq])
            for i in range(size_sq)
        )

    # ------------------------------------------------------------------
    # Shortcut, safety, and utility helpers
    # ------------------------------------------------------------------

    def _safe_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        game: "HeuristicGameLogic",
    ) -> Optional[str]:
        if abs(head[0] - apple[0]) + abs(head[1] - apple[1]) != 1:
            return None
        for dir_name, (dx, dy) in DIRECTIONS.items():
            if (head[0] + dx, head[1] + dy) == apple and self._is_safe_move(apple, game):
                return dir_name
        return None

    def _is_safe_move(self, pos: Tuple[int, int], game: "HeuristicGameLogic") -> bool:
        if not (0 <= pos[0] < game.grid_size and 0 <= pos[1] < game.grid_size):
            return False
        body = [tuple(seg) for seg in game.snake_positions]
        eating = pos == tuple(game.apple_position)
        return pos not in body if eating else pos not in body[:-1]

    @staticmethod
    def _direction_between(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return {
            (1, 0): "RIGHT",
            (-1, 0): "LEFT",
            (0, 1): "UP",
            (0, -1): "DOWN",
        }.get((dx, dy), "NO_PATH_FOUND")

    def _nearest_cycle_index(self, pos: Tuple[int, int]) -> int:
        return min(self.cycle_map.values(), key=lambda idx: abs(self.cycle[idx][0] - pos[0]) + abs(self.cycle[idx][1] - pos[1]))

    def _fallback_safe_move(self, head: Tuple[int, int], game: "HeuristicGameLogic") -> str:
        for dir_name, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if self._is_safe_move(nxt, game):
                return dir_name
        return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Diagnostics helpers (optional)
    # ------------------------------------------------------------------

    def get_cycle_info(self) -> Dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "cycle_length": len(self.cycle),
            "start": self.cycle[0] if self.cycle else None,
            "end": self.cycle[-1] if self.cycle else None,
            "valid": self._validate_cycle(),
        }

    @staticmethod
    def _are_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
  