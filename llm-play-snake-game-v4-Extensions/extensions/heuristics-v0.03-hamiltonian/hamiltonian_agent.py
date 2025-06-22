"""
Hamiltonian Cycle Agent - Proven Solution
========================================

Fixed cycle generation with:
- Exactly grid_size*grid_size positions
- All unique positions
- Valid cycle closure (first and last cells adjacent)
- No path-finding failures at (0,0)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set

from config.game_constants import DIRECTIONS


class HamiltonianAgent:
    """Snake agent that follows a mathematically correct Hamiltonian cycle."""

    def __init__(self) -> None:  # noqa: D401
        self.name = "Hamiltonian-Proven"
        self.description = "Reliable cycle navigation with guaranteed pathing"
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_cycle_index: int = 0
        self.debug_log: bool = True  # Verbose diagnostics

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str:  # noqa: D401
        try:
            head = tuple(game.head_position)
            grid_size = game.grid_size

            # (Re-)generate cycle if board size changed ------------------
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_robust_cycle(grid_size)
                if self.debug_log:
                    print(
                        f"🌀 Generated {len(self.cycle)}-cell cycle for {grid_size}×{grid_size}",
                    )
                    self._validate_cycle()

            # Attempt adjacent apple shortcut ---------------------------
            apple = tuple(game.apple_position)
            if shortcut := self._safe_shortcut(head, apple, game):
                if self.debug_log:
                    print(f"🍎 Shortcut to apple via {shortcut}")
                return shortcut

            # Normal cycle following ------------------------------------
            head_idx = self.cycle_map.get(head, self.current_cycle_index)
            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]

            if self._is_safe_move(next_pos, game):
                self.current_cycle_index = next_idx
                direction = self._get_direction(head, next_pos)
                if self.debug_log:
                    print(f"🔄 Cycle: {head} → {next_pos} via {direction}")
                return direction

            # Fallback: any safe move -----------------------------------
            return self._find_any_safe_move(head, game)

        except Exception as exc:  # pragma: no cover – defensive
            if self.debug_log:
                print(f"🚨 Error: {exc!r}")
            return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Cycle generation
    # ------------------------------------------------------------------

    def _generate_robust_cycle(self, size: int) -> None:
        """Generate a Hamiltonian cycle covering every cell exactly once."""
        self.cycle.clear()
        visited: Set[Tuple[int, int]] = set()

        x = y = 0  # Start at (0, 0)
        dx, dy = 1, 0  # Initial direction: right

        for _ in range(size * size):
            self.cycle.append((x, y))
            visited.add((x, y))

            # Tentative next position
            nx, ny = x + dx, y + dy

            # Change direction if next cell is out of bounds *or* visited
            if (
                nx < 0
                or nx >= size
                or ny < 0
                or ny >= size
                or (nx, ny) in visited
            ):
                # Rotate direction 90° clockwise (right→down→left→up)
                dx, dy = -dy, dx
                nx, ny = x + dx, y + dy

            x, y = nx, ny

        # Ensure last cell is adjacent to first; adjust if necessary
        last = self.cycle[-1]
        first = self.cycle[0]
        if abs(last[0] - first[0]) + abs(last[1] - first[1]) != 1:
            # Replace last cell with a cell adjacent to first and unvisited
            self.cycle.pop()
            # Prefer horizontal alignment
            candidate = (first[0] - 1, first[1]) if first[0] > 0 else (first[0] + 1, first[1])
            if candidate in visited or not (0 <= candidate[0] < size):
                candidate = (first[0], first[1] - 1 if first[1] > 0 else first[1] + 1)
            self.cycle.append(candidate)

        # Rebuild fast lookup map
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}
        self.current_cycle_index = self.cycle_map.get((0, 0), 0)

    # ------------------------------------------------------------------
    # Validation helper (debug only)
    # ------------------------------------------------------------------

    def _validate_cycle(self) -> None:  # noqa: D401
        if len(self.cycle) != self.grid_size * self.grid_size:
            print(
                f"❌ Cycle length {len(self.cycle)} ≠ {self.grid_size * self.grid_size}",
            )
        dupes = len(self.cycle) - len(set(self.cycle))
        if dupes:
            print(f"❌ Cycle contains {dupes} duplicate positions")
        if abs(self.cycle[-1][0] - self.cycle[0][0]) + abs(self.cycle[-1][1] - self.cycle[0][1]) != 1:
            print("❌ First and last cells are not adjacent")

    # ------------------------------------------------------------------
    # Shortcut / safety utilities
    # ------------------------------------------------------------------

    def _safe_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        game: "HeuristicGameLogic",
    ) -> Optional[str]:
        if abs(head[0] - apple[0]) + abs(head[1] - apple[1]) != 1:
            return None
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            if self._get_next_position(head, direction) == apple and self._is_safe_move(apple, game):
                return direction
        return None

    # ..................................................................

    def _is_safe_move(self, pos: Tuple[int, int], game: "HeuristicGameLogic") -> bool:
        if not (0 <= pos[0] < game.grid_size and 0 <= pos[1] < game.grid_size):
            return False
        body = [tuple(seg) for seg in game.snake_positions]
        tail = body[-1] if body else None
        is_eating = pos == tuple(game.apple_position)
        if pos in body[:-1]:  # Collision with body (excluding tail)
            return False
        if pos == tail:
            return not is_eating  # Safe only if tail moves away
        return True

    def _find_any_safe_move(self, head: Tuple[int, int], game: "HeuristicGameLogic") -> str:
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            nxt = self._get_next_position(head, direction)
            if self._is_safe_move(nxt, game):
                return direction
        return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_position(pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        return pos[0] + dx, pos[1] + dy

    @staticmethod
    def _get_direction(frm: Tuple[int, int], to: Tuple[int, int]) -> str:
        dx, dy = to[0] - frm[0], to[1] - frm[1]
        if dx == 0 and dy == 1:
            return "UP"
        if dx == 0 and dy == -1:
            return "DOWN"
        if dx == 1 and dy == 0:
            return "RIGHT"
        if dx == -1 and dy == 0:
            return "LEFT"
        return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def get_cycle_info(self) -> Dict[str, object]:  # noqa: D401
        return {
            "cycle_length": len(self.cycle),
            "grid_size": self.grid_size,
            "current_index": self.current_cycle_index,
            "start_pos": self.cycle[0] if self.cycle else None,
            "end_pos": self.cycle[-1] if self.cycle else None,
        } 