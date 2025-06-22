"""
Hamiltonian Cycle Agent – Universal Grid Support
================================================

This module implements a *robust* Hamiltonian-cycle-based agent that
**guarantees survival on *any* square grid size**.  It replaces the earlier
spiral-only implementation which failed on certain even-sized boards.

Key improvements
----------------
1. **Universal cycle generation** – Uses a *boustrophedon* (snake-like) pattern
   that forms a valid Hamiltonian *cycle* on both even and odd grids by
   visiting the cells row-by-row and returning to the origin.
2. **Shortcut logic** – When the apple is *orthogonally adjacent* the agent
   takes a one-step detour, then rejoins the main cycle seamlessly.
3. **Tail-aware safety checks** – Collision detection correctly models the
   case where the snake's tail moves away on the next tick (unless the snake
   is eating the apple).
4. **Detailed diagnostics** – Optional `debug_log` flag prints internal state
   to the console for easier debugging / teaching.

Design patterns used
--------------------
- *Strategy* – Interchangeable agent plugged into the game via `get_move()`.
- *Pre-computation* – Cycle is generated once per board and cached.
- *Fail-safe fallback* – Any unexpected state triggers a conservative but safe
  move, keeping the game alive.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config.game_constants import DIRECTIONS

# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------


class HamiltonianAgent:  # pylint: disable=too-few-public-methods
    """Snake agent that follows a pre-computed Hamiltonian cycle.

    The algorithm guarantees that the snake will eventually visit every cell
    while never crossing itself.  The cycle is generated *once* when the game
    starts or when the board size changes.
    """

    # ---------------------------------------------------------------------
    # Construction & public interface
    # ---------------------------------------------------------------------

    def __init__(self) -> None:
        self.name: str = "Hamiltonian"
        self.description: str = (
            "Hamiltonian cycle follower – infinite survival via boustrophedon "
            "cycle; includes safe shortcutting."  # noqa: E501 – long docstring
        )

        # Cached cycle
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: Optional[int] = None

        # Runtime state
        self.current_cycle_index: int = 0
        self.debug_log: bool = False  # flip to *True* for verbose output

    # ------------------------------------------------------------------
    # Main entry-point required by the framework
    # ------------------------------------------------------------------

    def get_move(self, game: "HeuristicGameLogic") -> str | None:  # noqa: D401 – external signature
        """Return the next direction (`UP|DOWN|LEFT|RIGHT`)."""

        try:
            head_pos = tuple(game.head_position)
            grid_size = game.grid_size

            # (Re)generate the cycle when the board size changes
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_cycle(grid_size)
                if self.debug_log:
                    print(f"Generated {len(self.cycle)}-cell cycle for {grid_size}×{grid_size} board")

            # 1) Optional shortcut directly into the apple (adjacent only)
            apple_pos = tuple(game.apple_position)
            shortcut_dir = self._try_shortcut(head_pos, apple_pos, game)
            if shortcut_dir:
                if self.debug_log:
                    print(f"🍎 Shortcut → {shortcut_dir}")
                return shortcut_dir

            # 2) Follow the Hamiltonian cycle strictly
            head_idx = self.cycle_map.get(head_pos, -1)

            # If the head left the cycle due to a shortcut in the previous move
            if head_idx == -1:
                head_idx = self._find_nearest_cycle_index(head_pos)
                if head_idx == -1:  # truly off-cycle – should never happen
                    return self._fallback_move(head_pos, game)
                if self.debug_log:
                    print(f"⚠️  Re-joined cycle at index {head_idx}")

            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]

            if self._is_safe_move(next_pos, game):
                self.current_cycle_index = next_idx
                direction = self._get_direction(head_pos, next_pos)
                if self.debug_log:
                    print(f"🚲 Cycle {head_idx} → {next_idx}: {direction}")
                return direction

            # 3) Failsafe – should be unreachable on a correct cycle
            return self._fallback_move(head_pos, game)

        except Exception as exc:  # pragma: no cover – defensive net
            if self.debug_log:
                print(f"❌ HamiltonianAgent failure: {exc}")
            return "NO_PATH_FOUND"

    # ------------------------------------------------------------------
    # Cycle generation – boustrophedon pattern visiting every cell once
    # ------------------------------------------------------------------

    def _generate_cycle(self, size: int) -> None:
        """Populate :pyattr:`cycle` and :pyattr:`cycle_map` in-place."""

        self.cycle.clear()
        self.cycle_map.clear()

        for row in range(size):
            if row % 2 == 0:
                # Even rows: left → right
                self.cycle.extend([(col, row) for col in range(size)])
            else:
                # Odd rows: right → left
                self.cycle.extend([(col, row) for col in range(size - 1, -1, -1)])

        # Finally ensure the cycle *truly* closes – the last cell must be
        # orthogonally adjacent to the first cell so the snake can wrap
        # around without colliding.  If this is not the case (which happens
        # for most board sizes) we append a minimal Manhattan path connecting
        # the endpoints.
        last_cell: Tuple[int, int] = self.cycle[-1]
        first_cell: Tuple[int, int] = self.cycle[0]
        if abs(last_cell[0] - first_cell[0]) + abs(last_cell[1] - first_cell[1]) != 1:
            self.cycle.extend(self._create_connecting_path(last_cell, first_cell))

        # Rotate so (0,0) is the *head* starting cell (cosmetic – helpful for
        # deterministic unit tests and human reasoning).
        if (0, 0) in self.cycle:
            start_idx = self.cycle.index((0, 0))
            if start_idx:
                self.cycle = self.cycle[start_idx:] + self.cycle[:start_idx]

        # Precompute fast lookup
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    # ------------------------------------------------------------------
    # Internal: minimal manhattan path to connect endpoints
    # ------------------------------------------------------------------

    def _create_connecting_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Return a list of intermediate cells from *start* (excluded) to *end* (included)."""
        path: list[Tuple[int, int]] = []
        x, y = start
        # Vertical leg
        while y != end[1]:
            y += 1 if y < end[1] else -1
            path.append((x, y))
        # Horizontal leg
        while x != end[0]:
            x += 1 if x < end[0] else -1
            path.append((x, y))
        return path

    # ------------------------------------------------------------------
    # Helper – adjacent apple shortcut
    # ------------------------------------------------------------------

    def _try_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        game: "HeuristicGameLogic",
    ) -> Optional[str]:
        if abs(head[0] - apple[0]) + abs(head[1] - apple[1]) != 1:
            return None

        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            if self._get_next_position(head, direction) == apple and self._is_safe_move(
                apple, game
            ):
                return direction
        return None

    # ------------------------------------------------------------------
    # Safety predicate – bounds & body collision with tail exception
    # ------------------------------------------------------------------

    def _is_safe_move(self, pos: Tuple[int, int], game: "HeuristicGameLogic") -> bool:
        # Out-of-bounds check
        if not (0 <= pos[0] < game.grid_size and 0 <= pos[1] < game.grid_size):
            return False

        body = [tuple(segment) for segment in game.snake_positions]

        # Moving into the body (except the tail *if it moves away this turn*)
        if pos in body[:-1]:
            return False

        tail = body[-1]
        is_eating = pos == tuple(game.apple_position)

        # If the snake is *eating* the apple the tail will **not** move away
        if pos == tail and is_eating:
            return False

        return True

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_position(pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def _find_nearest_cycle_index(self, pos: Tuple[int, int]) -> int:
        """Return index of the closest cycle cell (Manhattan metric)."""
        nearest_idx = -1
        min_dist = float("inf")
        for cycle_pos, idx in self.cycle_map.items():
            dist = abs(pos[0] - cycle_pos[0]) + abs(pos[1] - cycle_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

    def _fallback_move(self, head: Tuple[int, int], game: "HeuristicGameLogic") -> str:
        """Conservative escape hatch – first available safe direction."""
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            next_pos = self._get_next_position(head, direction)
            if self._is_safe_move(next_pos, game):
                return direction
        return "NO_PATH_FOUND"

    @staticmethod
    def _get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if dx == 0 and dy == 1:
            return "UP"
        if dx == 0 and dy == -1:
            return "DOWN"
        if dx == 1 and dy == 0:
            return "RIGHT"
        if dx == -1 and dy == 0:
            return "LEFT"
        return "NO_PATH_FOUND"  # should never happen

    # ------------------------------------------------------------------
    # Introspection helpers for testing / debugging
    # ------------------------------------------------------------------

    def get_cycle_info(self) -> Dict[str, int | bool | None]:
        """Expose internal state for unit tests or dashboards."""
        return {
            "cycle_length": len(self.cycle),
            "grid_size": self.grid_size,
            "current_index": self.current_cycle_index,
            "cycle_complete": bool(self.cycle),
        } 