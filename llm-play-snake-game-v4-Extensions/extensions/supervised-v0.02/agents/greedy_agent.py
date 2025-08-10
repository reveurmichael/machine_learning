from __future__ import annotations

from typing import Any, Optional, Tuple

from config.game_constants import DIRECTIONS

class GreedyAgent:
    algorithm_name: str = "GREEDY"

    def get_move(self, game: Any) -> Optional[str]:
        head = tuple(game.head) if hasattr(game, "head") else tuple(game.head_position)
        apple = tuple(game.apple) if hasattr(game, "apple") else tuple(game.apple_position)
        current_dir = (
            game.get_current_direction_key() if hasattr(game, "get_current_direction_key") else "NONE"
        )
        best: Tuple[int, Optional[str]] = (10**9, None)
        for key, vec in DIRECTIONS.items():
            if self._is_reverse(key, current_dir):
                continue
            nx, ny = head[0] + vec[0], head[1] + vec[1]
            dist = abs(nx - apple[0]) + abs(ny - apple[1])
            if dist < best[0]:
                best = (dist, key)
        return best[1] or "NO_PATH_FOUND"

    def _is_reverse(self, next_key: str, current_key: str) -> bool:
        if current_key == "NONE":
            return False
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return opposites.get(next_key) == current_key