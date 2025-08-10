from __future__ import annotations

from typing import Any, Optional

import pickle
from config.game_constants import DIRECTIONS

class XGBoostAgent:
    algorithm_name: str = "XGBOOST"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = None
        if model_path:
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def get_move(self, game: Any) -> Optional[str]:
        head = tuple(game.head) if hasattr(game, "head") else tuple(game.head_position)
        apple = tuple(game.apple) if hasattr(game, "apple") else tuple(game.apple_position)
        current_dir = (
            game.get_current_direction_key() if hasattr(game, "get_current_direction_key") else "NONE"
        )
        best_dist = 10**9
        best_key: Optional[str] = None
        for key, vec in DIRECTIONS.items():
            if self._is_reverse(key, current_dir):
                continue
            nx, ny = head[0] + vec[0], head[1] + vec[1]
            dist = abs(nx - apple[0]) + abs(ny - apple[1])
            if dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key or "NO_PATH_FOUND"

    def _is_reverse(self, next_key: str, current_key: str) -> bool:
        if current_key == "NONE":
            return False
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return opposites.get(next_key) == current_key