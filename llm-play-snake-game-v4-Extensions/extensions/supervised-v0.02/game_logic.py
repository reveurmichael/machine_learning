from __future__ import annotations

from typing import Any, Dict, List

from core.game_logic import BaseGameLogic
from config.ui_constants import GRID_SIZE

class SupervisedV02GameLogic(BaseGameLogic):
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        self.move_features: List[Dict[str, Any]] = []

    def _post_move(self, apple_eaten: bool) -> None:
        # Record minimal features
        features = {
            "head": tuple(self.head_position),
            "apple": tuple(self.apple_position),
            "score": self.score,
            "steps": self.steps,
            "apple_eaten": apple_eaten,
        }
        self.move_features.append(features)