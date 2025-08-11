from __future__ import annotations

from typing import Any, Dict, List

from core.game_logic import BaseGameLogic
from config.ui_constants import GRID_SIZE

class RLGameLogic(BaseGameLogic):
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        self.move_features: List[Dict[str, Any]] = []

    def _post_move(self, apple_eaten: bool) -> None:
        self.move_features.append({
            "head": tuple(self.head_position),
            "apple": tuple(self.apple_position),
            "score": self.score,
            "steps": self.steps,
            "apple_eaten": apple_eaten,
        })

    def compute_metrics(self) -> Dict[str, float]:
        apples_per_step = (self.score or 0) / max(1, self.steps)
        return {"apples_per_step": float(apples_per_step)}