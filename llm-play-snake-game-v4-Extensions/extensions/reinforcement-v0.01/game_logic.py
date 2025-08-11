from __future__ import annotations

from core.game_logic import BaseGameLogic
from config.ui_constants import GRID_SIZE

class RLGameLogic(BaseGameLogic):
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        self.algorithm_name: str = "DQN"

    def set_algorithm_name(self, name: str) -> None:
        self.algorithm_name = name