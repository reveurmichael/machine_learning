"""
Core package initialization.
This file exposes the core game components for use by the application.
"""

from core.game_controller import GameController
from core.game_logic import GameLogic
from core.game_data import GameData
from core.game_manager import GameManager

__all__ = ['GameController', 'GameLogic', 'GameData', 'GameManager'] 