"""
Core package initialization.
This file exposes the core game components for use by the application.
"""

# Controller imports moved to avoid circular dependencies
from core.game_logic import GameLogic
from core.game_data import GameData
from core.game_data import BaseGameData
from core.game_manager import GameManager, BaseGameManager
from core.game_loop import run_game_loop
from core.game_stats import GameStatistics, TimeStats, TokenStats, StepStats, RoundBuffer
from core.game_rounds import RoundManager
from core.game_agents import SnakeAgent
from core.game_runner import play as quick_play


__all__ = [
    'GameLogic',
    'GameData',
    'BaseGameData',
    'GameManager',
    'BaseGameManager',
    'run_game_loop',
    'GameStatistics',
    'TimeStats',
    'TokenStats',
    'StepStats',
    'RoundBuffer',
    'RoundManager',
    'SnakeAgent',
    'quick_play',
] 