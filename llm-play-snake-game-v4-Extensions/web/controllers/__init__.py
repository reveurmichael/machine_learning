"""
Web Controllers Module - MVC Architecture
--------------------

Implements role-based controller hierarchy following MVC principles.
Controllers handle HTTP requests, coordinate with models, and delegate to views.

Design Patterns:
    - Template Method Pattern: Base controllers define request handling flow
    - Strategy Pattern: Different controllers for different game modes
    - Chain of Responsibility: Request filtering and middleware
    - Observer Pattern: Event notification between layers

Controller Hierarchy:
    BaseWebController           - Abstract base with common functionality
    ├── BaseGamePlayController  - Generic base for active gameplay modes
    │   ├── GamePlayController  - Task-0 LLM-driven gameplay
    │   └── HumanGameController - Human player input handling
    └── BaseGameViewingController - Generic base for viewing/replay modes
        └── ReplayController    - Game replay and navigation

Educational Goals:
    - Demonstrate role-based inheritance
    - Show proper separation of request handling from business logic
    - Illustrate how design patterns solve real architectural problems
"""

from .base_controller import BaseWebController, RequestType, RequestContext
from .human_controller import HumanGameController
from .llm_controller import GamePlayController  # Task-0 gameplay controller
from .game_controllers import (
    BaseGamePlayController,
    BaseGameViewingController,
    GameMode,
)
from .replay_controller import ReplayController

__all__ = [
    'BaseWebController',
    'RequestType',
    'RequestContext',
    'HumanGameController',
    'GamePlayController',
    'BaseGamePlayController',
    'BaseGameViewingController',
    'GameMode',
    'ReplayController'
] 