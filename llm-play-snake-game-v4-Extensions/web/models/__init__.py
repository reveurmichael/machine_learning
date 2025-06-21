"""
Web Models Module - MVC Data Layer
=================================

Implements the Model layer of MVC architecture with game state management,
business logic, and observer pattern for state change notifications.

Design Patterns:
    - Observer Pattern: State change notifications to views and controllers
    - Strategy Pattern: Different StateProvider implementations
    - Factory Pattern: DataSource creation and management
    - Singleton Pattern: Centralized state management

Components:
    GameStateModel      - Central game state management
    StateProvider       - Abstract data source interface
    GameEvent          - Event objects for state changes
    Observer           - Observer interface for notifications
    DataSource         - Abstract data access layer

Educational Goals:
    - Demonstrate separation of data concerns from presentation
    - Show Observer pattern implementation
    - Illustrate how models encapsulate business logic
    - Provide clean interface for data access
"""

from .game_state_model import GameStateModel, StateProvider, LiveGameStateProvider, ReplayStateProvider
from .events import GameEvent, EventType, EventFactory
from .observers import Observer, LoggingObserver, StatisticsObserver

__all__ = [
    'GameStateModel',
    'StateProvider',
    'LiveGameStateProvider', 
    'ReplayStateProvider',
    'GameEvent',
    'EventType',
    'EventFactory',
    'Observer',
    'LoggingObserver',
    'StatisticsObserver'
] 