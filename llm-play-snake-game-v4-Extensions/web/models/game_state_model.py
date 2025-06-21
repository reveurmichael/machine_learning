"""
Game State Model - Central Data Management
=========================================

Central model for managing game state with observer pattern support.
Encapsulates game data, business logic, and state change notifications.

Design Patterns:
    - Observer Pattern: Notifies observers of state changes
    - Strategy Pattern: Pluggable state providers
    - Singleton Pattern: Centralized state management
    - Template Method: State update workflow

Educational Goals:
    - Demonstrate model layer responsibilities in MVC
    - Show observer pattern in practice
    - Illustrate separation of data and presentation concerns
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum

# Import core game components
from core.game_controller import GameController
from replay.replay_engine import ReplayEngine

# Import MVC components
from .events import GameEvent, EventFactory, EventType
from .observers import Observer

# Configure logging
logger = logging.getLogger(__name__)


class GameMode(Enum):
    """Types of game modes supported."""
    LIVE_HUMAN = "live_human"
    LIVE_LLM = "live_llm" 
    REPLAY = "replay"
    DEMO = "demo"


@dataclass
class GameState:
    """
    Immutable game state snapshot.
    
    Contains all information needed to represent the current state
    of the game at a specific point in time.
    
    Design Pattern: Immutable Object
    - State cannot be modified after creation
    - Thread-safe sharing between components
    - Enables state history and replay functionality
    """
    timestamp: float
    score: int
    steps: int
    game_over: bool
    game_mode: GameMode
    snake_positions: List[tuple]
    apple_position: tuple
    grid_size: int
    direction: Optional[str]
    end_reason: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "steps": self.steps,
            "game_over": self.game_over,
            "game_mode": self.game_mode.value,
            "snake_positions": self.snake_positions,
            "apple_position": self.apple_position,
            "grid_size": self.grid_size,
            "direction": self.direction,
            "end_reason": self.end_reason,
            "metadata": self.metadata
        }


class StateProvider(ABC):
    """
    Abstract interface for game state data sources.
    
    Enables different sources of game state (live game, replay, demo)
    to be used interchangeably through the Strategy pattern.
    
    Design Pattern: Strategy Pattern
    - Enables different data sources
    - Allows switching between live and replay modes
    - Provides consistent interface for state access
    """
    
    @abstractmethod
    def get_current_state(self) -> GameState:
        """Get the current game state."""
        pass
    
    @abstractmethod
    def get_game_mode(self) -> GameMode:
        """Get the current game mode."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if state provider is available."""
        pass
    
    @abstractmethod
    def reset_game(self) -> bool:
        """Reset the game state."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the state provider."""
        pass


class LiveGameStateProvider(StateProvider):
    """
    State provider for live game sessions.
    
    Wraps GameController to provide state for active gameplay.
    """
    
    def __init__(self, game_controller: GameController, game_mode: GameMode):
        """
        Initialize live game state provider.
        
        Args:
            game_controller: Active game controller instance
            game_mode: Type of live game (human/LLM)
        """
        self.game_controller = game_controller
        self.game_mode = game_mode
        self._last_state_timestamp = 0.0
    
    def get_current_state(self) -> GameState:
        """Get current state from live game controller."""
        if not self.game_controller:
            raise RuntimeError("Game controller not available")
        
        current_time = time.time()
        
        # Build state from game controller
        state = GameState(
            timestamp=current_time,
            score=getattr(self.game_controller, 'score', 0),
            steps=getattr(self.game_controller, 'steps', 0),
            game_over=getattr(self.game_controller, 'game_over', False),
            game_mode=self.game_mode,
            snake_positions=self._get_snake_positions(),
            apple_position=self._get_apple_position(),
            grid_size=getattr(self.game_controller, 'grid_size', 10),
            direction=getattr(self.game_controller, 'current_direction', None),
            end_reason=getattr(self.game_controller, 'end_reason', None),
            metadata=self._get_metadata()
        )
        
        self._last_state_timestamp = current_time
        return state
    
    def _get_snake_positions(self) -> List[tuple]:
        """Extract snake positions from game controller."""
        if hasattr(self.game_controller, 'snake_positions'):
            positions = self.game_controller.snake_positions
            if hasattr(positions, 'tolist'):
                return positions.tolist()
            return list(positions)
        return []
    
    def _get_apple_position(self) -> tuple:
        """Extract apple position from game controller."""
        if hasattr(self.game_controller, 'apple_position'):
            position = self.game_controller.apple_position
            if hasattr(position, 'tolist'):
                return tuple(position.tolist())
            return tuple(position)
        return (0, 0)
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Extract additional metadata from game controller."""
        metadata = {}
        
        # Add timing information
        if hasattr(self.game_controller, 'start_time'):
            metadata['game_duration'] = time.time() - self.game_controller.start_time
        
        # Add performance stats if available
        if hasattr(self.game_controller, 'get_performance_stats'):
            metadata['performance'] = self.game_controller.get_performance_stats()
        
        return metadata
    
    def get_game_mode(self) -> GameMode:
        """Get the game mode."""
        return self.game_mode
    
    def is_available(self) -> bool:
        """Check if game controller is available."""
        return self.game_controller is not None
    
    def reset_game(self) -> bool:
        """Reset the live game."""
        try:
            if self.game_controller and hasattr(self.game_controller, 'reset'):
                self.game_controller.reset()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reset live game: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of live game provider."""
        return {
            "provider_type": "live_game",
            "game_mode": self.game_mode.value,
            "controller_available": self.is_available(),
            "last_state_update": self._last_state_timestamp,
            "controller_class": self.game_controller.__class__.__name__ if self.game_controller else None
        }


class ReplayStateProvider(StateProvider):
    """
    State provider for game replay sessions.
    
    Wraps ReplayEngine to provide state for game replay.
    """
    
    def __init__(self, replay_engine: ReplayEngine):
        """
        Initialize replay state provider.
        
        Args:
            replay_engine: Active replay engine instance
        """
        self.replay_engine = replay_engine
        self._last_state_timestamp = 0.0
    
    def get_current_state(self) -> GameState:
        """Get current state from replay engine."""
        if not self.replay_engine:
            raise RuntimeError("Replay engine not available")
        
        current_time = time.time()
        
        # Build state from replay engine
        state = GameState(
            timestamp=current_time,
            score=getattr(self.replay_engine, 'score', 0),
            steps=getattr(self.replay_engine, 'steps', 0),
            game_over=getattr(self.replay_engine, 'game_over', False),
            game_mode=GameMode.REPLAY,
            snake_positions=self._get_replay_snake_positions(),
            apple_position=self._get_replay_apple_position(),
            grid_size=getattr(self.replay_engine, 'grid_size', 10),
            direction=getattr(self.replay_engine, 'current_direction', None),
            end_reason=getattr(self.replay_engine, 'end_reason', None),
            metadata=self._get_replay_metadata()
        )
        
        self._last_state_timestamp = current_time
        return state
    
    def _get_replay_snake_positions(self) -> List[tuple]:
        """Extract snake positions from replay engine."""
        if hasattr(self.replay_engine, 'snake_positions'):
            positions = self.replay_engine.snake_positions
            if hasattr(positions, 'tolist'):
                return positions.tolist()
            return list(positions)
        return []
    
    def _get_replay_apple_position(self) -> tuple:
        """Extract apple position from replay engine."""
        if hasattr(self.replay_engine, 'apple_position'):
            position = self.replay_engine.apple_position
            if hasattr(position, 'tolist'):
                return tuple(position.tolist())
            return tuple(position)
        return (0, 0)
    
    def _get_replay_metadata(self) -> Dict[str, Any]:
        """Extract metadata from replay engine."""
        metadata = {}
        
        if hasattr(self.replay_engine, 'game_number'):
            metadata['game_number'] = self.replay_engine.game_number
        
        if hasattr(self.replay_engine, 'total_games'):
            metadata['total_games'] = self.replay_engine.total_games
        
        if hasattr(self.replay_engine, 'paused'):
            metadata['paused'] = self.replay_engine.paused
        
        return metadata
    
    def get_game_mode(self) -> GameMode:
        """Get the game mode."""
        return GameMode.REPLAY
    
    def is_available(self) -> bool:
        """Check if replay engine is available."""
        return self.replay_engine is not None
    
    def reset_game(self) -> bool:
        """Reset the replay to beginning."""
        try:
            if self.replay_engine and hasattr(self.replay_engine, 'reset'):
                self.replay_engine.reset()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reset replay: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of replay provider."""
        return {
            "provider_type": "replay",
            "game_mode": GameMode.REPLAY.value,
            "engine_available": self.is_available(),
            "last_state_update": self._last_state_timestamp,
            "engine_class": self.replay_engine.__class__.__name__ if self.replay_engine else None
        }


class GameStateModel:
    """
    Central model for managing game state with observer notifications.
    
    Coordinates between state providers and observers, implementing
    the core of the MVC model layer.
    
    Design Patterns:
        - Observer Pattern: Notifies registered observers of changes
        - Strategy Pattern: Uses pluggable state providers
        - Singleton Pattern: Can be used as singleton for global state
        - Template Method: Consistent state update workflow
    """
    
    def __init__(self, state_provider: StateProvider):
        """
        Initialize game state model.
        
        Args:
            state_provider: Source of game state data
        """
        self.state_provider = state_provider
        self.observers: List[Observer] = []
        self._current_state: Optional[GameState] = None
        self._state_lock = threading.RLock()
        self._last_update = 0.0
        
        # Event generation
        self.event_factory = EventFactory()
        
        logger.info(f"Initialized GameStateModel with {type(state_provider).__name__}")
    
    def add_observer(self, observer: Observer) -> None:
        """
        Add observer to receive state change notifications.
        
        Args:
            observer: Observer instance to add
        """
        with self._state_lock:
            if observer not in self.observers:
                self.observers.append(observer)
                logger.debug(f"Added observer: {observer.get_observer_id()}")
    
    def remove_observer(self, observer: Observer) -> bool:
        """
        Remove observer from notifications.
        
        Args:
            observer: Observer instance to remove
            
        Returns:
            True if observer was removed, False if not found
        """
        with self._state_lock:
            try:
                self.observers.remove(observer)
                logger.debug(f"Removed observer: {observer.get_observer_id()}")
                return True
            except ValueError:
                return False
    
    def notify_observers(self, event: GameEvent) -> None:
        """
        Notify all observers of a game event.
        
        Args:
            event: Game event to broadcast
        """
        with self._state_lock:
            for observer in self.observers:
                try:
                    if observer.is_interested_in_event(event):
                        observer.on_game_event(event)
                except Exception as e:
                    logger.error(f"Observer {observer.get_observer_id()} failed: {e}")
    
    def get_current_state(self) -> GameState:
        """
        Get current game state with caching.
        
        Returns:
            Current game state snapshot
        """
        with self._state_lock:
            try:
                # Get fresh state from provider
                new_state = self.state_provider.get_current_state()
                
                # Check for changes and generate events
                if self._current_state is not None:
                    self._check_for_state_changes(self._current_state, new_state)
                
                # Update cached state
                self._current_state = new_state
                self._last_update = time.time()
                
                return new_state
                
            except Exception as e:
                logger.error(f"Failed to get current state: {e}")
                # Return cached state if available
                if self._current_state:
                    return self._current_state
                raise
    
    def _check_for_state_changes(self, old_state: GameState, new_state: GameState) -> None:
        """
        Compare states and generate appropriate events.
        
        Args:
            old_state: Previous game state
            new_state: New game state
        """
        # Check for move events
        if (old_state.snake_positions and new_state.snake_positions and 
            old_state.snake_positions != new_state.snake_positions):
            
            old_head = old_state.snake_positions[0] if old_state.snake_positions else (0, 0)
            new_head = new_state.snake_positions[0] if new_state.snake_positions else (0, 0)
            apple_eaten = old_state.score != new_state.score
            
            move_event = self.event_factory.create_move_event(
                direction=new_state.direction or "UNKNOWN",
                old_pos=old_head,
                new_pos=new_head,
                apple_eaten=apple_eaten,
                score_before=old_state.score,
                score_after=new_state.score
            )
            self.notify_observers(move_event)
        
        # Check for apple eaten events
        if old_state.score < new_state.score:
            apple_event = self.event_factory.create_apple_eaten_event(
                position=new_state.apple_position,
                points=new_state.score - old_state.score
            )
            self.notify_observers(apple_event)
        
        # Check for game over events
        if not old_state.game_over and new_state.game_over:
            game_over_event = GameEvent.create(
                EventType.GAME_OVER,
                source="game_state_model",
                reason=new_state.end_reason or "unknown",
                final_score=new_state.score,
                total_moves=new_state.steps
            )
            self.notify_observers(game_over_event)
    
    def reset_game(self) -> bool:
        """
        Reset the game through the state provider.
        
        Returns:
            True if reset was successful
        """
        with self._state_lock:
            try:
                # Store old state for event generation
                old_state = self._current_state
                
                # Reset through provider
                success = self.state_provider.reset_game()
                
                if success and old_state:
                    # Generate reset event
                    reset_event = GameEvent.create(
                        EventType.GAME_RESET,
                        source="game_state_model",
                        previous_score=old_state.score,
                        previous_moves=old_state.steps,
                        reason="manual_reset"
                    )
                    self.notify_observers(reset_event)
                    
                    # Clear cached state
                    self._current_state = None
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to reset game: {e}")
                return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the model and its dependencies.
        
        Returns:
            Dictionary containing health information
        """
        with self._state_lock:
            return {
                "model_status": "healthy",
                "observers_count": len(self.observers),
                "last_update": self._last_update,
                "cached_state_available": self._current_state is not None,
                "state_provider": self.state_provider.get_health_status() if self.state_provider else None
            }
    
    def get_observer_count(self) -> int:
        """Get number of registered observers."""
        with self._state_lock:
            return len(self.observers) 