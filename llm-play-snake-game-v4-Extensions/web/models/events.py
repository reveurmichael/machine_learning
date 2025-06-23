"""
Game Events for Observer Pattern
--------------------

Event classes for notifying observers about game state changes.
Implements event objects that carry information about what happened in the game.

Design Patterns:
    - Observer Pattern: Events notify registered observers
    - Factory Pattern: Event creation through factory methods
    - Immutable Object: Events are read-only after creation

Educational Goals:
    - Show how events decouple components
    - Demonstrate immutable event design
    - Illustrate event hierarchy and polymorphism
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum
import uuid


class EventType(Enum):
    """Types of game events."""
    MOVE = "move"
    APPLE_EATEN = "apple_eaten"
    APPLE_SPAWNED = "apple_spawned"
    GAME_OVER = "game_over"
    GAME_RESET = "game_reset"
    SCORE_CHANGED = "score_changed"
    PAUSE_TOGGLED = "pause_toggled"
    SPEED_CHANGED = "speed_changed"


@dataclass(frozen=True)
class GameEvent(ABC):
    """
    Base class for all game events.
    
    Immutable event object that carries information about game state changes.
    All events have a unique ID, timestamp, and event type.
    
    Design Pattern: Immutable Object
    - Events cannot be modified after creation
    - Provides thread safety and consistency
    - Enables safe sharing between components
    """
    event_id: str
    timestamp: datetime
    event_type: EventType
    source: str
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, event_type: EventType, source: str = "unknown", **kwargs):
        """
        Factory method to create events with auto-generated ID and timestamp.
        
        Args:
            event_type: Type of event being created
            source: Component that generated the event
            **kwargs: Additional event-specific data
            
        Returns:
            New event instance
        """
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            source=source,
            metadata=kwargs
        )


@dataclass(frozen=True)
class MoveEvent(GameEvent):
    """
    Event fired when the snake moves.
    
    Contains information about the move including direction,
    old position, new position, and whether an apple was eaten.
    """
    direction: str
    old_head_position: tuple
    new_head_position: tuple
    apple_eaten: bool
    score_before: int
    score_after: int
    
    @classmethod
    def create_move_event(cls, direction: str, old_pos: tuple, new_pos: tuple, 
                         apple_eaten: bool, score_before: int, score_after: int,
                         source: str = "game_engine"):
        """Factory method for move events."""
        base_event = GameEvent.create(
            EventType.MOVE,
            source=source,
            direction=direction,
            old_position=old_pos,
            new_position=new_pos,
            apple_eaten=apple_eaten
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            direction=direction,
            old_head_position=old_pos,
            new_head_position=new_pos,
            apple_eaten=apple_eaten,
            score_before=score_before,
            score_after=score_after
        )


@dataclass(frozen=True)
class AppleEvent(GameEvent):
    """
    Event fired when an apple is eaten or spawned.
    
    Contains information about the apple including its position
    and the action that occurred (eaten/spawned).
    """
    apple_position: tuple
    action: str  # "eaten" or "spawned"
    points_awarded: Optional[int] = None
    
    @classmethod
    def create_apple_eaten_event(cls, position: tuple, points: int, source: str = "game_engine"):
        """Factory method for apple eaten events."""
        base_event = GameEvent.create(
            EventType.APPLE_EATEN,
            source=source,
            position=position,
            points=points
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            apple_position=position,
            action="eaten",
            points_awarded=points
        )
    
    @classmethod
    def create_apple_spawned_event(cls, position: tuple, source: str = "game_engine"):
        """Factory method for apple spawned events."""
        base_event = GameEvent.create(
            EventType.APPLE_SPAWNED,
            source=source,
            position=position
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            apple_position=position,
            action="spawned"
        )


@dataclass(frozen=True)
class GameOverEvent(GameEvent):
    """
    Event fired when the game ends.
    
    Contains information about why the game ended and final statistics.
    """
    reason: str
    final_score: int
    total_moves: int
    game_duration_seconds: float
    
    @classmethod
    def create_game_over_event(cls, reason: str, final_score: int, total_moves: int,
                              duration: float, source: str = "game_engine"):
        """Factory method for game over events."""
        base_event = GameEvent.create(
            EventType.GAME_OVER,
            source=source,
            reason=reason,
            final_score=final_score,
            total_moves=total_moves,
            duration=duration
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            reason=reason,
            final_score=final_score,
            total_moves=total_moves,
            game_duration_seconds=duration
        )


@dataclass(frozen=True)
class GameResetEvent(GameEvent):
    """
    Event fired when the game is reset.
    
    Contains information about the reset including previous game statistics.
    """
    previous_score: int
    previous_moves: int
    reset_reason: str
    
    @classmethod
    def create_reset_event(cls, prev_score: int, prev_moves: int, reason: str,
                          source: str = "game_controller"):
        """Factory method for game reset events."""
        base_event = GameEvent.create(
            EventType.GAME_RESET,
            source=source,
            previous_score=prev_score,
            previous_moves=prev_moves,
            reason=reason
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            previous_score=prev_score,
            previous_moves=prev_moves,
            reset_reason=reason
        )


@dataclass(frozen=True)
class ScoreChangedEvent(GameEvent):
    """
    Event fired when the score changes.
    
    Contains information about the score change including old and new values.
    """
    old_score: int
    new_score: int
    change_amount: int
    change_reason: str
    
    @classmethod
    def create_score_changed_event(cls, old_score: int, new_score: int, reason: str,
                                  source: str = "score_manager"):
        """Factory method for score change events."""
        change = new_score - old_score
        base_event = GameEvent.create(
            EventType.SCORE_CHANGED,
            source=source,
            old_score=old_score,
            new_score=new_score,
            change=change,
            reason=reason
        )
        
        return cls(
            event_id=base_event.event_id,
            timestamp=base_event.timestamp,
            event_type=base_event.event_type,
            source=base_event.source,
            metadata=base_event.metadata,
            old_score=old_score,
            new_score=new_score,
            change_amount=change,
            change_reason=reason
        )


class EventFactory:
    """
    Factory class for creating game events.
    
    Provides convenient methods for creating different types of events
    with proper validation and default values.
    
    Design Pattern: Factory Pattern
    - Centralizes event creation logic
    - Ensures consistent event structure
    - Provides validation and error handling
    """
    
    @staticmethod
    def create_move_event(direction: str, old_pos: tuple, new_pos: tuple,
                         apple_eaten: bool = False, score_before: int = 0, 
                         score_after: int = 0, source: str = "game_engine") -> MoveEvent:
        """Create a move event with validation."""
        if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            raise ValueError(f"Invalid direction: {direction}")
        
        if len(old_pos) != 2 or len(new_pos) != 2:
            raise ValueError("Positions must be 2-tuples")
        
        return MoveEvent.create_move_event(
            direction, old_pos, new_pos, apple_eaten, score_before, score_after, source
        )
    
    @staticmethod
    def create_apple_eaten_event(position: tuple, points: int = 1, 
                                source: str = "game_engine") -> AppleEvent:
        """Create an apple eaten event with validation."""
        if len(position) != 2:
            raise ValueError("Position must be a 2-tuple")
        
        if points <= 0:
            raise ValueError("Points must be positive")
        
        return AppleEvent.create_apple_eaten_event(position, points, source)
    
    @staticmethod
    def create_apple_spawned_event(position: tuple, 
                                  source: str = "game_engine") -> AppleEvent:
        """Create an apple spawned event with validation."""
        if len(position) != 2:
            raise ValueError("Position must be a 2-tuple")
        
        return AppleEvent.create_apple_spawned_event(position, source)
    
    @staticmethod
    def create_game_over_event(reason: str, final_score: int, total_moves: int,
                              duration: float, source: str = "game_engine") -> GameOverEvent:
        """Create a game over event with validation."""
        if not reason:
            raise ValueError("Reason cannot be empty")
        
        if final_score < 0 or total_moves < 0 or duration < 0:
            raise ValueError("Statistics must be non-negative")
        
        return GameOverEvent.create_game_over_event(
            reason, final_score, total_moves, duration, source
        ) 