"""
Observer Pattern Implementation
==============================

Observer interface and concrete implementations for handling game state changes.
Enables loose coupling between model, view, and controller components.

Design Patterns:
    - Observer Pattern: Core notification mechanism
    - Strategy Pattern: Different observer behaviors
    - Composite Pattern: Observer collections and hierarchies

Educational Goals:
    - Demonstrate Observer pattern implementation
    - Show how to decouple components with events
    - Illustrate interface-based design
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Optional
import logging
from datetime import datetime

from .events import GameEvent, EventType

# Configure logging
logger = logging.getLogger(__name__)


class Observer(ABC):
    """
    Abstract observer interface for game state changes.
    
    Implements the Observer pattern allowing components to be notified
    of game state changes without tight coupling to the game model.
    
    Design Pattern: Observer Pattern
    - Defines interface for receiving notifications
    - Enables loose coupling between publisher and subscribers
    - Allows multiple observers for same events
    """
    
    @abstractmethod
    def on_game_event(self, event: GameEvent) -> None:
        """
        Handle game state change event.
        
        Args:
            event: Game event containing state change information
        """
        pass
    
    def get_observer_id(self) -> str:
        """
        Get unique identifier for this observer.
        
        Returns:
            Unique string identifier
        """
        return f"{self.__class__.__name__}_{id(self)}"
    
    def is_interested_in_event(self, event: GameEvent) -> bool:
        """
        Check if observer is interested in this event type.
        
        Default implementation returns True for all events.
        Override in subclasses for filtering.
        
        Args:
            event: Game event to check
            
        Returns:
            True if observer should be notified of this event
        """
        return True


class EventTypeFilteredObserver(Observer):
    """
    Observer that filters events by type.
    
    Only receives notifications for specific event types.
    Useful for components that only care about certain types of changes.
    """
    
    def __init__(self, interested_event_types: Set[EventType]):
        """
        Initialize filtered observer.
        
        Args:
            interested_event_types: Set of event types to observe
        """
        self.interested_event_types = interested_event_types
    
    def is_interested_in_event(self, event: GameEvent) -> bool:
        """Check if event type is in our interest set."""
        return event.event_type in self.interested_event_types
    
    @abstractmethod
    def on_game_event(self, event: GameEvent) -> None:
        """Handle filtered game events."""
        pass


class LoggingObserver(Observer):
    """
    Observer that logs all game events.
    
    Useful for debugging and monitoring game state changes.
    Demonstrates simple observer implementation.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize logging observer with specified log level."""
        self.log_level = log_level
        self.event_count = 0
    
    def on_game_event(self, event: GameEvent) -> None:
        """Log the game event."""
        self.event_count += 1
        logger.log(
            self.log_level,
            f"Game Event #{self.event_count}: {event.event_type.value} "
            f"from {event.source} at {event.timestamp}"
        )
        
        # Log event-specific details
        if hasattr(event, 'direction'):
            logger.log(self.log_level, f"  Direction: {event.direction}")
        if hasattr(event, 'apple_eaten'):
            logger.log(self.log_level, f"  Apple eaten: {event.apple_eaten}")
        if hasattr(event, 'final_score'):
            logger.log(self.log_level, f"  Final score: {event.final_score}")


class StatisticsObserver(EventTypeFilteredObserver):
    """
    Observer that collects game statistics.
    
    Tracks various metrics about game performance and state changes.
    Demonstrates how observers can maintain their own state.
    """
    
    def __init__(self):
        # Only interested in specific event types for statistics
        interested_types = {
            EventType.MOVE,
            EventType.APPLE_EATEN,
            EventType.GAME_OVER,
            EventType.GAME_RESET
        }
        super().__init__(interested_types)
        
        # Statistics tracking
        self.total_moves = 0
        self.total_apples_eaten = 0
        self.total_games_played = 0
        self.total_score = 0
        self.game_start_time: Optional[datetime] = None
        self.longest_game_duration = 0.0
        self.highest_score = 0
        self.event_history: List[GameEvent] = []
    
    def on_game_event(self, event: GameEvent) -> None:
        """Update statistics based on game event."""
        # Store all events for analysis
        self.event_history.append(event)
        
        if event.event_type == EventType.MOVE:
            self.total_moves += 1
            if self.game_start_time is None:
                self.game_start_time = event.timestamp
        
        elif event.event_type == EventType.APPLE_EATEN:
            self.total_apples_eaten += 1
            if hasattr(event, 'points_awarded') and event.points_awarded:
                self.total_score += event.points_awarded
        
        elif event.event_type == EventType.GAME_OVER:
            self.total_games_played += 1
            
            if hasattr(event, 'final_score'):
                self.highest_score = max(self.highest_score, event.final_score)
            
            if hasattr(event, 'game_duration_seconds'):
                self.longest_game_duration = max(
                    self.longest_game_duration, 
                    event.game_duration_seconds
                )
        
        elif event.event_type == EventType.GAME_RESET:
            self.game_start_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        avg_moves_per_game = (
            self.total_moves / max(self.total_games_played, 1)
        )
        
        avg_apples_per_game = (
            self.total_apples_eaten / max(self.total_games_played, 1)
        )
        
        return {
            "total_moves": self.total_moves,
            "total_apples_eaten": self.total_apples_eaten,
            "total_games_played": self.total_games_played,
            "total_score": self.total_score,
            "highest_score": self.highest_score,
            "longest_game_duration": self.longest_game_duration,
            "average_moves_per_game": avg_moves_per_game,
            "average_apples_per_game": avg_apples_per_game,
            "total_events_recorded": len(self.event_history)
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics to initial state."""
        self.total_moves = 0
        self.total_apples_eaten = 0
        self.total_games_played = 0
        self.total_score = 0
        self.game_start_time = None
        self.longest_game_duration = 0.0
        self.highest_score = 0
        self.event_history.clear()


class CompositeObserver(Observer):
    """
    Observer that manages a collection of other observers.
    
    Implements the Composite pattern to allow treating multiple
    observers as a single observer. Useful for organizing observer
    hierarchies and broadcasting events to multiple handlers.
    
    Design Pattern: Composite Pattern
    - Treats individual observers and collections uniformly
    - Enables observer hierarchies and groupings
    - Simplifies observer management
    """
    
    def __init__(self, observers: List[Observer] = None):
        """
        Initialize composite observer.
        
        Args:
            observers: Initial list of observers to manage
        """
        self.observers: List[Observer] = observers or []
        self.event_broadcast_count = 0
    
    def add_observer(self, observer: Observer) -> None:
        """
        Add observer to the collection.
        
        Args:
            observer: Observer to add
        """
        if observer not in self.observers:
            self.observers.append(observer)
            logger.debug(f"Added observer {observer.get_observer_id()} to composite")
    
    def remove_observer(self, observer: Observer) -> bool:
        """
        Remove observer from the collection.
        
        Args:
            observer: Observer to remove
            
        Returns:
            True if observer was removed, False if not found
        """
        try:
            self.observers.remove(observer)
            logger.debug(f"Removed observer {observer.get_observer_id()} from composite")
            return True
        except ValueError:
            return False
    
    def on_game_event(self, event: GameEvent) -> None:
        """
        Broadcast event to all managed observers.
        
        Args:
            event: Game event to broadcast
        """
        self.event_broadcast_count += 1
        
        for observer in self.observers:
            try:
                if observer.is_interested_in_event(event):
                    observer.on_game_event(event)
            except Exception as e:
                logger.error(
                    f"Observer {observer.get_observer_id()} failed to handle event: {e}"
                )
    
    def get_observer_count(self) -> int:
        """Get number of managed observers."""
        return len(self.observers)
    
    def get_observer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all managed observers that support it."""
        stats = {}
        
        for observer in self.observers:
            observer_id = observer.get_observer_id()
            if hasattr(observer, 'get_statistics'):
                try:
                    stats[observer_id] = observer.get_statistics()
                except Exception as e:
                    stats[observer_id] = {"error": str(e)}
            else:
                stats[observer_id] = {"type": observer.__class__.__name__}
        
        return stats


class WebSocketObserver(Observer):
    """
    Observer that broadcasts events via WebSocket connections.
    
    Enables real-time updates to web clients by sending game events
    through WebSocket connections. Useful for live game monitoring
    and real-time dashboard updates.
    """
    
    def __init__(self, websocket_manager):
        """
        Initialize WebSocket observer.
        
        Args:
            websocket_manager: WebSocket connection manager
        """
        self.websocket_manager = websocket_manager
        self.events_sent = 0
        self.connection_errors = 0
    
    def on_game_event(self, event: GameEvent) -> None:
        """
        Broadcast game event via WebSocket.
        
        Args:
            event: Game event to broadcast
        """
        try:
            event_data = self._serialize_event_data(event)
            
            # Send to all connected clients
            if hasattr(self.websocket_manager, 'broadcast'):
                self.websocket_manager.broadcast(event_data)
                self.events_sent += 1
            else:
                logger.warning("WebSocket manager does not support broadcasting")
                
        except Exception as e:
            self.connection_errors += 1
            logger.error(f"Failed to broadcast event via WebSocket: {e}")
    
    def _serialize_event_data(self, event: GameEvent) -> Dict[str, Any]:
        """
        Serialize game event for WebSocket transmission.
        
        Args:
            event: Game event to serialize
            
        Returns:
            Serializable dictionary representation
        """
        return {
            "type": "game_event",
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
            "source": event.source,
            "data": event.data
        }
    
    def get_websocket_stats(self) -> Dict[str, Any]:
        """Get WebSocket observer statistics."""
        return {
            "events_sent": self.events_sent,
            "connection_errors": self.connection_errors,
            "has_websocket_manager": self.websocket_manager is not None
        } 