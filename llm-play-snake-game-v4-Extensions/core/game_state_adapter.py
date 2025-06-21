"""
Game State Adapter for Consecutive Limits Manager
================================================

This module provides a clean adapter implementation that bridges the gap between
the game manager and the consecutive limits manager, following the Adapter Pattern
to provide a consistent interface while avoiding code duplication.

Design Philosophy:
- Single Responsibility: Focused solely on adapting game manager interface
- Adapter Pattern: Provides the interface expected by limits manager
- DRY Principle: Eliminates duplicate adapter code across the codebase
- Loose Coupling: Reduces dependencies between components

Educational Value:
This demonstrates how the Adapter Pattern can be used to integrate different
components with incompatible interfaces while maintaining clean architecture.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.game_manager import BaseGameManager


class GameStateAdapter:
    """
    Adapter to provide game state interface to the consecutive limits manager.
    
    This class implements the Adapter Pattern to bridge the gap between the
    game manager's interface and the interface expected by the ConsecutiveLimitsManager.
    
    Key Features:
    - Clean separation of concerns
    - Consistent interface for limits management
    - Proper error handling and state management
    - Support for both active and inactive game states
    
    Design Patterns:
    - Adapter Pattern: Adapts game manager interface for limits manager
    - Facade Pattern: Simplifies complex game state operations
    """
    
    def __init__(self, game_manager: "BaseGameManager", 
                 override_game_active: Optional[bool] = None):
        """
        Initialize the game state adapter.
        
        Args:
            game_manager: The game manager instance to adapt
            override_game_active: Optional override for game active state
                                 (useful when game state changes during move execution)
        """
        self.game_manager = game_manager
        self.override_game_active = override_game_active
    
    def record_game_end(self, reason: str) -> None:
        """
        Record the end of a game with the specified reason.
        
        This method handles the complete game termination process:
        1. Sets the game as inactive
        2. Records the collision type for debugging
        3. Updates the game state with the end reason
        
        Args:
            reason: The reason why the game ended (e.g., "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")
        """
        # Set game as inactive
        self.game_manager.game_active = False
        
        # Record collision type for debugging and statistics
        if hasattr(self.game_manager, 'game') and hasattr(self.game_manager.game, 'last_collision_type'):
            self.game_manager.game.last_collision_type = reason
        
        # Record the game end in the game state for proper logging
        if (hasattr(self.game_manager, 'game') and 
            hasattr(self.game_manager.game, 'game_state') and 
            hasattr(self.game_manager.game.game_state, 'record_game_end')):
            self.game_manager.game.game_state.record_game_end(reason)
    
    def is_game_active(self) -> bool:
        """
        Check if the game is currently active.
        
        This method provides a consistent way to check game state across
        different contexts, with support for temporary state overrides.
        
        Returns:
            True if the game is active, False otherwise
        """
        # Use override if provided (useful during move execution)
        if self.override_game_active is not None:
            return self.override_game_active
        
        # Check the game manager's game_active flag
        return getattr(self.game_manager, 'game_active', True)
    
    def get_game_info(self) -> dict:
        """
        Get comprehensive game information for debugging and logging.
        
        Returns:
            Dictionary containing current game state information
        """
        info = {
            "game_active": self.is_game_active(),
            "game_count": getattr(self.game_manager, 'game_count', 0),
            "round_count": getattr(self.game_manager, 'round_count', 1),
        }
        
        # Add game-specific information if available
        if hasattr(self.game_manager, 'game') and self.game_manager.game:
            game = self.game_manager.game
            info.update({
                "score": getattr(game, 'score', 0),
                "steps": getattr(game, 'steps', 0),
                "snake_length": len(getattr(game, 'snake_positions', [])),
            })
        
        return info
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"GameStateAdapter(active={self.is_game_active()}, game_count={getattr(self.game_manager, 'game_count', 0)})"


def create_game_state_adapter(game_manager: "BaseGameManager", 
                             override_game_active: Optional[bool] = None) -> GameStateAdapter:
    """
    Factory function for creating game state adapters.
    
    This function implements the Factory Pattern to provide a clean way
    to create properly configured game state adapters.
    
    Args:
        game_manager: The game manager instance to adapt
        override_game_active: Optional override for game active state
        
    Returns:
        Configured GameStateAdapter instance
    """
    return GameStateAdapter(game_manager, override_game_active) 