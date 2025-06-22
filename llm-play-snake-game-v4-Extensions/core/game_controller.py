"""
Game Controller - Base Controller for Snake Game Applications
============================================================

This module provides the base controller classes that can be used by different
interfaces (GUI, Web, CLI) to interact with the game engine. It follows OOP
principles and the MVC pattern.

Design Patterns Used:
- Template Method: Base controller defines the algorithm, subclasses implement specifics
- Adapter Pattern: Adapts GameManager interface for different UI frameworks
- Strategy Pattern: Different execution strategies for different interfaces
"""

from __future__ import annotations
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from core.game_manager import GameManager

logger = logging.getLogger(__name__)


class BaseGameController(ABC):
    """
    Abstract base controller for Snake game applications.
    
    This class defines the common interface and shared functionality for all
    game controllers, regardless of the UI framework (PyGame, Web, CLI).
    
    Design Pattern: Template Method
    - Defines the skeleton of game control algorithms
    - Subclasses implement specific UI-related methods
    - Ensures consistent behavior across different interfaces
    
    Why Base prefix?
    - Signals this is a reusable foundation class
    - Task-0 will instantiate concrete subclasses
    - Extensions can inherit from this base
    """
    
    def __init__(self, game_manager: GameManager, use_gui: bool = True):
        """
        Initialize base controller with game manager.
        
        Args:
            game_manager: The core game manager instance
            use_gui: Whether this controller uses a GUI interface
        """
        self.game_manager = game_manager
        self.use_gui = use_gui
        self._start_time = time.time()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    # ==========================================
    # Template Method Pattern - Core Algorithm
    # ==========================================
    
    def run_game_session(self) -> None:
        """
        Template method for running a complete game session.
        
        This method defines the standard algorithm for game execution:
        1. Initialize the session
        2. Run the main game loop
        3. Handle cleanup
        
        Subclasses can override specific steps while maintaining the overall flow.
        """
        try:
            self.initialize_session()
            self.execute_main_loop()
        except Exception as e:
            self.handle_session_error(e)
        finally:
            self.cleanup_session()
    
    @abstractmethod
    def initialize_session(self) -> None:
        """Initialize the game session. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def execute_main_loop(self) -> None:
        """Execute the main game loop. Implemented by subclasses."""
        pass
    
    def handle_session_error(self, error: Exception) -> None:
        """Handle session-level errors. Can be overridden by subclasses."""
        logger.error(f"Game session error: {error}")
        raise
    
    def cleanup_session(self) -> None:
        """Clean up resources after session ends. Can be overridden by subclasses."""
        logger.info("Game session cleanup completed")
    
    # ==========================================
    # Shared Game State Interface
    # ==========================================
    
    @property
    def game(self):
        """Get the current game instance from GameManager."""
        return getattr(self.game_manager, 'game', None)
    
    @property
    def score(self) -> int:
        """Get current game score."""
        return getattr(self.game, 'score', 0) if self.game else 0
    
    @property
    def steps(self) -> int:
        """Get current step count."""
        return getattr(self.game, 'steps', 0) if self.game else 0
    
    @property
    def game_over(self) -> bool:
        """Check if game is over."""
        return not getattr(self.game_manager, 'game_active', True)
    
    @property
    def snake_positions(self) -> list:
        """Get snake body positions as JSON-serializable list."""
        game = self.game
        if game and hasattr(game, 'snake_positions'):
            # Convert numpy array to list for JSON serialization
            positions = game.snake_positions
            return getattr(positions, 'tolist', lambda: list(positions))()
        return []
    
    @property
    def apple_position(self) -> tuple:
        """Get apple position as JSON-serializable tuple."""
        game = self.game
        if game and hasattr(game, 'apple_position'):
            # Convert numpy array to tuple for JSON serialization
            position = game.apple_position
            return tuple(getattr(position, 'tolist', lambda: tuple(position))())
        return (0, 0)
    
    @property
    def current_direction(self) -> str:
        """Get current movement direction as string key ('UP', 'LEFT', etc.)."""
        game = self.game
        if game and hasattr(game, 'get_current_direction_key'):
            return game.get_current_direction_key()
        return "NONE"  # Default when game not initialized
    
    def get_current_direction_key(self) -> str:
        """
        Get current movement direction as a string key.
        
        This method is required by the web MVC framework for proper state updates.
        It delegates to the current_direction property to avoid code duplication.
        
        Design Pattern: Delegation
        - Delegates to the property to maintain single source of truth
        - Provides the exact interface expected by web framework
        - Avoids code duplication between property and method
        
        Returns:
            Current direction as string key or "NONE" if not available
        """
        return self.current_direction
    
    @property
    def end_reason(self) -> Optional[str]:
        """Get game end reason."""
        game = self.game
        if game and hasattr(game, 'game_state') and hasattr(game.game_state, 'game_end_reason'):
            return game.game_state.game_end_reason
        return None
    
    @property
    def start_time(self) -> float:
        """Get game start time."""
        return self._start_time
    
    @property
    def grid_size(self) -> int:
        """Get game grid size."""
        game = self.game
        if game and hasattr(game, 'grid_size'):
            return game.grid_size
        return 10  # Default fallback
    
    # ==========================================
    # Game Control Operations
    # ==========================================
    
    def reset_game(self) -> None:
        """
        Reset the game to initial state.
        
        This method provides a consistent interface for game reset across
        all controller types while delegating to the appropriate GameManager method.
        """
        try:
            if hasattr(self.game_manager, 'reset_game'):
                self.game_manager.reset_game()
            elif self.game and hasattr(self.game, 'reset'):
                self.game.reset()
            
            self._start_time = time.time()
            logger.info("Game reset via controller")
            
        except Exception as e:
            logger.error(f"Failed to reset game via controller: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        duration = time.time() - self._start_time
        return {
            "game_duration": duration,
            "moves_per_second": self.steps / max(duration, 1) if self.steps else 0.0,
            "controller_type": self.__class__.__name__
        }
    
    def get_game_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive game state for debugging/monitoring."""
        return {
            "score": self.score,
            "steps": self.steps,
            "game_over": self.game_over,
            "direction": self.current_direction,
            "snake_length": len(self.snake_positions),
            "apple_position": self.apple_position,
            "end_reason": self.end_reason,
            "performance": self.get_performance_stats()
        }


class GameControllerAdapter(BaseGameController):
    """
    Adapter to make GameManager compatible with web MVC architecture.
    
    This adapter follows the naming convention (no Task0 prefix since it's
    in the root and Task-0 is implicit). It provides the specific interface
    expected by the web MVC framework while inheriting common functionality.
    
    Design Pattern: Adapter Pattern
    - Adapts GameManager interface for MVC compatibility
    - Maintains Task-0 functionality while enabling MVC benefits
    - Provides clean integration point between game engine and web framework
    
    Why not Task0GameControllerAdapter?
    - Violates naming conventions (Task-0 is implicit in root)
    - Should be just GameControllerAdapter
    - Extensions will create HeuristicGameControllerAdapter, etc.
    """
    
    def __init__(self, game_manager: GameManager):
        """
        Initialize adapter for web MVC framework.
        
        Args:
            game_manager: Task-0 GameManager instance
        """
        # Web mode is strictly headless – no local PyGame window
        super().__init__(game_manager, use_gui=False)
        
        # IMPORTANT: During adapter construction the GameManager has **not**
        # yet run `setup_game()`, which means `game_manager.game` is *None*.
        # The base class properties handle this gracefully by checking for None.
        
        logger.info("Initialized GameControllerAdapter for web MVC")
    
    def initialize_session(self) -> None:
        """Initialize web session - handled by external web framework."""
        logger.info("Web session initialization delegated to web framework")
    
    def execute_main_loop(self) -> None:
        """Execute main loop - handled by external web framework."""
        logger.info("Main loop execution delegated to web framework")
    
    def make_move(self, direction: str) -> Tuple[bool, bool]:
        """
        Execute a move through the GameManager.
        
        For Task-0, moves are handled by the LLM agent. This method is a
        compatibility stub for the web framework, which expects a callable
        `make_move`. The actual game state is mutated by the `GameManager`
        in its background thread.
        
        Args:
            direction: Movement direction (not used in LLM mode)
            
        Returns:
            Tuple of (game_still_active, apple_eaten)
        """
        try:
            old_score = self.score
            
            # We don't execute a move here; just report current status
            game_active = not self.game_over
            apple_eaten = self.score > old_score
            
            return game_active, apple_eaten
            
        except Exception as e:
            logger.error(f"Move execution failed in adapter: {e}")
            return False, False


class CLIGameController(BaseGameController):
    """
    Controller for command-line interface game execution.
    
    This controller handles the traditional pygame-based or headless execution
    that was previously in main.py. It follows OOP principles while maintaining
    the same functionality.
    """
    
    def __init__(self, game_manager: GameManager, use_gui: bool = True):
        """
        Initialize CLI controller.
        
        Args:
            game_manager: Task-0 GameManager instance
            use_gui: Whether to show pygame GUI
        """
        super().__init__(game_manager, use_gui)
        logger.info(f"Initialized CLIGameController (GUI: {use_gui})")
    
    def initialize_session(self) -> None:
        """Initialize CLI session."""
        logger.info("Initializing CLI game session")
        # Additional CLI-specific initialization can go here
    
    def execute_main_loop(self) -> None:
        """Execute the main game loop for CLI mode."""
        logger.info("Starting CLI game execution")
        
        # Delegate to GameManager's run method
        # This maintains the existing behavior from main.py
        if hasattr(self.game_manager, 'run'):
            self.game_manager.run()
        else:
            logger.error("GameManager missing run() method")
            raise AttributeError("GameManager must implement run() method")