"""
Game Controllers Base Classes - MVC Architecture
===============================================

Base controllers for different game modes following role-based inheritance.
Provides common functionality for gameplay and viewing controllers.

Design Patterns Used:
    - Template Method Pattern: Common request handling flow
    - Strategy Pattern: Different handling strategies for different modes
    - Chain of Responsibility: Request processing pipeline
    - Observer Pattern: Event-driven architecture

Controller Hierarchy & Naming Convention (Task-0 First-Citizen):
    BaseWebController                      – generic cross-mode behaviours

    # Interactive gameplay controllers
    ├── BaseGamePlayController             – **base** for *all* tasks (0-5)
    │   ├── GamePlayController             – Task-0 concrete LLM gameplay
    │   └── HumanGameController            – Task-0 concrete human gameplay
    
    # Passive viewing / replay controllers
    └── BaseGameViewingController          – **base** for *all* tasks (0-5)
        └── ReplayController               – Task-0 concrete replay viewer

Naming Rules enforced across the repo (see System-Prompt.txt):

    1.  If a class is **generic and meant to be reused** by Tasks 0-5 it lives
        in the root package and is prefixed with `Base…` (e.g.
        `BaseGamePlayController`).

    2.  Task-0 concrete implementations drop the prefix – they are the default
        in the root namespace (e.g. `GamePlayController`, `GameManager`,
        `GameGUI`).  They may contain LLM-specific logic.

    3.  Future extension tasks implement their own concrete subclasses inside
        `extensions/<task_name>/…` (e.g. `HeuristicGamePlayController`) that
        inherit from the *Base* classes – **never** from Task-0 classes unless
        explicitly documented.  This ensures Task-0 remains the
        first-citizen and no extension code pollutes the root.

    4.  Legacy names like `LLMGameController` have been retired.  Aliases
        linger only inside factories/tests for backward compatibility and will
        be removed after a deprecation window.

This docstring serves as a single canonical description of the naming scheme
for the web MVC layer.

Educational Goals:
    - Demonstrate role-based inheritance patterns
    - Show how abstract base classes define common behavior
    - Illustrate proper separation of concerns between different game modes

Naming convention reminder:
    • Generic *base* → **BaseGamePlayController** (this class)
    • Task-0 concrete → **GamePlayController** in `llm_controller.py`
    • Extension concretes live in `extensions/<task>/…`, e.g.
      `HeuristicGamePlayController`, and must subclass **this** base, *not*
      the Task-0 class.

Design Patterns:
    - Template Method: Defines common gameplay request handling
    - Strategy Pattern: Different play strategies (human vs AI)
    - Observer Pattern: Monitors game events for gameplay logic
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from enum import Enum

from .base_controller import BaseWebController, RequestContext
from ..models import GameStateModel, GameEvent
from ..views import WebViewRenderer
from utils.web_utils import build_color_map

logger = logging.getLogger(__name__)


class GameMode(Enum):
    """Enumeration of different game modes."""
    HUMAN_PLAY = "human_play"
    LLM_PLAY = "llm_play"
    REPLAY = "replay"  # Future experimental modes should live in extensions


class BaseGamePlayController(BaseWebController, ABC):
    """
    Abstract base controller for interactive gameplay modes.
    
    Provides common functionality for controllers that handle active game sessions
    where moves are being made and game state is actively changing.
    
    Naming convention reminder:
        • Generic *base* → **BaseGamePlayController** (this class)
        • Task-0 concrete → **GamePlayController** in `llm_controller.py`
        • Extension concretes live in `extensions/<task>/…`, e.g.
          `HeuristicGamePlayController`, and must subclass **this** base, *not*
          the Task-0 class.

    Design Patterns:
        - Template Method: Defines common gameplay request handling
        - Strategy Pattern: Different play strategies (human vs AI)
        - Observer Pattern: Monitors game events for gameplay logic
    """
    
    def __init__(self, model_manager: GameStateModel, view_renderer: WebViewRenderer, **kwargs):
        """
        Initialize gameplay controller.
        
        Args:
            model_manager: Game state model for data access
            view_renderer: View renderer for response generation
            **kwargs: Additional configuration options
        """
        super().__init__(model_manager, view_renderer, **kwargs)
        
        # Gameplay-specific configuration
        self.game_mode = kwargs.get('game_mode', GameMode.HUMAN_PLAY)
        self.allow_pause = kwargs.get('allow_pause', True)
        self.allow_reset = kwargs.get('allow_reset', True)
        self.track_statistics = kwargs.get('track_statistics', True)
        
        # Game state tracking
        self.is_paused = False
        self.game_start_time = None
        self.move_count = 0
        self.score = 0
        
        # Statistics tracking
        if self.track_statistics:
            self.statistics = {
                'games_played': 0,
                'total_moves': 0,
                'highest_score': 0,
                'average_score': 0,
                'total_play_time': 0
            }
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.game_mode.value} mode")
    
    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle state requests for gameplay sessions.
        
        Template Method Pattern: Extends base state handling with gameplay-specific info.
        """
        # Build the canonical state directly from the model (avoids the need
        # to rely on the abstract BaseWebController implementation).
        game_state = self.model_manager.get_current_state()

        base_state: Dict[str, Any] = {
            "timestamp": game_state.timestamp,
            "score": game_state.score,
            "steps": game_state.steps,
            "game_over": game_state.game_over,
            "snake_positions": game_state.snake_positions,
            "apple_position": game_state.apple_position,
            "grid_size": game_state.grid_size,
            "direction": game_state.direction,
            "end_reason": game_state.end_reason,
            # single-source colour palette
            "colors": build_color_map(),
        }

        # Add gameplay-specific state
        gameplay_state = {
            'game_mode': self.game_mode.value,
            'is_paused': self.is_paused,
            'move_count': self.move_count,
            'current_score': self.score,
            'gameplay_config': {
                'allow_pause': self.allow_pause,
                'allow_reset': self.allow_reset,
                'track_statistics': self.track_statistics
            }
        }
        
        # Add statistics if tracking is enabled
        if self.track_statistics:
            gameplay_state['statistics'] = self.statistics.copy()
        
        return {**base_state, **gameplay_state}
    
    def handle_control_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle control requests for gameplay sessions.
        
        Template Method Pattern: Processes gameplay-specific controls.
        """
        action = context.data.get('action', '')
        
        try:
            # Handle gameplay-specific actions
            if action == 'pause' and self.allow_pause:
                return self._handle_pause()
            elif action == 'resume' and self.allow_pause:
                return self._handle_resume()
            elif action == 'reset' and self.allow_reset:
                return self._handle_reset()
            elif action == 'move':
                return self._handle_move(context.data.get('direction', ''))
            elif action == 'get_statistics':
                return self._get_statistics()
            else:
                # Delegate to subclass for mode-specific handling
                return self._handle_gameplay_action(action, context)
                
        except Exception as e:
            logger.error(f"Error handling gameplay control: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action
            }
    
    @abstractmethod
    def _handle_gameplay_action(self, action: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle gameplay-specific actions.
        
        Template Method Pattern: Abstract method for subclass-specific behavior.
        """
        pass
    
    def _handle_pause(self) -> Dict[str, Any]:
        """Handle game pause request."""
        if self.is_paused:
            return {
                'success': False,
                'message': 'Game is already paused'
            }
        
        self.is_paused = True
        logger.info("Game paused")
        
        return {
            'success': True,
            'is_paused': True,
            'message': 'Game paused'
        }
    
    def _handle_resume(self) -> Dict[str, Any]:
        """Handle game resume request."""
        if not self.is_paused:
            return {
                'success': False,
                'message': 'Game is not paused'
            }
        
        self.is_paused = False
        logger.info("Game resumed")
        
        return {
            'success': True,
            'is_paused': False,
            'message': 'Game resumed'
        }
    
    def _handle_reset(self) -> Dict[str, Any]:
        """Handle game reset request."""
        # Update statistics before reset
        if self.track_statistics:
            self._update_game_statistics()
        
        # Reset game state
        self.move_count = 0
        self.score = 0
        self.is_paused = False
        self.game_start_time = None
        
        # Notify model to reset
        self.model_manager.reset_game()
        
        logger.info("Game reset")
        
        return {
            'success': True,
            'message': 'Game reset',
            'move_count': self.move_count,
            'score': self.score
        }
    
    def _handle_move(self, direction: str) -> Dict[str, Any]:
        """
        Handle move request.
        
        Template method that can be overridden by subclasses for specific move handling.
        """
        if self.is_paused:
            return {
                'success': False,
                'message': 'Cannot move while game is paused'
            }
        
        if direction not in ['up', 'down', 'left', 'right']:
            return {
                'success': False,
                'message': f'Invalid direction: {direction}'
            }
        
        # Execute move through model
        result = self.model_manager.make_move(direction)
        
        if result.get('success', False):
            self.move_count += 1
            self.score = result.get('score', self.score)
        
        return result
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get current game statistics."""
        if not self.track_statistics:
            return {
                'success': False,
                'message': 'Statistics tracking is disabled'
            }
        
        return {
            'success': True,
            'statistics': self.statistics.copy(),
            'current_session': {
                'move_count': self.move_count,
                'current_score': self.score,
                'is_paused': self.is_paused
            }
        }
    
    def _update_game_statistics(self):
        """Update statistics after a game ends."""
        if not self.track_statistics:
            return
        
        self.statistics['games_played'] += 1
        self.statistics['total_moves'] += self.move_count
        self.statistics['highest_score'] = max(self.statistics['highest_score'], self.score)
        
        # Calculate average score
        if self.statistics['games_played'] > 0:
            total_score = (self.statistics['average_score'] * (self.statistics['games_played'] - 1)) + self.score
            self.statistics['average_score'] = total_score / self.statistics['games_played']
    
    def on_event(self, event: GameEvent):
        """
        Handle game events for gameplay controllers.
        
        Observer Pattern: Responds to game state changes.
        """
        # Update score and move count based on events
        if event.event_type.value == 'move_made':
            self.move_count += 1
        elif event.event_type.value == 'apple_eaten':
            self.score += event.data.get('points', 10)
        elif event.event_type.value == 'game_over':
            self._update_game_statistics()


class BaseGameViewingController(BaseWebController, ABC):
    """
    Abstract base controller for viewing/replay modes.
    
    Provides common functionality for controllers that handle viewing game sessions
    without active gameplay (replays, demos, spectating).
    
    Naming convention reminder:
        • Generic *base* → **BaseGameViewingController** (this class)
        • Task-0 concrete → **ReplayController** in `replay_controller.py`.
        • Extensions provide their own viewer subclasses under
          `extensions/<task>/…`.

    Design Patterns:
        - Template Method: Defines common viewing request handling
        - Strategy Pattern: Different viewing strategies
        - Observer Pattern: Monitors replay events for analytics
    """
    
    def __init__(self, model_manager: GameStateModel, view_renderer: WebViewRenderer, **kwargs):
        """
        Initialize viewing controller.
        
        Args:
            model_manager: Game state model for data access
            view_renderer: View renderer for response generation
            **kwargs: Additional configuration options
        """
        super().__init__(model_manager, view_renderer, **kwargs)
        
        # Viewing-specific configuration
        self.viewing_mode = kwargs.get('viewing_mode', 'replay')
        self.allow_navigation = kwargs.get('allow_navigation', True)
        self.show_analytics = kwargs.get('show_analytics', True)
        
        # Viewing state
        self.current_position = 0
        self.total_positions = 0
        self.playback_speed = 1.0
        self.is_playing = False
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.viewing_mode} mode")
    
    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle state requests for viewing sessions.
        
        Template Method Pattern: Extends base state handling with viewing-specific info.
        """
        from utils.web_utils import build_color_map

        # Base snapshot of current state (even in viewing mode we maintain a
        # live `GameStateModel`, e.g. for replay playback position 0).
        game_state = self.model_manager.get_current_state()

        base_state: Dict[str, Any] = {
            "timestamp": game_state.timestamp,
            "score": game_state.score,
            "steps": game_state.steps,
            "game_over": game_state.game_over,
            "snake_positions": game_state.snake_positions,
            "apple_position": game_state.apple_position,
            "grid_size": game_state.grid_size,
            "direction": game_state.direction,
            "end_reason": game_state.end_reason,
            # Palette injection for SSoT UI theming
            "colors": build_color_map(),
        }

        # Add viewing-specific state
        viewing_state = {
            'viewing_mode': self.viewing_mode,
            'current_position': self.current_position,
            'total_positions': self.total_positions,
            'playback_speed': self.playback_speed,
            'is_playing': self.is_playing,
            'viewing_config': {
                'allow_navigation': self.allow_navigation,
                'show_analytics': self.show_analytics
            }
        }

        # Add analytics if enabled
        if self.show_analytics:
            viewing_state['analytics'] = self._get_analytics()

        return {**base_state, **viewing_state}
    
    def handle_control_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle control requests for viewing sessions.
        
        Template Method Pattern: Processes viewing-specific controls.
        """
        action = context.data.get('action', '')
        
        try:
            # Handle viewing-specific actions
            if action == 'play':
                return self._handle_play()
            elif action == 'pause':
                return self._handle_pause()
            elif action == 'seek':
                position = context.data.get('position', self.current_position)
                return self._handle_seek(position)
            elif action == 'set_speed':
                speed = context.data.get('speed', self.playback_speed)
                return self._handle_set_speed(speed)
            elif action == 'next':
                return self._handle_next()
            elif action == 'previous':
                return self._handle_previous()
            else:
                # Delegate to subclass for mode-specific handling
                return self._handle_viewing_action(action, context)
                
        except Exception as e:
            logger.error(f"Error handling viewing control: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action
            }
    
    @abstractmethod
    def _handle_viewing_action(self, action: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle viewing-specific actions.
        
        Template Method Pattern: Abstract method for subclass-specific behavior.
        """
        pass
    
    @abstractmethod
    def _get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics data for the viewing session.
        
        Template Method Pattern: Abstract method for subclass-specific analytics.
        """
        pass
    
    def _handle_play(self) -> Dict[str, Any]:
        """Handle play request."""
        self.is_playing = True
        logger.info("Playback started")
        
        return {
            'success': True,
            'is_playing': True,
            'message': 'Playback started'
        }
    
    def _handle_pause(self) -> Dict[str, Any]:
        """Handle pause request."""
        self.is_playing = False
        logger.info("Playback paused")
        
        return {
            'success': True,
            'is_playing': False,
            'message': 'Playback paused'
        }
    
    def _handle_seek(self, position: int) -> Dict[str, Any]:
        """Handle seek request."""
        if not self.allow_navigation:
            return {
                'success': False,
                'message': 'Navigation is disabled'
            }
        
        if position < 0 or position >= self.total_positions:
            return {
                'success': False,
                'message': f'Invalid position: {position}'
            }
        
        self.current_position = position
        logger.info(f"Seeked to position {position}")
        
        return {
            'success': True,
            'current_position': self.current_position,
            'message': f'Seeked to position {position}'
        }
    
    def _handle_set_speed(self, speed: float) -> Dict[str, Any]:
        """Handle playback speed change."""
        if speed <= 0 or speed > 10:
            return {
                'success': False,
                'message': 'Speed must be between 0 and 10'
            }
        
        self.playback_speed = speed
        logger.info(f"Playback speed set to {speed}x")
        
        return {
            'success': True,
            'playback_speed': self.playback_speed,
            'message': f'Playback speed set to {speed}x'
        }
    
    def _handle_next(self) -> Dict[str, Any]:
        """Handle next position request."""
        if not self.allow_navigation:
            return {
                'success': False,
                'message': 'Navigation is disabled'
            }
        
        if self.current_position >= self.total_positions - 1:
            return {
                'success': False,
                'message': 'Already at the last position'
            }
        
        self.current_position += 1
        
        return {
            'success': True,
            'current_position': self.current_position,
            'message': 'Moved to next position'
        }
    
    def _handle_previous(self) -> Dict[str, Any]:
        """Handle previous position request."""
        if not self.allow_navigation:
            return {
                'success': False,
                'message': 'Navigation is disabled'
            }
        
        if self.current_position <= 0:
            return {
                'success': False,
                'message': 'Already at the first position'
            }
        
        self.current_position -= 1
        
        return {
            'success': True,
            'current_position': self.current_position,
            'message': 'Moved to previous position'
        } 