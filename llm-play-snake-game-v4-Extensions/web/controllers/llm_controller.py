"""
LLM Game Controller - MVC Architecture
=====================================

Controller for LLM-driven Snake game sessions.
Handles AI decision making, game state analysis, and automated gameplay.

Design Patterns Used:
    - Template Method: Inherits request handling flow from BaseWebController
    - Strategy Pattern: Different LLM providers can be plugged in
    - Command Pattern: LLM decisions converted to game commands
    - Observer Pattern: Monitors game state changes for decision making

Educational Goals:
    - Show how controllers can orchestrate complex AI workflows
    - Demonstrate separation of AI logic from web request handling
    - Illustrate how inheritance enables code reuse while allowing specialization
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .base_controller import RequestType, RequestContext
from .game_controllers import BaseGamePlayController, GameMode
from ..models import GameStateModel, GameEvent
from ..views import WebViewRenderer

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUNYUAN = "hunyuan"
    LOCAL = "local"


@dataclass
class LLMDecision:
    """
    Represents an LLM's decision for the next game move.
    
    Immutable Object Pattern: Ensures decision integrity.
    """
    move: str
    confidence: float
    reasoning: str
    processing_time: float
    provider: LLMProvider
    
    def __post_init__(self):
        """Validate decision parameters."""
        if self.move not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid move: {self.move}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")


class LLMStrategy:
    """
    Abstract base class for LLM decision strategies.
    
    Strategy Pattern: Allows different LLM implementations to be used interchangeably.
    """
    
    def make_decision(self, game_state: Dict[str, Any]) -> LLMDecision:
        """
        Make a decision based on current game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            LLMDecision with move and metadata
        """
        raise NotImplementedError("Subclasses must implement make_decision")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the LLM provider."""
        raise NotImplementedError("Subclasses must implement get_provider_info")


class MockLLMStrategy(LLMStrategy):
    """
    Mock LLM strategy for testing and demonstration.
    
    Implements simple heuristic-based decision making for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize mock strategy."""
        self.provider = LLMProvider.LOCAL
        self.decision_count = 0
    
    def make_decision(self, game_state: Dict[str, Any]) -> LLMDecision:
        """
        Make a simple heuristic-based decision.
        
        This is a simplified decision maker for demonstration.
        Real implementations would use actual LLM APIs.
        """
        start_time = time.time()
        
        # Simple heuristic: avoid walls and try to reach food
        snake_head = game_state.get('snake_head', [10, 10])
        food_pos = game_state.get('food_position', [15, 15])
        
        # Calculate direction to food
        dx = food_pos[0] - snake_head[0]
        dy = food_pos[1] - snake_head[1]
        
        # Simple decision logic
        if abs(dx) > abs(dy):
            move = 'right' if dx > 0 else 'left'
        else:
            move = 'down' if dy > 0 else 'up'
        
        processing_time = time.time() - start_time
        self.decision_count += 1
        
        return LLMDecision(
            move=move,
            confidence=0.7,
            reasoning=f"Moving towards food at {food_pos}",
            processing_time=processing_time,
            provider=self.provider
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get mock provider information."""
        return {
            'provider': self.provider.value,
            'model': 'mock-heuristic-v1',
            'decisions_made': self.decision_count,
            'average_response_time': 0.01
        }


class GamePlayController(BaseGamePlayController):
    """
    Controller for LLM-driven Snake game sessions.
    
    ── Naming convention ──────────────────────────────────────────────
    • This class is the **Task-0 concrete** gameplay controller and therefore
      keeps the terse name *GamePlayController*.
    • Generic/shared behaviour lives in `BaseGamePlayController`.
    • Extensions must create their own concrete subclasses (e.g.
      `HeuristicGamePlayController`) **inside the `extensions/` tree** and
      inherit from the *base* – never from this Task-0 implementation.
    • The old name `LLMGameController` has been **removed**; importing it
      will raise `ImportError`.

    Orchestrates AI decision making with web interface requirements.
    Inherits request handling framework from BaseGamePlayController.
    
    Design Patterns:
        - Template Method: Uses parent's request handling flow
        - Strategy Pattern: Pluggable LLM providers
        - Observer Pattern: Monitors game events for decision triggers
        - Command Pattern: Converts LLM decisions to game commands
    """
    
    def __init__(self, model_manager: GameStateModel, view_renderer: WebViewRenderer,
                 llm_strategy: Optional[LLMStrategy] = None, **kwargs):
        """
        Initialize LLM game controller.
        
        Args:
            model_manager: Game state model for data access
            view_renderer: View renderer for response generation
            llm_strategy: LLM decision strategy (defaults to MockLLMStrategy)
            **kwargs: Additional configuration options
        """
        super().__init__(model_manager, view_renderer, game_mode=GameMode.LLM_PLAY, **kwargs)
        
        # Initialize LLM strategy (Strategy Pattern)
        self.llm_strategy = llm_strategy or MockLLMStrategy()
        
        # LLM-specific configuration
        self.auto_play_enabled = kwargs.get('auto_play_enabled', True)
        self.decision_delay = kwargs.get('decision_delay', 1.0)  # seconds
        self.max_thinking_time = kwargs.get('max_thinking_time', 5.0)
        
        # Performance tracking
        self.decision_history: List[LLMDecision] = []
        self.total_decisions = 0
        self.successful_decisions = 0
        
        # Register as observer for game events
        self.model_manager.add_observer(self)
        
        logger.info(f"Initialized LLM controller with {self.llm_strategy.__class__.__name__}")
    
    def _handle_gameplay_action(self, action: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle LLM-specific gameplay actions.
        
        Template Method Pattern: Implements abstract method from BaseGamePlayController.
        """
        try:
            if action == 'toggle_auto_play':
                return self._toggle_auto_play()
            elif action == 'make_decision':
                return self._make_llm_decision()
            elif action == 'set_decision_delay':
                delay = float(context.data.get('delay', self.decision_delay))
                return self._set_decision_delay(delay)
            elif action == 'get_llm_info':
                return {'llm_info': self.llm_strategy.get_provider_info()}
            else:
                return {
                    'success': False,
                    'message': f'Unknown LLM gameplay action: {action}'
                }
                
        except Exception as e:
            logger.error(f"Error handling LLM gameplay action {action}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action
            }
    
    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle requests for game state information.
        
        Template Method Pattern: Implements abstract method from parent.
        Adds LLM-specific state information to base game state.
        """
        # Get base game state
        base_state = super().handle_state_request(context)
        
        # Add LLM-specific information
        llm_state = {
            'llm_info': self.llm_strategy.get_provider_info(),
            'auto_play_enabled': self.auto_play_enabled,
            'decision_delay': self.decision_delay,
            'performance_stats': {
                'total_decisions': self.total_decisions,
                'successful_decisions': self.successful_decisions,
                'success_rate': self.successful_decisions / max(1, self.total_decisions),
                'recent_decisions': [
                    {
                        'move': d.move,
                        'confidence': d.confidence,
                        'reasoning': d.reasoning,
                        'processing_time': d.processing_time
                    }
                    for d in self.decision_history[-5:]  # Last 5 decisions
                ]
            }
        }
        
        # Merge states
        return {**base_state, **llm_state}
    

    
    def _toggle_auto_play(self) -> Dict[str, Any]:
        """Toggle automatic LLM gameplay."""
        self.auto_play_enabled = not self.auto_play_enabled
        
        logger.info(f"Auto-play {'enabled' if self.auto_play_enabled else 'disabled'}")
        
        return {
            'success': True,
            'auto_play_enabled': self.auto_play_enabled,
            'message': f"Auto-play {'enabled' if self.auto_play_enabled else 'disabled'}"
        }
    
    def _make_llm_decision(self) -> Dict[str, Any]:
        """
        Make an LLM decision for the current game state.
        
        Command Pattern: Converts LLM decision into game command.
        """
        try:
            # Get current game state
            game_state = self.model_manager.get_current_state()
            
            # Make LLM decision
            decision = self.llm_strategy.make_decision(game_state.__dict__)
            
            # Record decision
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:  # Keep last 100 decisions
                self.decision_history.pop(0)
            
            self.total_decisions += 1
            
            # Execute decision (convert to game command)
            move_result = self._execute_move(decision.move)
            
            if move_result.get('success', False):
                self.successful_decisions += 1
            
            logger.debug(f"LLM decision: {decision.move} (confidence: {decision.confidence:.2f})")
            
            return {
                'success': True,
                'decision': {
                    'move': decision.move,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'processing_time': decision.processing_time
                },
                'move_result': move_result
            }
            
        except Exception as e:
            logger.error(f"Error making LLM decision: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_move(self, move: str) -> Dict[str, Any]:
        """
        Execute a move command in the game.
        
        Args:
            move: Direction to move ('up', 'down', 'left', 'right')
            
        Returns:
            Result of the move execution
        """
        # Create move command context
        move_context = RequestContext(
            request_type=RequestType.CONTROL,
            data={'action': 'move', 'direction': move},
            client_info={'source': 'llm_decision'}
        )
        
        # Execute through parent's control handling
        return super().handle_control_request(move_context)
    
    def _set_decision_delay(self, delay: float) -> Dict[str, Any]:
        """Set the delay between automatic decisions."""
        if delay < 0.1 or delay > 10.0:
            return {
                'success': False,
                'error': 'Decision delay must be between 0.1 and 10.0 seconds'
            }
        
        self.decision_delay = delay
        logger.info(f"Decision delay set to {delay} seconds")
        
        return {
            'success': True,
            'decision_delay': self.decision_delay,
            'message': f"Decision delay set to {delay} seconds"
        }
    
    def on_event(self, event: GameEvent):
        """
        Handle game events for automatic decision making.
        
        Observer Pattern: Responds to game state changes.
        """
        # Auto-play logic: make decision when game state changes
        if (self.auto_play_enabled and 
            event.event_type.value in ['move_made', 'apple_eaten', 'game_started']):
            
            # Add small delay to prevent rapid-fire decisions
            time.sleep(self.decision_delay)
            
            # Make automatic decision
            self._make_llm_decision()
    
    def get_controller_info(self) -> Dict[str, Any]:
        """
        Get comprehensive controller information.
        
        Returns detailed information about the LLM controller state.
        """
        base_info = super().get_controller_info()
        
        llm_info = {
            'controller_type': 'GamePlayController',
            'llm_provider': self.llm_strategy.get_provider_info(),
            'auto_play_enabled': self.auto_play_enabled,
            'decision_delay': self.decision_delay,
            'performance_metrics': {
                'total_decisions': self.total_decisions,
                'successful_decisions': self.successful_decisions,
                'success_rate': self.successful_decisions / max(1, self.total_decisions),
                'average_confidence': sum(d.confidence for d in self.decision_history[-10:]) / len(self.decision_history[-10:]) if self.decision_history else 0,
                'average_processing_time': sum(d.processing_time for d in self.decision_history[-10:]) / len(self.decision_history[-10:]) if self.decision_history else 0
            }
        }
        
        return {**base_info, **llm_info} 