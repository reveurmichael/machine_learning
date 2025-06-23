"""
Human Game Controller - Interactive Gameplay
--------------------

Controller for human-driven Snake gameplay through web interface.
Handles user input, move validation, and real-time game state updates.

Design Patterns:
    - Template Method: Inherits request processing from base controller
    - Strategy: Different input validation strategies
    - Command: User inputs converted to game commands
    - Observer: Notifies observers of game events

Educational Goals:
    - Demonstrate role-based controller inheritance
    - Show input validation and command processing
    - Illustrate real-time web game interaction
"""

from typing import Dict, Any
import logging

from .base_controller import RequestContext
from .game_controllers import BaseGamePlayController, GameMode
from ..models import EventFactory
from utils.web_utils import translate_end_reason, build_color_map

logger = logging.getLogger(__name__)


class HumanGameController(BaseGamePlayController):
    """
    Controller for human player Snake game sessions.
    
    Naming convention reminder:
        • Task-0 concrete – lives in root namespace as *HumanGameController*.
        • Generic gameplay behaviour is inherited from `BaseGamePlayController`.
        • Extensions that require custom human input logic (e.g. VR input)
          should subclass the *base* under their own package.
    
    Extends BaseWebController to handle human input through web interface.
    Provides move validation, input processing, and real-time state updates.
    
    Design Patterns:
        - Template Method: Uses base controller request processing
        - Command: Converts user input to game commands
        - Strategy: Pluggable input validation
        - Observer: Generates events for game actions
    """
    
    VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}
    VALID_COMMANDS = {"pause", "resume", "restart", "reset"}
    
    def __init__(self, model_manager, view_renderer, **config):
        """
        Initialize human game controller.
        
        Args:
            model_manager: Game state model
            view_renderer: View rendering system
            **config: Controller configuration
        """
        super().__init__(model_manager, view_renderer, game_mode=GameMode.HUMAN_PLAY, **config)
        
        # Human-specific configuration
        self.input_validation_enabled = config.get('input_validation', True)
        self.move_history_size = config.get('move_history_size', 10)
        self.allow_reverse_moves = config.get('allow_reverse_moves', False)
        
        # Human gameplay state
        self.last_moves: list = []
        self.input_stats = {
            'total_moves': 0,
            'invalid_moves': 0,
            'rapid_inputs': 0,
            'last_input_time': 0.0
        }
        
        logger.info("Initialized HumanGameController")
    
    def _handle_gameplay_action(self, action: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle human-specific gameplay actions.
        
        Template Method Pattern: Implements abstract method from BaseGamePlayController.
        """
        # Handle human-specific actions not covered by base class
        if action == 'get_input_stats':
            return {
                'success': True,
                'input_stats': self.input_stats.copy()
            }
        elif action == 'get_tips':
            game_state = self.model_manager.get_current_state()
            return {
                'success': True,
                'tips': self._get_gameplay_tips(game_state)
            }
        else:
            return {
                'success': False,
                'message': f'Unknown human gameplay action: {action}'
            }
    
    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle /api/state requests for human gameplay.
        
        Returns current game state with human-specific enhancements
        like input statistics and move suggestions.
        """
        try:
            # Get base state from model
            game_state = self.model_manager.get_current_state()
            
            # Convert to web-friendly format
            state_dict = {
                "timestamp": game_state.timestamp,
                "score": game_state.score,
                "steps": game_state.steps,
                "game_over": game_state.game_over,
                "snake_positions": game_state.snake_positions,
                "apple_position": game_state.apple_position,
                "grid_size": game_state.grid_size,
                "direction": game_state.direction,
                "end_reason": translate_end_reason(game_state.end_reason) if game_state.end_reason else None,
                
                # Human-specific state additions
                "game_mode": "human",
                "controller_type": "human_player",
                "valid_directions": list(self.VALID_DIRECTIONS),
                "last_moves": self.last_moves[-5:],  # Last 5 moves
                "input_stats": self.input_stats.copy(),
                "gameplay_tips": self._get_gameplay_tips(game_state),
                
                # Colour palette injected for single-source-of-truth UI theming
                "colors": build_color_map(),
            }
            
            # Add performance metadata
            state_dict["metadata"] = {
                "state_source": "live_human_game",
                "controller_performance": self.get_performance_stats()
            }
            
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to get human game state: {e}")
            return {
                "error": f"State retrieval failed: {str(e)}",
                "game_mode": "human",
                "controller_type": "human_player"
            }
    
    def handle_control_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle /api/control requests for human player input.
        
        Processes move commands, validates input, and updates game state.
        """
        try:
            command = context.data.get("command", "").upper()
            
            if not command:
                return {
                    "status": "error",
                    "message": "Command parameter required",
                    "valid_commands": list(self.VALID_COMMANDS | self.VALID_DIRECTIONS)
                }
            
            # Handle movement commands
            if command in self.VALID_DIRECTIONS:
                return self._handle_move_command(command, context)
            
            # Handle game control commands
            elif command.lower() in self.VALID_COMMANDS:
                return self._handle_game_command(command.lower(), context)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown command: {command}",
                    "valid_commands": list(self.VALID_COMMANDS | self.VALID_DIRECTIONS)
                }
                
        except Exception as e:
            logger.error(f"Control request failed: {e}")
            return {
                "status": "error",
                "message": f"Command processing failed: {str(e)}"
            }
    
    def _handle_move_command(self, direction: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle player movement commands.
        
        Args:
            direction: Movement direction (UP/DOWN/LEFT/RIGHT)
            context: Request context
            
        Returns:
            Dictionary containing move result
        """
        import time
        
        # Input validation
        if self.input_validation_enabled:
            validation_result = self._validate_move_input(direction, context)
            if not validation_result["valid"]:
                self.input_stats['invalid_moves'] += 1
                return {
                    "status": "error",
                    "message": validation_result["reason"],
                    "input_stats": self.input_stats.copy()
                }
        
        # Check for rapid inputs
        current_time = time.time()
        if (self.input_stats['last_input_time'] > 0 and 
            current_time - self.input_stats['last_input_time'] < 0.1):
            self.input_stats['rapid_inputs'] += 1
            logger.warning(f"Rapid input detected from {context.client_ip}")
        
        self.input_stats['last_input_time'] = current_time
        
        try:
            # ------------------
            # Real integration with the *live* GameController
            # ------------------
            # Retrieve the active GameController from the state provider.
            state_provider = getattr(self.model_manager, "state_provider", None)
            game_controller = getattr(state_provider, "game_controller", None)

            if game_controller is None:
                raise RuntimeError("Live GameController not available in state provider")

            # Preserve the old head position for the response payload (useful
            # for front-end animations if ever required).
            old_position = (
                tuple(game_controller.head_position.tolist())
                if hasattr(game_controller, "head_position") else (0, 0)
            )

            # Execute the move via the core game logic.
            game_active, apple_eaten = game_controller.make_move(direction)

            # After mutating the controller we can query its *fresh* state via
            # the model – this guarantees we reflect whatever side-effects the
            # controller recorded (score increment, steps, etc.).
            new_state = self.model_manager.get_current_state()

            score_after = new_state.score
            score_before = score_after - (1 if apple_eaten else 0)

            new_position = (
                tuple(game_controller.head_position.tolist())
                if hasattr(game_controller, "head_position") else (0, 0)
            )

            # Update statistics
            self.input_stats['total_moves'] += 1
            self.last_moves.append({
                "direction": direction,
                "timestamp": current_time,
                "successful": True
            })
            
            # Trim move history
            if len(self.last_moves) > self.move_history_size:
                self.last_moves = self.last_moves[-self.move_history_size:]
            
            # Generate move event
            event = EventFactory.create_move_event(
                direction=direction,
                old_pos=old_position,
                new_pos=new_position,
                apple_eaten=apple_eaten,
                score_before=score_before,
                score_after=score_after,
                source="human_controller"
            )
            self.notify_observers(event)
            
            return {
                "status": "success" if game_active else "error",
                "message": "Move executed successfully",
                "old_position": old_position,
                "new_position": new_position,
                "apple_eaten": apple_eaten,
                "new_score": score_after,
                "game_active": game_active,
                "input_stats": self.input_stats.copy()
            }
            
        except Exception as exc:
            # Any failure gets bubbled up so the caller can surface it to the
            # UI.  We still return a structured error response rather than
            # raising – keeps the REST contract consistent.
            logger.error(f"GameController.make_move failed: {exc}")
            return {
                "status": "error",
                "message": f"Move execution failed: {exc}",
                "game_over": False,
                "input_stats": self.input_stats.copy()
            }
    
    def _handle_game_command(self, command: str, context: RequestContext) -> Dict[str, Any]:
        """
        Handle game control commands (pause, resume, restart, reset).
        
        Args:
            command: Game control command
            context: Request context
            
        Returns:
            Dictionary containing command result
        """
        try:
            if command == "reset" or command == "restart":
                success = self.model_manager.reset_game()
                if success:
                    # Reset controller state
                    self.last_moves.clear()
                    self.input_stats = {
                        'total_moves': 0,
                        'invalid_moves': 0,
                        'rapid_inputs': 0,
                        'last_input_time': 0.0
                    }
                    
                    return {
                        "status": "success",
                        "message": "Game reset successfully",
                        "action": command
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Game reset failed",
                        "action": command
                    }
            
            elif command == "pause":
                # Note: Pause functionality would be implemented in the game controller
                return {
                    "status": "success",
                    "message": "Game paused",
                    "action": command
                }
            
            elif command == "resume":
                # Note: Resume functionality would be implemented in the game controller
                return {
                    "status": "success",
                    "message": "Game resumed",
                    "action": command
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown game command: {command}"
                }
                
        except Exception as e:
            logger.error(f"Game command failed: {e}")
            return {
                "status": "error",
                "message": f"Command execution failed: {str(e)}"
            }
    
    def _validate_move_input(self, direction: str, context: RequestContext) -> Dict[str, Any]:
        """
        Validate player move input.
        
        Args:
            direction: Requested move direction
            context: Request context
            
        Returns:
            Dictionary with validation result
        """
        # Basic direction validation
        if direction not in self.VALID_DIRECTIONS:
            return {
                "valid": False,
                "reason": f"Invalid direction: {direction}"
            }
        
        # Check for reverse moves if disabled
        if not self.allow_reverse_moves and self.last_moves:
            last_move = self.last_moves[-1]
            opposite_directions = {
                "UP": "DOWN", "DOWN": "UP",
                "LEFT": "RIGHT", "RIGHT": "LEFT"
            }
            
            if (last_move["direction"] in opposite_directions and 
                opposite_directions[last_move["direction"]] == direction):
                return {
                    "valid": False,
                    "reason": "Reverse moves not allowed"
                }
        
        # Check game state
        try:
            current_state = self.model_manager.get_current_state()
            if current_state.game_over:
                return {
                    "valid": False,
                    "reason": "Game is over"
                }
        except Exception as e:
            logger.warning(f"Could not validate game state: {e}")
        
        return {"valid": True, "reason": None}
    
    def _get_gameplay_tips(self, game_state) -> list:
        """
        Generate helpful gameplay tips based on current state.
        
        Args:
            game_state: Current game state
            
        Returns:
            List of gameplay tips
        """
        tips = []
        
        if game_state.steps < 5:
            tips.append("Use arrow keys or WASD to move the snake")
        
        if game_state.score == 0:
            tips.append("Eat the red apple to grow and increase your score")
        
        if game_state.score > 0 and len(game_state.snake_positions) > 3:
            tips.append("Avoid hitting walls and your own body")
        
        if self.input_stats['rapid_inputs'] > 5:
            tips.append("Try to time your moves - rapid inputs may cause issues")
        
        if game_state.game_over:
            tips.append("Press Reset to start a new game")
        
        return tips
    
    def get_index_template_name(self) -> str:
        """Get template name for human game index page."""
        return "human_play.html"
    
    def get_index_template_context(self) -> Dict[str, Any]:
        """Get template context for human game page."""
        return {
            "controller_name": "Human Game Controller",
            "game_mode": "human",
            "valid_directions": list(self.VALID_DIRECTIONS),
            "features": [
                "Real-time keyboard controls",
                "Move validation",
                "Input statistics",
                "Gameplay tips",
                "Performance tracking"
            ]
        } 