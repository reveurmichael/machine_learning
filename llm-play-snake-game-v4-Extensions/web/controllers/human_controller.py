"""
Human Game Controller - Interactive Gameplay
===========================================

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
from .game_controllers import GamePlayController, GameMode
from ..models import EventFactory
from utils.web_utils import translate_end_reason

logger = logging.getLogger(__name__)


class HumanGameController(GamePlayController):
    """
    Controller for human player Snake game sessions.
    
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
        
        Template Method Pattern: Implements abstract method from GamePlayController.
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
                "gameplay_tips": self._get_gameplay_tips(game_state)
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
            # Get current state before move
            old_state = self.model_manager.get_current_state()
            
            # Execute move through game controller
            # Note: This would integrate with the actual game controller
            # For now, we'll simulate the move execution
            move_result = self._execute_move(direction, old_state)
            
            # Update statistics
            self.input_stats['total_moves'] += 1
            self.last_moves.append({
                "direction": direction,
                "timestamp": current_time,
                "successful": move_result["success"]
            })
            
            # Trim move history
            if len(self.last_moves) > self.move_history_size:
                self.last_moves = self.last_moves[-self.move_history_size:]
            
            # Generate move event
            if move_result["success"]:
                event = EventFactory.create_move_event(
                    direction=direction,
                    old_pos=move_result.get("old_position", (0, 0)),
                    new_pos=move_result.get("new_position", (0, 0)),
                    apple_eaten=move_result.get("apple_eaten", False),
                    score_before=old_state.score,
                    score_after=move_result.get("new_score", old_state.score),
                    source="human_controller"
                )
                self.notify_observers(event)
            
            return {
                "status": "success" if move_result["success"] else "error",
                "message": move_result.get("message", "Move executed"),
                "direction": direction,
                "game_active": not move_result.get("game_over", False),
                "apple_eaten": move_result.get("apple_eaten", False),
                "score": move_result.get("new_score", old_state.score),
                "input_stats": self.input_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Move execution failed: {e}")
            self.input_stats['invalid_moves'] += 1
            return {
                "status": "error",
                "message": f"Move execution failed: {str(e)}",
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
    
    def _execute_move(self, direction: str, current_state) -> Dict[str, Any]:
        """
        Execute move through game controller.
        
        Note: This is a simplified implementation. In a real system,
        this would interface with the actual GameController.
        
        Args:
            direction: Move direction
            current_state: Current game state
            
        Returns:
            Dictionary containing move execution result
        """
        # This would interface with the actual game controller
        # For demonstration, we'll return a mock result
        return {
            "success": True,
            "message": "Move executed successfully",
            "old_position": current_state.snake_positions[0] if current_state.snake_positions else (0, 0),
            "new_position": (5, 5),  # Mock new position
            "apple_eaten": False,  # Mock apple eating
            "new_score": current_state.score,
            "game_over": False
        }
    
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