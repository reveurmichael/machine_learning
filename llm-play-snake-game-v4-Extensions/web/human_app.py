"""
Human Web Game Application with Enhanced Naming
==============================================

Web application for human-controlled Snake game using existing GameLogic.
Provides Flask interface for human players with consistent game behavior.
Uses enhanced naming for maximum clarity and consistency.

Design Philosophy:
- SSOT: Uses existing GameLogic from core.game_logic for consistent behavior
- KISS: Simple web interface for human gameplay
- DRY: Reuses existing game infrastructure without duplication
- Enhanced Naming: Clear, explicit naming that indicates web + game domain
- Educational: Shows proper integration of web and game logic

Educational Value:
- Shows proper separation of web interface from game logic
- Demonstrates GameLogic integration matching human_play.py
- Provides template for extension web applications with enhanced naming

Extension Pattern:
Extensions can copy this pattern and replace GameLogic with their own
game logic classes while maintaining the same web interface structure.

Reference: web/base_app.py for layered web infrastructure
"""

from typing import Dict, Any

# Import Task-0 core components for proper game logic integration
from core.game_logic import GameLogic

# Import base application class from layered web infrastructure
from web.base_app import SimpleFlaskApp

# Import utilities following SSOT principles
from utils.web_utils import to_list, build_color_map
from utils.print_utils import create_logger

# Create logger for this module
print_log = create_logger("HumanWebGameApp")


class HumanWebGameApp(SimpleFlaskApp):
    """
    Human player web game application with enhanced naming.
    
    Uses the existing GameLogic class from core.game_logic for consistent 
    game behavior across all interfaces (GUI, web, CLI). Enhanced naming
    clearly indicates this is a web-based game application for human players.
    
    Design Pattern: Adapter Pattern (Web + Game Integration)
    Purpose: Adapts GameLogic interface for web API consumption with clear naming
    Educational Value: Shows proper separation of game logic from presentation
    Extension Pattern: Copy this enhanced pattern for any algorithm/model
    
    Enhanced Features:
    - Clear naming: HumanWebGameApp indicates web + game + human focus
    - Layered inheritance: Inherits from SimpleFlaskApp for game-specific features
    - GameLogic integration: Same behavior as human_play.py CLI interface
    - Educational clarity: Enhanced naming for better understanding
    """
    
    def __init__(self, grid_size: int = 10, **config):
        """Initialize human web game app with proper GameLogic integration.
        
        Args:
            grid_size: Size of the game grid
            **config: Additional configuration options
            
        Educational Value: Shows proper dependency injection and setup with enhanced naming
        Extension Pattern: Extensions can copy this initialization pattern
        """
        # Extract port if provided
        port = config.pop('port', None)
        super().__init__("Human Snake Web Game", port=port)
        self.grid_size = grid_size
        self.config = config
        
        # Use the existing GameLogic class - same as human_play.py
        # This ensures consistent game behavior across CLI, GUI, and web interfaces
        self.game = GameLogic(grid_size=grid_size, use_gui=False)
        
        print_log(f"Human web game mode: {grid_size}x{grid_size} grid")
        print_log("GameLogic integrated - same behavior as human_play.py CLI interface")
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get application data for template rendering.
        
        Returns:
            Dictionary with application data for HTML template
            
        Educational Value: Shows data preparation for template rendering with enhanced naming
        Extension Pattern: Extensions can copy this for their template data
        """
        return {
            'name': self.name,
            'mode': 'human_web',
            'grid_size': self.grid_size,
            'status': 'ready',
            'type': 'human_web_game_app'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get human game state from GameLogic instance.
        
        Returns:
            Dictionary with current game state for API consumption
            
        Educational Value: Shows how to extract state from GameLogic with enhanced structure
        Extension Pattern: Extensions can copy this state extraction pattern
        """
        return {
            'mode': 'human_web',
            'app_type': 'human_web_game_app',
            'grid_size': self.grid_size,
            'snake_positions': to_list(self.game.snake_positions),
            'apple_position': to_list(self.game.apple_position),
            'score': self.game.score,
            'steps': self.game.steps,
            'running': not self.game.game_over,
            'game_active': not self.game.game_over,
            'game_over': self.game.game_over,
            'status': 'ready',
            'colors': build_color_map(as_list=True)
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human web game controls using GameLogic methods.
        
        Args:
            data: Control data from client containing action and direction
            
        Returns:
            Response dictionary with operation result
            
        Educational Value: Shows proper delegation to GameLogic with enhanced context
        Extension Pattern: Extensions can copy this control handling pattern
        """
        action = data.get('action', '')
        direction = data.get('direction', '')
        
        if action == 'move' and direction:
            print_log(f"Human web game move: {direction}")
            if self.game.game_over:
                return {
                    'status': 'error', 
                    'message': 'Game is over',
                    'app_type': 'human_web_game_app'
                }
            
            # Use GameLogic.make_move() - same as human_play.py
            # This ensures identical behavior across all interfaces
            game_active, move_successful = self.game.make_move(direction.upper())
            
            if move_successful:
                return {
                    'status': 'ok',
                    'app_type': 'human_web_game_app',
                    'game_active': game_active,
                    'score': self.game.score,
                    'steps': self.game.steps
                }
            else:
                return {
                    'status': 'error', 
                    'message': 'Invalid move',
                    'app_type': 'human_web_game_app'
                }
                
        elif action == 'reset':
            print_log("Human web game reset")
            # Use GameLogic.reset() - same as human_play.py
            self.game.reset()
            return {
                'status': 'ok',
                'message': 'Game reset',
                'app_type': 'human_web_game_app',
                'score': self.game.score,
                'steps': self.game.steps
            }
        
        return {
            'status': 'error', 
            'message': 'Unknown action',
            'app_type': 'human_web_game_app'
        }
    
    def get_template_name(self) -> str:
        """Get template name for human web game interface.
        
        Returns:
            Template filename for human web game interface
            
        Educational Value: Shows template selection patterns with enhanced naming
        Extension Pattern: Extensions can override for custom templates
        """
        return 'human_play.html'
