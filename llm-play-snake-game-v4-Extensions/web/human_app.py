"""
Human Web Game Application - Simple Flask Interface
==================================================

Simple Flask web interface for human-controlled Snake game.
Uses existing GameLogic from core for consistent behavior.

Design Philosophy:
- KISS: Simple, direct implementation
- DRY: Reuses existing GameLogic from core
- No Over-Preparation: Only implements what's needed now
- Extensible: Easy for Tasks 1-5 to copy and modify

Educational Value:
- Shows proper integration of Flask with game logic
- Demonstrates simple web API design
- Provides template for extension web interfaces
"""

from typing import Dict, Any

from core.game_logic import GameLogic
from web.base_app import GameFlaskApp
from utils.web_utils import to_list, build_color_map, translate_end_reason


class HumanWebApp(GameFlaskApp):
    """
    Simple Flask web app for human Snake game players.
    
    Uses existing GameLogic class for consistent game behavior.
    Provides web API for human players via browser interface.
    """
    
    def __init__(self, grid_size: int = 10, port: int = None):
        """Initialize human web app."""
        super().__init__("Human Snake Game", port)
        self.grid_size = grid_size
        
        # Use existing GameLogic - same as CLI human_play.py
        self.game = GameLogic(grid_size=grid_size, use_gui=False)
        
        print(f"[HumanWebApp] {grid_size}x{grid_size} grid ready")
    
    def get_template_name(self) -> str:
        """Get template for human gameplay."""
        return 'human_play.html'
    
    def get_template_data(self) -> Dict[str, Any]:
        """Get template data for human gameplay."""
        return {
            'name': self.name,
            'mode': 'human',
            'grid_size': self.grid_size
        }
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state for web API."""
        return {
            'mode': 'human',
            'grid_size': self.grid_size,
            'snake_positions': to_list(self.game.snake_positions),
            'apple_position': to_list(self.game.apple_position),
            'score': self.game.score,
            'steps': self.game.steps,
            'game_over': self.game.game_over,
            'game_end_reason': translate_end_reason(self.game.game_state.game_end_reason),
            'colors': build_color_map(as_list=True)
        }
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move from web interface."""
        direction = data.get('direction', '')
        if not direction:
            return {'status': 'error', 'message': 'No direction provided'}
        
        if self.game.game_over:
            return {'status': 'error', 'message': 'Game is over'}
        
        # Use GameLogic.make_move() - same as CLI interface
        game_active, move_successful = self.game.make_move(direction.upper())
        
        if move_successful:
            return {
                'status': 'ok',
                'game_active': game_active,
                'score': self.game.score,
                'steps': self.game.steps
            }
        else:
            return {'status': 'error', 'message': 'Invalid move'}
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control commands from web interface."""
        action = data.get('action', '')
        
        if action == 'move':
            return self.handle_move(data)
        elif action == 'reset':
            return self.handle_reset()
        else:
            return {'status': 'error', 'message': 'Unknown action'}
    
    def handle_reset(self) -> Dict[str, Any]:
        """Reset the game."""
        self.game.reset()
        return {
            'status': 'ok',
            'message': 'Game reset',
            'score': self.game.score,
            'steps': self.game.steps
        }
