"""
LLM Web Game Application - Simple Flask Interface
================================================

Simple Flask web interface for LLM-controlled Snake game.
Provides lightweight demo interface for LLM gameplay.

Design Philosophy:
- KISS: Simple, direct implementation
- No Over-Preparation: Basic demo interface only
- Extensible: Easy for Tasks 1-5 to copy and modify

Note: For full LLM functionality with GameManager integration,
use scripts/main_web.py which provides complete session management.
This provides a lightweight demo interface.
"""

from typing import Dict, Any

from web.base_app import GameFlaskApp
from utils.web_utils import to_list, build_color_map


class LLMWebApp(GameFlaskApp):
    """
    Simple Flask web app for LLM Snake game demo.
    
    Provides basic web interface for LLM gameplay demonstration.
    For full functionality, use scripts/main_web.py instead.
    """
    
    def __init__(self, provider: str = "hunyuan", model: str = "hunyuan-turbos-latest", 
                 grid_size: int = 10, port: int = None):
        """Initialize LLM web app."""
        super().__init__("LLM Snake Game", port)
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        
        # Simple demo state (not full LLM integration)
        self.demo_state = {
            'snake_positions': [(5, 5)],
            'apple_position': (7, 7),
            'score': 0,
            'steps': 0,
            'game_over': False
        }
        
        print(f"[LLMWebApp] Demo mode: {provider}/{model}")
    
    def get_template_name(self) -> str:
        """Get template for LLM gameplay."""
        return 'main.html'
    
    def get_template_data(self) -> Dict[str, Any]:
        """Get template data for LLM gameplay."""
        return {
            'name': self.name,
            'mode': 'llm',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size
        }
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current demo game state."""
        return {
            'mode': 'llm_demo',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'snake_positions': self.demo_state['snake_positions'],
            'apple_position': self.demo_state['apple_position'],
            'score': self.demo_state['score'],
            'steps': self.demo_state['steps'],
            'game_over': self.demo_state['game_over'],
            'colors': build_color_map(as_list=True)
        }
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle demo move."""
        return {'status': 'demo', 'message': 'Use scripts/main_web.py for full LLM functionality'}
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle demo control."""
        return {'status': 'demo', 'message': 'Use scripts/main_web.py for full LLM functionality'}
    
    def handle_reset(self) -> Dict[str, Any]:
        """Handle demo reset."""
        self.demo_state = {
            'snake_positions': [(5, 5)],
            'apple_position': (7, 7),
            'score': 0,
            'steps': 0,
            'game_over': False
        }
        return {'status': 'ok', 'message': 'Demo state reset'}

