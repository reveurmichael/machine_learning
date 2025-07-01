"""
LLM Web Script - Full GameManager Integration
============================================

Script for launching LLM-controlled Snake game with full GameManager integration.
This provides complete LLM functionality, unlike the demo LLMWebApp in web module.

Design Philosophy:
- Full Functionality: Complete LLM gameplay with GameManager integration
- Task-0 Specific: Uses actual LLM providers and game logic
- Educational: Shows full Task-0 LLM architecture
- Extensible: Template for Tasks 1-5 LLM implementations

Note: This script provides full LLM functionality with GameManager integration.
The simplified LLMWebApp in the web module is just a demo interface.
For actual LLM gameplay, use this script.

Usage:
    python scripts/main_web.py                          # Default LLM settings
    python scripts/main_web.py --provider deepseek      # Different LLM provider  
    python scripts/main_web.py --model gpt-4 --port 8080  # Custom model and port
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Task-0 core components for full LLM functionality
from core.game_manager import GameManager
from web.base_app import GameFlaskApp
from utils.validation_utils import validate_grid_size, validate_port
from utils.web_utils import to_list, build_color_map
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE


class FullLLMWebApp(GameFlaskApp):
    """
    Full LLM web application with complete GameManager integration.
    
    Unlike the demo LLMWebApp, this provides complete LLM functionality
    with actual game manager, session management, and LLM providers.
    """
    
    def __init__(self, provider: str, model: str, grid_size: int = 10, port: int = None):
        """Initialize full LLM web app with GameManager."""
        super().__init__("LLM Snake Game (Full)", port)
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        
        # Create GameManager args for full LLM functionality
        import argparse
        args = argparse.Namespace(
            provider=provider,
            model=model,
            grid_size=grid_size,
            max_games=1,
            continue_from_folder=None,
            visualization=False,
            web_mode=True
        )
        
        # Initialize full GameManager for complete LLM functionality
        self.game_manager = GameManager(args)
        
        print(f"[FullLLMWebApp] Initialized with {provider}/{model}")
    
    def get_template_name(self) -> str:
        """Get template for LLM interface."""
        return 'main.html'
    
    def get_template_data(self) -> dict:
        """Get template data for LLM interface."""
        return {
            'name': self.name,
            'mode': 'llm_full',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size
        }
    
    def get_game_state(self) -> dict:
        """Get current game state from GameManager."""
        if not hasattr(self.game_manager, 'current_game_state'):
            return {
                'mode': 'llm_full',
                'provider': self.provider,
                'model': self.model,
                'grid_size': self.grid_size,
                'status': 'ready',
                'colors': build_color_map(as_list=True)
            }
        
        state = self.game_manager.current_game_state
        return {
            'mode': 'llm_full',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'snake_positions': to_list(state.get('snake_positions', [])),
            'apple_position': to_list(state.get('apple_position', [0, 0])),
            'score': state.get('score', 0),
            'steps': state.get('steps', 0),
            'game_over': state.get('game_over', False),
            'colors': build_color_map(as_list=True)
        }
    
    def handle_control(self, data: dict) -> dict:
        """Handle game controls with full GameManager."""
        action = data.get('action', '')
        
        if action == 'start':
            # Start new game with full LLM
            try:
                self.game_manager.run_single_game()
                return {'status': 'ok', 'message': 'Game started with LLM'}
            except Exception as e:
                return {'status': 'error', 'message': f'Failed to start game: {e}'}
        elif action == 'reset':
            # Reset game manager
            self.game_manager.reset_session()
            return {'status': 'ok', 'message': 'Game manager reset'}
        
        return {'status': 'error', 'message': 'Unknown action'}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snake Game - Full LLM Web Interface")
    
    parser.add_argument(
        "--provider",
        type=str,
        default="hunyuan",
        help="LLM provider (default: hunyuan)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuan-turbos-latest",
        help="LLM model (default: hunyuan-turbos-latest)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size (default: {DEFAULT_GRID_SIZE})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number (default: auto-detect)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for full LLM web interface."""
    try:
        print("[LLMWebFull] Starting Snake Game - Full LLM Web Interface")
        
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        grid_size = validate_grid_size(args.grid_size)
        port = validate_port(args.port) if args.port else None
        
        # Create full LLM app with GameManager integration
        app = FullLLMWebApp(
            provider=args.provider,
            model=args.model,
            grid_size=grid_size,
            port=port
        )
        
        print(f"[LLMWebFull] Server starting on {app.url}")
        print(f"[LLMWebFull] LLM Provider: {args.provider}/{args.model}")
        print("[LLMWebFull] This provides full LLM functionality with GameManager")
        print("[LLMWebFull] Press Ctrl+C to stop")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n[LLMWebFull] Server stopped by user")
    except Exception as e:
        print(f"[LLMWebFull] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 