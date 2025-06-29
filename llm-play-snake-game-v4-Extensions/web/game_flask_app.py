"""
Task-0 Flask Web Interface - Minimal KISS Architecture
--------------------

Truly minimal, KISS, DRY, and extensible web backend for Task-0 and all future extensions.
Direct integration with existing game logic without complex patterns or Task-0 pollution.

Design Philosophy:
- KISS: Keep It Simple, Stupid - no over-engineering
- DRY: Don't Repeat Yourself - reusable patterns
- Minimal: Only essential functionality
- Extensible: Easy to extend for future tasks

Educational Goals:
- Show direct Flask integration with existing components
- Demonstrate minimal web architecture
- Provide copy-paste templates for extensions
- Keep functionality while removing complexity

Extension Pattern:
Copy this file → Replace game components → Maintain same simple structure
"""

import os
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request

# Import utilities (no Task-0 pollution)
from utils.network_utils import random_free_port

# Simple logging following SUPREME_RULES
print_log = lambda msg: print(f"[WebApp] {msg}")


class SimpleFlaskApp:
    """
    Minimal Flask application foundation.
    
    Design Pattern: Template Method Pattern (Simple Lifecycle)
    Purpose: Provides minimal web interface foundation
    Educational Value: Shows KISS principles in web applications
    Extension Pattern: Copy and modify for any task
    """
    
    def __init__(self, name: str = "SnakeGame", port: Optional[int] = None):
        """
        Initialize minimal Flask application.
        
        Args:
            name: Application name
            port: Port number (auto-assigned if None)
        """
        self.name = name
        self.port = port or random_free_port()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.configure_app()
        self.setup_routes()
        
        print_log(f"Initialized {name} on port {self.port}")
    
    def configure_app(self) -> None:
        """Configure Flask application with minimal settings."""
        self.app.config.update({
            'SECRET_KEY': os.environ.get('SECRET_KEY', 'snake-game-secret'),
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': True,
        })
        print_log("Flask configured")
    
    def setup_routes(self) -> None:
        """Set up minimal Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main interface."""
            return render_template('base.html', 
                                 app_name=self.name,
                                 game_data=self.get_game_data())
        
        @self.app.route('/api/state')
        def get_state():
            """Get current state."""
            return jsonify(self.get_api_state())
        
        @self.app.route('/api/control', methods=['POST'])
        def control():
            """Handle controls."""
            data = request.get_json() or {}
            return jsonify(self.handle_control(data))
        
        @self.app.route('/api/health')
        def health():
            """Health check."""
            return jsonify({'status': 'healthy', 'app': self.name})
        
        print_log("Routes configured")
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get data for template rendering."""
        return {
            'name': self.name,
            'status': 'ready'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get API state - override in subclasses."""
        return {
            'app': self.name,
            'status': 'ready'
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control requests - override in subclasses."""
        action = data.get('action', 'unknown')
        return {'action': action, 'status': 'processed'}
    
    def run(self, host: str = "127.0.0.1", debug: bool = False):
        """Run Flask application."""
        print_log(f"Starting {self.name} on http://{host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug)


class HumanGameApp(SimpleFlaskApp):
    """
    Human player web application.
    
    Extension Pattern: Copy this pattern for any algorithm/model
    Educational Value: Shows minimal specialization
    """
    
    def __init__(self, grid_size: int = 10, **config):
        """Initialize human game app."""
        super().__init__("Human Snake Game")
        self.grid_size = grid_size
        self.config = config
        print_log(f"Human mode: {grid_size}x{grid_size} grid")
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get human game data."""
        return {
            'name': self.name,
            'mode': 'human',
            'grid_size': self.grid_size,
            'controls': ['Arrow Keys', 'WASD', 'Reset: R'],
            'status': 'ready'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get human game state."""
        return {
            'mode': 'human',
            'grid_size': self.grid_size,
            'status': 'ready'
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human controls."""
        action = data.get('action', '')
        direction = data.get('direction', '')
        
        if action == 'move' and direction:
            print_log(f"Human move: {direction}")
            return {'action': 'move', 'direction': direction, 'status': 'processed'}
        elif action == 'reset':
            print_log("Human reset")
            return {'action': 'reset', 'status': 'processed'}
        
        return {'error': 'Unknown action'}


class LLMGameApp(SimpleFlaskApp):
    """
    LLM player web application.
    
    Extension Pattern: Copy this pattern for any algorithm/model
    Educational Value: Shows minimal specialization with external components
    """
    
    def __init__(self, provider: str = "hunyuan", model: str = "hunyuan-turbos-latest", 
                 grid_size: int = 10, **config):
        """Initialize LLM game app."""
        super().__init__("LLM Snake Game")
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        self.config = config
        
        # Initialize game components when needed
        self.game_manager = None
        self.agent = None
        
        print_log(f"LLM mode: {provider}/{model}, {grid_size}x{grid_size} grid")
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get LLM game data."""
        return {
            'name': self.name,
            'mode': 'llm',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'features': ['AI Reasoning', 'Real-time Play', 'Performance Metrics'],
            'status': 'ready'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get LLM game state."""
        return {
            'mode': 'llm',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'status': 'ready'
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM controls."""
        action = data.get('action', '')
        
        if action == 'start':
            print_log("LLM game start")
            return {'action': 'start', 'status': 'started'}
        elif action == 'pause':
            print_log("LLM game pause")
            return {'action': 'pause', 'status': 'paused'}
        elif action == 'reset':
            print_log("LLM game reset")
            return {'action': 'reset', 'status': 'reset'}
        
        return {'error': 'Unknown action'}


class ReplayGameApp(SimpleFlaskApp):
    """
    Replay viewer web application.
    
    Extension Pattern: Copy this pattern for any replay/analysis needs
    Educational Value: Shows minimal specialization with data loading
    """
    
    def __init__(self, log_dir: str, game_number: int = 1, **config):
        """Initialize replay app."""
        super().__init__("Snake Game Replay")
        self.log_dir = log_dir
        self.game_number = game_number
        self.config = config
        
        print_log(f"Replay mode: {log_dir}, starting game {game_number}")
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get replay data."""
        return {
            'name': self.name,
            'mode': 'replay',
            'log_dir': os.path.basename(self.log_dir),
            'game_number': self.game_number,
            'controls': ['Play/Pause', 'Step Forward/Back', 'Game Selection'],
            'status': 'ready'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get replay state."""
        return {
            'mode': 'replay',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'status': 'ready'
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay controls."""
        action = data.get('action', '')
        
        if action == 'play':
            print_log("Replay play")
            return {'action': 'play', 'status': 'playing'}
        elif action == 'pause':
            print_log("Replay pause")
            return {'action': 'pause', 'status': 'paused'}
        elif action == 'step':
            direction = data.get('direction', 'forward')
            print_log(f"Replay step {direction}")
            return {'action': 'step', 'direction': direction, 'status': 'stepped'}
        
        return {'error': 'Unknown action'}


# Simple factory functions (KISS pattern)

def create_human_app(grid_size: int = 10, **config) -> HumanGameApp:
    """Create human game app."""
    return HumanGameApp(grid_size=grid_size, **config)


def create_llm_app(provider: str = "hunyuan", model: str = "hunyuan-turbos-latest",
                   grid_size: int = 10, **config) -> LLMGameApp:
    """Create LLM game app."""
    return LLMGameApp(provider=provider, model=model, grid_size=grid_size, **config)


def create_replay_app(log_dir: str, game_number: int = 1, **config) -> ReplayGameApp:
    """Create replay app."""
    return ReplayGameApp(log_dir=log_dir, game_number=game_number, **config)


# Extension template for future tasks
"""
Extension Pattern Template:

class YourTaskApp(SimpleFlaskApp):
    '''Your task web application.'''
    
    def __init__(self, your_params, **config):
        super().__init__("Your Task Name")
        self.your_params = your_params
        # Initialize your components here
    
    def get_game_data(self):
        return {
            'name': self.name,
            'mode': 'your_mode',
            'your_data': self.your_params,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        # Handle your task-specific controls
        return {'status': 'processed'}

def create_your_app(**config):
    return YourTaskApp(**config)
""" 