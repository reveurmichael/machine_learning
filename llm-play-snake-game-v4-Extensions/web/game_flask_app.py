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
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Serve favicon to prevent 404 errors."""
            from flask import send_from_directory
            return send_from_directory('static', 'favicon.ico')
        
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
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = False):
        """Run Flask application."""
        # Use provided port or fall back to instance port
        actual_port = port if port is not None else self.port
        print_log(f"Starting {self.name} on http://{host}:{actual_port}")
        self.app.run(host=host, port=actual_port, debug=debug)


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
        # Initialize a simple game state for human play
        center = self.grid_size // 2
        return {
            'mode': 'human',
            'grid_size': self.grid_size,
            'snake_positions': [[center, center]],  # Snake starts in center
            'apple_position': [center + 2, center + 2],  # Apple nearby
            'score': 0,
            'steps': 0,
            'running': True,
            'game_active': True,
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
        # Initialize a simple game state for LLM play
        center = self.grid_size // 2
        return {
            'mode': 'llm',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'snake_positions': [[center, center]],  # Snake starts in center
            'apple_position': [center + 2, center + 2],  # Apple nearby
            'score': 0,
            'steps': 0,
            'running': True,
            'game_active': True,
            'llm_response': 'Ready to start playing...',
            'planned_moves': [],
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
        # Try to load actual game data from log files
        try:
            import json
            from pathlib import Path
            
            # Look for game log file
            log_path = Path(self.log_dir)
            game_file = log_path / f"game_{self.game_number}.json"
            
            if game_file.exists():
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                
                # Extract game state from log data
                if 'snake_positions' in game_data and 'apple_position' in game_data:
                    return {
                        'mode': 'replay',
                        'grid_size': game_data.get('grid_size', 10),
                        'snake_positions': game_data['snake_positions'],
                        'apple_position': game_data['apple_position'],
                        'score': game_data.get('final_score', 0),
                        'steps': game_data.get('total_steps', 0),
                        'running': False,  # Replay is static
                        'game_active': False,
                        'end_reason': game_data.get('end_reason', 'Game completed'),
                        'log_dir': self.log_dir,
                        'game_number': self.game_number,
                        'status': 'loaded'
                    }
        except Exception as e:
            print_log(f"Error loading game data: {e}")
        
        # Fallback to demo state if no log data available
        return {
            'mode': 'replay',
            'grid_size': 10,
            'snake_positions': [[5, 5], [5, 4], [5, 3]],  # Demo snake
            'apple_position': [7, 7],  # Demo apple
            'score': 3,
            'steps': 15,
            'running': False,
            'game_active': False,
            'end_reason': 'Demo replay data',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'status': 'demo'
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