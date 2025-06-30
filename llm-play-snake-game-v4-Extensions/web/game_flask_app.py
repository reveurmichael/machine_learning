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

# Import central colour palette
from config.ui_constants import COLORS as UI_COLORS

# Map UI_COLORS keys to camelCase expected by JS
_COLOR_KEY_MAP = {
    'SNAKE_HEAD': 'snake_head',
    'SNAKE_BODY': 'snake_body',
    'APPLE': 'apple',
    'BACKGROUND': 'background',
    'GRID': 'grid',
}


def _build_color_payload() -> Dict[str, list]:
    """Return colour palette as {key: [r,g,b]} expected by front-end JS."""
    payload: Dict[str, list] = {}
    for k_ui, k_js in _COLOR_KEY_MAP.items():
        rgb = UI_COLORS.get(k_ui)
        if rgb is not None:
            payload[k_js] = list(rgb)
    return payload


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
            return render_template(self.get_template_name(),
                                   app_name=self.name,
                                   game_data=self.get_game_data())
        
        @self.app.route('/api/state')
        def get_state():
            """Get current state."""
            return jsonify(self.get_api_state())
        
        @self.app.route('/api/move', methods=['POST'])
        def make_move():
            """Handle move requests (for human play)."""
            data = request.get_json() or {}
            direction = data.get('direction', '')
            
            if not direction:
                return jsonify({'status': 'error', 'message': 'No direction provided'})
            
            # Convert to control format and delegate
            control_data = {'action': 'move', 'direction': direction}
            result = self.handle_control(control_data)
            return jsonify(result)
        
        @self.app.route('/api/control', methods=['POST'])
        def control():
            """Handle controls."""
            data = request.get_json() or {}
            return jsonify(self.handle_control(data))
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset():
            """Handle reset requests."""
            result = self.handle_control({'action': 'reset'})
            return jsonify(result)
        
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
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = True):
        """Run Flask application."""
        # Use provided port or fall back to instance port
        actual_port = port if port is not None else self.port
        print_log(f"Starting {self.name} on http://{host}:{actual_port}")
        self.app.run(host=host, port=actual_port, debug=debug)

    # ------------------------------------------------------------------
    # Template selection helpers
    # ------------------------------------------------------------------

    def get_template_name(self) -> str:
        """Return the Jinja2 template to render for the index route.

        Sub-classes override this to supply their mode-specific template
        (e.g. ``human_play.html``).  Default is ``base.html`` so that
        existing behaviour keeps working for quick experiments.
        """
        return 'base.html'


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
        # Minimal state for demo movement
        c = grid_size // 2
        self.snake_positions = [[c, c]]  # head last element per JS expectation
        self.apple_position = [c + 2, c + 2]
        self.score = 0
        self.steps = 0
        self.game_over = False
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
            'snake_positions': self.snake_positions,
            'apple_position': self.apple_position,
            'score': self.score,
            'steps': self.steps,
            'running': not self.game_over,
            'game_active': not self.game_over,
            'game_over': self.game_over,
            'status': 'ready',
            'colors': _build_color_payload()
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human controls."""
        action = data.get('action', '')
        direction = data.get('direction', '')
        
        if action == 'move' and direction:
            print_log(f"Human move: {direction}")
            if self.game_over:
                return {'status': 'error', 'message': 'Game is over'}
            
            success = self._apply_move(direction)
            if success:
                return {
                    'status': 'ok',
                    'game_active': not self.game_over,
                    'score': self.score,
                    'steps': self.steps
                }
            else:
                return {'status': 'error', 'message': 'Invalid move'}
                
        elif action == 'reset':
            print_log("Human reset")
            self._reset_game()
            return {'status': 'success', 'message': 'Game reset'}
        
        return {'status': 'error', 'message': 'Unknown action'}

    # ---------------- internal helpers ----------------

    _DIR_MAP = {
        'UP': (0, 1),
        'DOWN': (0, -1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
    }

    def _apply_move(self, direction: str) -> bool:
        """Update snake position for a single-step move (demo logic)."""
        if self.game_over:
            return False
            
        direction = direction.upper()
        if direction not in self._DIR_MAP:
            return False
            
        dx, dy = self._DIR_MAP[direction]
        head_x, head_y = self.snake_positions[-1]
        new_x, new_y = head_x + dx, head_y + dy
        
        # bounds check
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            self.game_over = True
            return False
            
        # self collision check
        if [new_x, new_y] in self.snake_positions:
            self.game_over = True
            return False
            
        # move snake
        self.snake_positions.append([new_x, new_y])
        
        # apple eaten?
        if [new_x, new_y] == self.apple_position:
            self.score += 1
            self._generate_new_apple()
        else:
            # remove tail to keep length constant (classic snake demo)
            self.snake_positions.pop(0)
            
        self.steps += 1
        return True
    
    def _generate_new_apple(self):
        """Generate new apple position avoiding snake body."""
        from random import randint
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            ax, ay = randint(0, self.grid_size - 1), randint(0, self.grid_size - 1)
            if [ax, ay] not in self.snake_positions:
                self.apple_position = [ax, ay]
                return
            attempts += 1
        
        # Fallback: find first empty position
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if [x, y] not in self.snake_positions:
                    self.apple_position = [x, y]
                    return
    
    def _reset_game(self):
        """Reset game to initial state."""
        center = self.grid_size // 2
        self.snake_positions = [[center, center]]
        self.apple_position = [center + 2, center + 2]
        self.score = 0
        self.steps = 0
        self.game_over = False
        print_log("Game reset to initial state")

    # -------------------- Template override ---------------------------

    def get_template_name(self) -> str:
        return 'human_play.html'


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
        # TODO: Integrate with actual GameManager for real LLM gameplay
        # For now, provide demo state that matches JavaScript expectations
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
            'game_over': False,
            'llm_response': 'Ready to start LLM-powered Snake Game...\n\nThe AI will analyze the game state and make strategic moves to maximize score while avoiding collisions.',
            'planned_moves': ['UP', 'RIGHT', 'UP'],  # Demo planned moves
            'status': 'ready',
            'colors': _build_color_payload()
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

    def get_template_name(self) -> str:
        return 'main.html'


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
                with open(game_file, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                
                # Extract final game state from log data
                final_snake_positions = []
                final_apple_position = [5, 5]  # Default fallback
                
                # Try to get final positions from detailed_history
                detailed = game_data.get('detailed_history', {})
                if detailed:
                    # Get final snake position (reconstruct from moves)
                    moves = detailed.get('moves', [])
                    apple_positions = detailed.get('apple_positions', [])
                    
                    if apple_positions:
                        last_apple = apple_positions[-1]  # Last apple position
                        # Handle both formats: {x: 5, y: 4} or [5, 4]
                        if isinstance(last_apple, dict) and 'x' in last_apple and 'y' in last_apple:
                            final_apple_position = [last_apple['x'], last_apple['y']]
                        elif isinstance(last_apple, (list, tuple)) and len(last_apple) >= 2:
                            final_apple_position = [last_apple[0], last_apple[1]]
                        else:
                            print_log(f"Unexpected apple position format: {last_apple}")
                            final_apple_position = [5, 5]
                    
                    # Reconstruct final snake position (simplified)
                    grid_size = game_data.get('grid_size', 10)
                    center = grid_size // 2
                    final_snake_positions = [[center, center]]  # Start position
                    
                    # Apply some moves to show snake progression (demo)
                    if moves and len(moves) > 3:
                        # Show snake with some length based on score
                        score = game_data.get('final_score', 0)
                        snake_length = min(score + 1, 5)  # Limit demo length
                        for i in range(snake_length):
                            x = center + i
                            y = center
                            if 0 <= x < grid_size:
                                final_snake_positions.append([x, y])
                
                return {
                    'mode': 'replay',
                    'grid_size': game_data.get('grid_size', 10),
                    'snake_positions': final_snake_positions or [[5, 5]],
                    'apple_position': final_apple_position,
                    'score': game_data.get('final_score', 0),
                    'steps': game_data.get('total_steps', 0),
                    'running': False,  # Replay is static
                    'game_active': False,
                    'game_over': True,
                    'end_reason': game_data.get('game_end_reason', 'Game completed'),
                    'log_dir': self.log_dir,
                    'game_number': self.game_number,
                    'status': 'loaded',
                    'colors': _build_color_payload(),
                    # Add replay-specific info
                    'timestamp': game_data.get('timestamp', ''),
                    'primary_llm': game_data.get('primary_llm', ''),
                    'parser_llm': game_data.get('parser_llm', '')
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
            'game_over': True,
            'end_reason': 'Demo replay data - no log files found',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'status': 'demo',
            'colors': _build_color_payload()
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

    def get_template_name(self) -> str:
        return 'replay.html'


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