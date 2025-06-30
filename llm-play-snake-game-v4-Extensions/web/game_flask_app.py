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
import sys
import time
import threading
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request

# Import utilities (no Task-0 pollution)
from utils.network_utils import random_free_port
from utils.web_utils import build_color_map

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
    """Build color payload for web interface."""
    color_map = build_color_map()
    return {
        'snake_head': list(color_map['snake_head']),
        'snake_body': list(color_map['snake_body']),
        'apple': list(color_map['apple']),
        'background': list(color_map['background']),
        'grid': list(color_map['grid']),
    }


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
        
        # Initialize the actual replay engine (like pygame version)
        from replay.replay_engine import ReplayEngine
        self.replay_engine = ReplayEngine(
            log_dir=log_dir,
            pause_between_moves=1.0,
            auto_advance=False,
            use_gui=False  # No GUI for web mode
        )
        
        # Set initial game number
        self.replay_engine.game_number = game_number
        
        # Load the initial game data
        if not self.replay_engine.load_game_data(game_number):
            print_log(f"Warning: Could not load game {game_number} from {log_dir}")
        else:
            print_log(f"Successfully loaded game {game_number}")
            print_log(f"DEBUG: After load_game_data, game_end_reason = {getattr(self.replay_engine, 'game_end_reason', 'NOT_SET')}")
            print_log(f"DEBUG: After load_game_data, hasattr = {hasattr(self.replay_engine, 'game_end_reason')}")
        
        # Start **unpaused** so replay begins immediately
        self.replay_engine.paused = False
        
        # Start background thread for replay updates
        self.replay_thread = None
        self.replay_running = True
        self._start_replay_thread()
        
        print_log(f"Replay mode: {log_dir}, starting game {game_number}")
    
    def _start_replay_thread(self):
        """Start background thread for replay updates."""
        def replay_update_loop():
            """Background loop to update replay engine state."""
            while self.replay_running:
                try:
                    if hasattr(self, 'replay_engine') and self.replay_engine:
                        # Update replay engine (like pygame version)
                        if not self.replay_engine.paused and self.replay_engine.running:
                            # Check if it's time for the next move
                            current_time = time.time()
                            if current_time - self.replay_engine.last_move_time >= self.replay_engine.pause_between_moves:
                                # Execute next move
                                if self.replay_engine.move_index < len(self.replay_engine.moves):
                                    next_move = self.replay_engine.moves[self.replay_engine.move_index]
                                    game_continues = self.replay_engine.execute_replay_move(next_move)
                                    self.replay_engine.move_index += 1
                                    self.replay_engine.last_move_time = current_time
                                    
                                    if not game_continues:
                                        print_log(f"Game {self.replay_engine.game_number} ended")
                                        # Use the actual end reason from the game data, not hardcoded text
                                        if hasattr(self.replay_engine, 'game_end_reason') and self.replay_engine.game_end_reason:
                                            # Keep the original end reason from the game data
                                            pass
                                        else:
                                            # Fallback to a generic reason if none exists
                                            self.replay_engine.game_end_reason = "Game ended"
                                        self.replay_engine.running = False
                                else:
                                    # All moves completed - use the actual end reason from game data
                                    if hasattr(self.replay_engine, 'game_end_reason') and self.replay_engine.game_end_reason:
                                        # Keep the original end reason from the game data
                                        pass
                                    else:
                                        # Fallback if no end reason exists
                                        self.replay_engine.game_end_reason = "All moves completed"
                                    self.replay_engine.running = False
                    
                    # Sleep to avoid busy loop
                    time.sleep(0.1)  # 100ms update rate
                    
                except Exception as e:
                    print_log(f"Error in replay update loop: {e}")
                    time.sleep(0.1)
        
        self.replay_thread = threading.Thread(target=replay_update_loop, daemon=True)
        self.replay_thread.start()
        print_log("Started replay update thread")
    
    def __del__(self):
        """Cleanup when app is destroyed."""
        self.replay_running = False
        if self.replay_thread and self.replay_thread.is_alive():
            self.replay_thread.join(timeout=1.0)
    
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
        """Get replay state from the actual replay engine."""
        if not hasattr(self, 'replay_engine') or self.replay_engine is None:
            return self._get_fallback_state()
        
        try:
            # Get state from the replay engine (like pygame version)
            state = self.replay_engine._build_state_base()
            
            # Convert numpy arrays to lists for JSON serialization
            if 'snake_positions' in state and state['snake_positions'] is not None:
                state['snake_positions'] = state['snake_positions'].tolist()
            if 'apple_position' in state and state['apple_position'] is not None:
                state['apple_position'] = state['apple_position'].tolist()
            
            # Always show the canonical end reason for the loaded game
            end_reason = getattr(self.replay_engine, 'game_end_reason', None)
            
            # Debug logging
            print_log(f"DEBUG: replay_engine.game_end_reason = {end_reason}")
            print_log(f"DEBUG: hasattr(replay_engine, 'game_end_reason') = {hasattr(self.replay_engine, 'game_end_reason')}")
            
            # Add web-specific fields
            total_moves = len(self.replay_engine.moves) if hasattr(self.replay_engine, 'moves') else 0
            is_at_end = self.replay_engine.move_index >= total_moves
            # Game is over only if we're at the end of moves OR the replay engine is not running
            # Having a game_end_reason doesn't mean the game is over during replay
            game_over = is_at_end or not self.replay_engine.running
            state.update({
                'mode': 'replay',
                'log_dir': self.log_dir,
                'game_number': self.replay_engine.game_number,
                'status': 'loaded',
                'colors': _build_color_payload(),
                'paused': self.replay_engine.paused,
                'move_index': self.replay_engine.move_index,
                'total_moves': total_moves,
                'pause_between_moves': self.replay_engine.pause_between_moves,
                'running': self.replay_engine.running,
                'game_active': self.replay_engine.running and not game_over,
                'game_over': game_over,
                'end_reason': end_reason,
                'primary_llm': getattr(self.replay_engine, 'primary_llm', ''),
                'parser_llm': getattr(self.replay_engine, 'secondary_llm', ''),
                'timestamp': getattr(self.replay_engine, 'game_timestamp', '')
            })
            
            return state
            
        except Exception as e:
            print_log(f"Error getting replay state: {e}")
            return self._get_fallback_state()
    
    def _get_fallback_state(self) -> Dict[str, Any]:
        """Fallback state if replay engine is not available."""
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
            'end_reason': 'Demo replay data - replay engine not available',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'status': 'demo',
            'colors': _build_color_payload()
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay controls using the actual replay engine."""
        if not hasattr(self, 'replay_engine') or self.replay_engine is None:
            return {'error': 'Replay engine not available'}
        
        action = data.get('action', '')
        command = data.get('command', '')
        
        try:
            if action == 'play' or command == 'play':
                print_log("Replay play")
                self.replay_engine.paused = False
                return {'action': 'play', 'status': 'playing'}
                
            elif action == 'pause' or command == 'pause':
                print_log("Replay pause")
                self.replay_engine.paused = True
                return {'action': 'pause', 'status': 'paused'}
                
            elif command == 'next_game':
                print_log("Next game")
                self.replay_engine.game_number += 1
                if not self.replay_engine.load_game_data(self.replay_engine.game_number):
                    self.replay_engine.game_number -= 1
                    return {'status': 'error', 'message': 'No next game'}
                # Auto-play new game
                self.replay_engine.paused = False
                return {'status': 'ok'}
                
            elif command == 'prev_game':
                print_log("Previous game")
                if self.replay_engine.game_number > 1:
                    self.replay_engine.game_number -= 1
                    self.replay_engine.load_game_data(self.replay_engine.game_number)
                    self.replay_engine.paused = False
                    return {'status': 'ok'}
                return {'status': 'error', 'message': 'Already at first game'}
                
            elif command == 'restart_game':
                print_log("Restart game")
                self.replay_engine.load_game_data(self.replay_engine.game_number)
                self.replay_engine.paused = False
                return {'status': 'ok'}
                
            elif command == 'speed_up':
                print_log("Speed up")
                self.replay_engine.pause_between_moves = max(0.1, self.replay_engine.pause_between_moves * 0.75)
                return {'status': 'ok', 'move_pause': self.replay_engine.pause_between_moves}
                
            elif command == 'speed_down':
                print_log("Speed down")
                self.replay_engine.pause_between_moves = min(3.0, self.replay_engine.pause_between_moves * 1.25)
                return {'status': 'ok', 'move_pause': self.replay_engine.pause_between_moves}
            
        except Exception as e:
            print_log(f"Error handling replay control: {e}")
            return {'error': f'Control error: {e}'}
        
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