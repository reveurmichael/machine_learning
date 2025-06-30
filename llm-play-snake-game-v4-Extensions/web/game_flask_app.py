"""
Task-0 Flask Web Interface - Enhanced KISS Architecture
--------------------

Enhanced minimal, KISS, DRY, and extensible web backend for Task-0 and all future extensions.
Direct integration with existing game logic without complex patterns or Task-0 pollution.

IMPORTANT: Now uses existing GameLogic and GameManager classes for consistency
across all game interfaces (CLI, GUI, web). This ensures single source of truth
for game behavior and eliminates code duplication.

Enhanced Features (following replay.py and main.py patterns):
- Standardized argument parsing with get_parser() and parse_arguments() functions
- Factory Pattern with canonical create() methods (per SUPREME_RULES)
- OOP Application classes following MainApplication pattern
- Proper configuration integration and path management
- Educational docstrings and design pattern explanations
- Simple logging using print() statements (SUPREME_RULES compliance)

Game Component Integration:
- HumanGameApp: Uses GameLogic from core.game_logic (same as human_play.py)
- LLMGameApp: References GameManager pattern (full integration via scripts/main_web.py)
- ReplayGameApp: Uses ReplayEngine from replay module (already correct)

Design Philosophy:
- KISS: Keep It Simple, Stupid - no over-engineering
- DRY: Don't Repeat Yourself - reuse existing game components
- Minimal: Only essential functionality
- Extensible: Easy to extend for future tasks
- Consistent: Same game logic across all interfaces
- Educational: Demonstrates proper architectural patterns

Educational Goals:
- Show direct Flask integration with existing components
- Demonstrate minimal web architecture with standard patterns
- Provide copy-paste templates for extensions
- Keep functionality while removing complexity
- Illustrate proper separation of game logic from presentation
- Demonstrate Factory Pattern with canonical create() methods
- Show argument parsing reusability patterns

Extension Pattern:
Copy this file → Replace game components → Maintain same simple structure
"""

import os
import sys
import time
import threading
import argparse
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request

# Import Task-0 core components for proper game logic integration
from core.game_logic import GameLogic
from core.game_manager import GameManager

# Import utilities following SSOT principles
from utils.network_utils import random_free_port
from utils.web_utils import build_color_map
from utils.path_utils import ensure_project_root

# Import configuration following SSOT hierarchy (universal constants)
from config.ui_constants import COLORS as UI_COLORS
from config.game_constants import DIRECTIONS, VALID_MOVES

# Ensure project root for consistent behavior
ensure_project_root()

# Map UI_COLORS keys to camelCase expected by JS
_COLOR_KEY_MAP = {
    'SNAKE_HEAD': 'snake_head',
    'SNAKE_BODY': 'snake_body',
    'APPLE': 'apple',
    'BACKGROUND': 'background',
    'GRID': 'grid',
}


def _build_color_payload() -> Dict[str, list]:
    """Build color payload for web interface following SSOT principles."""
    color_map = build_color_map()
    return {
        'snake_head': list(color_map['snake_head']),
        'snake_body': list(color_map['snake_body']),
        'apple': list(color_map['apple']),
        'background': list(color_map['background']),
        'grid': list(color_map['grid']),
    }


# Helper function to convert numpy arrays to lists for JSON serialization
def _numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Simple logging following SUPREME_RULES (final-decision-10.md compliance)
print_log = lambda msg: print(f"[WebApp] {msg}")


# ------------------
# Argument Parsing Functions (following scripts/main.py and scripts/replay.py patterns)
# ------------------

def get_human_parser() -> argparse.ArgumentParser:
    """Creates argument parser for human game web interface.
    
    Following the pattern from scripts/main.py and scripts/replay.py, this function
    creates a reusable argument parser that can be imported by other scripts.
    
    Design Pattern: Factory Pattern (Argument Parser Creation)
    Educational Value: Shows standardized argument parsing across web interfaces
    Extension Pattern: Future tasks can copy this pattern for their web interfaces
    
    Returns:
        An argparse.ArgumentParser instance for human game configuration
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Human Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -c "from web.game_flask_app import create_human_app; create_human_app().run()"
  
Extension Pattern:
  Future tasks can copy this parser structure and modify for their needs
  while maintaining consistent argument handling patterns.
        """
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Size of the game grid (default: 10)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for the web server (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode for development"
    )
    
    return parser


def get_replay_parser() -> argparse.ArgumentParser:
    """Creates argument parser for replay web interface.
    
    Follows the same pattern as scripts/replay.py for consistency.
    
    Design Pattern: Factory Pattern (Argument Parser Creation)
    Educational Value: Shows how replay arguments are standardized
    Extension Pattern: Extensions can reuse this for their replay interfaces
    
    Returns:
        An argparse.ArgumentParser instance for replay configuration
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Replay Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing game logs"
    )
    
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Game number to replay (default: 1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for the web server (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode for development"
    )
    
    return parser


def parse_human_arguments():
    """Parse command line arguments for human web interface.
    
    Follows the pattern from scripts/main.py and scripts/replay.py.
    
    Educational Value: Shows standardized argument parsing
    Extension Pattern: Future tasks can copy this exact pattern
    """
    parser = get_human_parser()
    return parser.parse_args()


def parse_replay_arguments():
    """Parse command line arguments for replay web interface.
    
    Follows the pattern from scripts/main.py and scripts/replay.py.
    
    Educational Value: Shows standardized argument parsing
    Extension Pattern: Future tasks can copy this exact pattern
    """
    parser = get_replay_parser()
    return parser.parse_args()


# ------------------
# Web Application Factory (following Factory Pattern with canonical create() methods)
# ------------------

class WebAppFactory:
    """
    Factory for creating web applications with canonical create() methods.
    
    Design Pattern: Factory Pattern (Canonical Implementation per SUPREME_RULES)
    Purpose: Create web applications using canonical create() method
    Educational Value: Shows factory pattern following final-decision-10.md standards
    Extension Pattern: Extensions can copy this factory pattern
    
    IMPORTANT: Uses canonical create() method name as mandated by SUPREME_RULES
    """
    
    _registry = {
        "human": "HumanGameApp",
        "llm": "LLMGameApp", 
        "replay": "ReplayGameApp",
    }
    
    @classmethod
    def create(cls, app_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create web application using canonical create() method.
        
        Following SUPREME_RULES from final-decision-10.md, all factories must use
        the canonical create() method name for consistency across the project.
        
        Args:
            app_type: Type of application to create ('human', 'llm', 'replay')
            **kwargs: Configuration parameters for the application
            
        Returns:
            Configured web application instance
            
        Raises:
            ValueError: If app_type is not supported
        """
        app_class_name = cls._registry.get(app_type.lower())
        if not app_class_name:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown app type: {app_type}. Available: {available}")
        
        print_log(f"Creating web app: {app_type}")  # Simple logging per SUPREME_RULES
        
        # Get the actual class and instantiate it
        if app_class_name == "HumanGameApp":
            return HumanGameApp(**kwargs)
        elif app_class_name == "LLMGameApp":
            return LLMGameApp(**kwargs)
        elif app_class_name == "ReplayGameApp":
            return ReplayGameApp(**kwargs)
        else:
            raise ValueError(f"Unknown app class: {app_class_name}")


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
    Human player web application with enhanced GameLogic integration.
    
    Uses the existing GameLogic class from core.game_logic for consistent 
    game behavior across all interfaces (GUI, web, CLI). Follows the same
    integration pattern as scripts/human_play.py.
    
    Design Pattern: Adapter Pattern + Template Method Pattern
    Purpose: Adapts GameLogic interface for web API consumption
    Educational Value: Shows proper separation of game logic from presentation
    Extension Pattern: Copy this pattern for any algorithm/model
    
    Enhanced Features:
    - Proper error handling and recovery
    - Configuration validation
    - State management following SSOT principles
    - Educational documentation with design patterns
    """
    
    def __init__(self, grid_size: int = 10, **config):
        """Initialize human game app with enhanced GameLogic integration.
        
        Args:
            grid_size: Size of the game grid (default: 10)
            **config: Additional configuration options
            
        Design Pattern: Factory Method Pattern (via GameLogic instantiation)
        Educational Value: Shows how to properly initialize game components
        """
        super().__init__("Human Snake Game")
        self.grid_size = self._validate_grid_size(grid_size)
        self.config = config
        
        # Use the existing GameLogic class - same integration as human_play.py
        # This ensures consistency across all game interfaces
        try:
            self.game = GameLogic(grid_size=self.grid_size, use_gui=False)
            print_log(f"Human mode: {self.grid_size}x{self.grid_size} grid, using GameLogic")
            print_log("GameLogic integration successful - consistent with human_play.py")
        except Exception as e:
            print_log(f"Error initializing GameLogic: {e}")
            raise
    
    def _validate_grid_size(self, grid_size: int) -> int:
        """Validate grid size parameter.
        
        Args:
            grid_size: Grid size to validate
            
        Returns:
            Validated grid size
            
        Raises:
            ValueError: If grid size is invalid
            
        Educational Value: Shows input validation patterns
        """
        if not isinstance(grid_size, int) or grid_size < 5 or grid_size > 50:
            raise ValueError(f"Grid size must be between 5 and 50, got: {grid_size}")
        return grid_size
    
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
        """Get human game state from GameLogic instance."""
        return {
            'mode': 'human',
            'grid_size': self.grid_size,
            'snake_positions': _numpy_to_list(self.game.snake_positions),
            'apple_position': _numpy_to_list(self.game.apple_position),
            'score': self.game.score,
            'steps': self.game.steps,
            'running': not self.game.game_over,
            'game_active': not self.game.game_over,
            'game_over': self.game.game_over,
            'status': 'ready',
            'colors': _build_color_payload()
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human controls using GameLogic methods.
        
        Args:
            data: Control data containing action and parameters
            
        Returns:
            Response dictionary with status and relevant information
            
        Design Pattern: Command Pattern
        Purpose: Encapsulates user actions as command objects
        Educational Value: Shows how to handle user input systematically
        """
        action = data.get('action', '')
        direction = data.get('direction', '')
        
        try:
            if action == 'move' and direction:
                return self._handle_move_command(direction)
            elif action == 'reset':
                return self._handle_reset_command()
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            print_log(f"Error handling control: {e}")
            return {'status': 'error', 'message': f'Internal error: {str(e)}'}
    
    def _handle_move_command(self, direction: str) -> Dict[str, Any]:
        """Handle move command with proper validation.
        
        Args:
            direction: Direction to move ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Returns:
            Response dictionary with move result
            
        Educational Value: Shows move validation and error handling
        """
        print_log(f"Human move: {direction}")
        
        # Validate direction input
        direction_upper = direction.upper()
        if direction_upper not in VALID_MOVES:
            return {'status': 'error', 'message': f'Invalid direction: {direction}'}
        
        # Check game state
        if self.game.game_over:
            return {'status': 'error', 'message': 'Game is over'}
        
        # Use GameLogic.make_move() - same integration as human_play.py
        # This ensures identical behavior across CLI, GUI, and web interfaces
        game_active, move_successful = self.game.make_move(direction_upper)
        
        if move_successful:
            return {
                'status': 'ok',
                'game_active': game_active,
                'score': self.game.score,
                'steps': self.game.steps,
                'message': f'Moved {direction_upper}'
            }
        else:
            return {'status': 'error', 'message': 'Invalid move - collision or reverse direction'}
    
    def _handle_reset_command(self) -> Dict[str, Any]:
        """Handle reset command with proper cleanup.
        
        Returns:
            Response dictionary with reset result
            
        Educational Value: Shows state reset patterns
        """
        print_log("Human reset")
        
        try:
            # Use GameLogic.reset() - same integration as human_play.py
            self.game.reset()
            
            return {
                'status': 'success', 
                'message': 'Game reset successfully',
                'score': self.game.score,
                'steps': self.game.steps
            }
        except Exception as e:
            print_log(f"Error during reset: {e}")
            return {'status': 'error', 'message': f'Reset failed: {str(e)}'}

    # Custom game logic methods removed - now using GameLogic from core.game_logic
    # This ensures consistency with human_play.py and all other game interfaces

    # -------------------- Template override ---------------------------

    def get_template_name(self) -> str:
        return 'human_play.html'


class LLMGameApp(SimpleFlaskApp):
    """
    LLM player web application.
    
    Uses the existing GameManager class from core.game_manager for LLM 
    integration, ensuring consistency with the main LLM gameplay mode.
    
    Design Pattern: Facade Pattern
    Purpose: Provides web interface facade over GameManager functionality
    Educational Value: Shows proper integration with existing LLM components
    Extension Pattern: Copy this pattern for any algorithm/model
    """
    
    def __init__(self, provider: str = "hunyuan", model: str = "hunyuan-turbos-latest", 
                 grid_size: int = 10, **config):
        """Initialize LLM game app with GameManager integration."""
        super().__init__("LLM Snake Game")
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        self.config = config
        
        # Create a mock args object for GameManager initialization
        # TODO: Consider refactoring GameManager to accept individual parameters
        self.game_manager = None
        self.agent = None
        
        print_log(f"LLM mode: {provider}/{model}, {grid_size}x{grid_size} grid")
        print_log("Note: Full LLM integration requires proper argument setup")
    
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
        """Get LLM game state.
        
        Note: Full LLM integration would require GameManager setup with proper
        argument parsing, LLM client initialization, and session management.
        This would follow the same pattern as scripts/main_web.py but requires
        more architectural work to properly integrate web and CLI patterns.
        """
        # Demo state until proper GameManager integration
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
            'llm_response': 'Ready to start LLM-powered Snake Game...\n\nNote: Full LLM integration requires GameManager setup.\nFor complete LLM functionality, use scripts/main_web.py',
            'planned_moves': ['UP', 'RIGHT', 'UP'],  # Demo planned moves
            'status': 'ready',
            'colors': _build_color_payload()
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM controls.
        
        Note: Full implementation would delegate to GameManager methods
        for actual LLM gameplay, session management, and statistics tracking.
        """
        action = data.get('action', '')
        
        if action == 'start':
            print_log("LLM game start - demo mode")
            return {'action': 'start', 'status': 'started', 'note': 'Demo mode - use scripts/main_web.py for full LLM'}
        elif action == 'pause':
            print_log("LLM game pause - demo mode")
            return {'action': 'pause', 'status': 'paused'}
        elif action == 'reset':
            print_log("LLM game reset - demo mode")
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


# ------------------
# Factory Functions (Enhanced with canonical create() methods)
# ------------------

def create_human_app(grid_size: int = 10, **config) -> HumanGameApp:
    """Create human game app using canonical factory pattern.
    
    Args:
        grid_size: Size of the game grid (default: 10)
        **config: Additional configuration options
        
    Returns:
        Configured HumanGameApp instance
        
    Design Pattern: Factory Function (Convenience Wrapper)
    Purpose: Provides simple function interface to WebAppFactory.create()
    Educational Value: Shows how to provide multiple interfaces to factories
    Extension Pattern: Extensions can copy this pattern for their app creation
    
    Example:
        >>> app = create_human_app(grid_size=15)
        >>> app.run()
    """
    return WebAppFactory.create("human", grid_size=grid_size, **config)


def create_llm_app(provider: str = "hunyuan", model: str = "hunyuan-turbos-latest",
                   grid_size: int = 10, **config) -> LLMGameApp:
    """Create LLM game app using canonical factory pattern.
    
    Args:
        provider: LLM provider name (default: 'hunyuan')
        model: LLM model name (default: 'hunyuan-turbos-latest')
        grid_size: Size of the game grid (default: 10)
        **config: Additional configuration options
        
    Returns:
        Configured LLMGameApp instance
        
    Design Pattern: Factory Function (Convenience Wrapper)
    Purpose: Provides simple function interface to WebAppFactory.create()
    Educational Value: Shows parameter forwarding in factory functions
    Extension Pattern: Extensions can copy this pattern for their app creation
    
    Note: For full LLM functionality with GameManager integration,
    use scripts/main_web.py which provides complete LLM session management.
    This function provides a lightweight demo interface.
    
    Example:
        >>> app = create_llm_app(provider='deepseek', model='deepseek-chat')
        >>> app.run()
    """
    return WebAppFactory.create("llm", provider=provider, model=model, 
                               grid_size=grid_size, **config)


def create_replay_app(log_dir: str, game_number: int = 1, **config) -> ReplayGameApp:
    """Create replay game app using canonical factory pattern.
    
    Args:
        log_dir: Directory containing game logs
        game_number: Game number to replay (default: 1)
        **config: Additional configuration options
        
    Returns:
        Configured ReplayGameApp instance
        
    Design Pattern: Factory Function (Convenience Wrapper)
    Purpose: Provides simple function interface to WebAppFactory.create()
    Educational Value: Shows required vs optional parameter handling
    Extension Pattern: Extensions can copy this pattern for their app creation
    
    Example:
        >>> app = create_replay_app('logs/session_20250101_120000', game_number=3)
        >>> app.run()
    """
    return WebAppFactory.create("replay", log_dir=log_dir, game_number=game_number, **config)


# ------------------
# Application Entry Points (following scripts/main.py pattern)
# ------------------

class HumanWebApplication:
    """
    OOP wrapper for human web game application.
    
    Following the pattern from scripts/main.py's MainApplication class,
    this provides a structured approach to web application lifecycle.
    
    Design Pattern: Facade Pattern + Template Method Pattern
    Purpose: Encapsulates web application setup, execution, and cleanup
    Educational Value: Shows how to structure application lifecycle
    Extension Pattern: Extensions can copy this pattern for their web apps
    """
    
    def __init__(self, args=None):
        """Initialize human web application.
        
        Args:
            args: Parsed command line arguments (optional, will parse if None)
        """
        self.args = args or parse_human_arguments()
        self.app = None
        
        print_log("Initialized HumanWebApplication")
    
    def setup_application(self) -> None:
        """Set up the web application and validate configuration."""
        try:
            # Create application using factory pattern
            self.app = WebAppFactory.create(
                "human",
                grid_size=self.args.grid_size,
                port=self.args.port,
                debug=self.args.debug
            )
            print_log("Human web application setup complete")
            
        except Exception as e:
            print_log(f"Error setting up application: {e}")
            raise
    
    def run_application(self) -> None:
        """Run the complete web application."""
        try:
            self.setup_application()
            
            print_log("🐍 Starting Snake Game - Human Web Interface")
            print_log(f"📐 Grid: {self.args.grid_size}x{self.args.grid_size}")
            print_log(f"🌐 Server: http://{self.args.host}:{self.app.port}")
            print_log("🎮 Use arrow keys or WASD to control the snake")
            print()
            
            # Run the Flask application
            self.app.run(
                host=self.args.host,
                port=self.args.port,
                debug=self.args.debug
            )
            
        except KeyboardInterrupt:
            print_log("⚠️ Application interrupted by user")
        except Exception as e:
            print_log(f"❌ Fatal error: {e}")
            raise


def main_human():
    """Main entry point for human web interface.
    
    Following the pattern from scripts/main.py.
    
    Educational Value: Shows clean application entry point
    Extension Pattern: Extensions can copy this pattern
    """
    app = HumanWebApplication()
    app.run_application()


if __name__ == "__main__":
    # Allow running as standalone script for testing
    main_human()

 