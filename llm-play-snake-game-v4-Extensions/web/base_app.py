"""
Simple Flask Base Application for Snake Game AI
===============================================

Provides minimal, KISS-compliant Flask foundation for Task-0 and all extensions.
Follows no-over-preparation principle: builds only what's needed now.

Design Philosophy:
- KISS: Simple, straightforward Flask application base
- DRY: Single base class, no complex inheritance hierarchy  
- No Over-Preparation: Only implements features actually used
- Extensible: Easy for Tasks 1-5 to inherit and extend

Educational Value:
- Shows simple, clean Flask application patterns
- Demonstrates minimal viable web infrastructure
- Provides extensible base without over-engineering

Extension Pattern:
Tasks 1-5 inherit from FlaskGameApp and override methods as needed.
"""

import os
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request

from utils.network_utils import random_free_port, ensure_free_port
from config.web_constants import FLASK_DEBUG_MODE


class FlaskGameApp:
    """
    Simple Flask application base for Snake Game AI.
    
    Provides minimal Flask infrastructure needed by Task-0 and extensions.
    Uses KISS principle: simple, direct, no unnecessary abstraction.
    
    Design Pattern: Template Method Pattern (Simple Implementation)
    Purpose: Minimal Flask foundation with extension points
    Educational Value: Shows clean, simple Flask application design
    """
    
    def __init__(self, name: str = "Snake Game", port: Optional[int] = None):
        """Initialize Flask application with automatic port allocation."""
        self.name = name
        self.port = port or random_free_port()
        
        # Simple Flask setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        
        # Setup routes
        self.setup_routes()
        
        print(f"[{self.name}] Initialized on port {self.port}")
    
    def setup_routes(self) -> None:
        """Setup basic Flask routes. Override in subclasses for specific functionality."""
        @self.app.route('/')
        def index():
            """Main page route."""
            return render_template(
                self.get_template_name(), 
                debug_mode=FLASK_DEBUG_MODE,
                **self.get_template_data()
            )
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'app': self.name})
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Favicon route."""
            return self.app.send_static_file('favicon.ico')
    
    def get_template_name(self) -> str:
        """Get template name. Override in subclasses."""
        return 'base.html'
    
    def get_template_data(self) -> Dict[str, Any]:
        """Get template data. Override in subclasses."""
        return {'name': self.name, 'status': 'ready'}
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = FLASK_DEBUG_MODE):
        """Run Flask application."""
        actual_port = ensure_free_port(port or self.port)
        self.port = actual_port
        
        print(f"[{self.name}] Starting on http://{host}:{actual_port}")
        self.app.run(host=host, port=actual_port, debug=debug)
    
    @property
    def url(self) -> str:
        """Get application URL."""
        return f"http://127.0.0.1:{self.port}"


class GameFlaskApp(FlaskGameApp):
    """
    Flask application with game-specific routes.
    
    Adds game control API routes needed by Task-0.
    Extensions can inherit and override methods as needed.
    """
    
    def __init__(self, name: str = "Snake Game", port: Optional[int] = None):
        """Initialize game Flask application."""
        super().__init__(name, port)
        self.setup_game_routes()
    
    def setup_game_routes(self) -> None:
        """Setup game-specific API routes."""
        @self.app.route('/api/state')
        def get_state():
            """Get current game state."""
            return jsonify(self.get_game_state())
        
        @self.app.route('/api/move', methods=['POST'])
        def make_move():
            """Handle game moves."""
            data = request.get_json() or {}
            result = self.handle_move(data)
            return jsonify(result)
        
        @self.app.route('/api/control', methods=['POST'])
        def control():
            """Handle game controls."""
            data = request.get_json() or {}
            result = self.handle_control(data)
            return jsonify(result)
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset():
            """Reset game."""
            result = self.handle_reset()
            return jsonify(result)
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get game state. Override in subclasses."""
        return {'status': 'no game state'}
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move. Override in subclasses."""
        return {'status': 'not implemented'}
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control. Override in subclasses."""
        return {'status': 'not implemented'}
    
    def handle_reset(self) -> Dict[str, Any]:
        """Handle reset. Override in subclasses."""
        return {'status': 'not implemented'}


# Simple factory functions (no complex factory classes)
def create_flask_app(name: str = "Snake Game", port: Optional[int] = None) -> FlaskGameApp:
    """Create basic Flask application."""
    return FlaskGameApp(name, port)


def create_game_app(name: str = "Snake Game", port: Optional[int] = None) -> GameFlaskApp:
    """Create game Flask application."""
    return GameFlaskApp(name, port) 