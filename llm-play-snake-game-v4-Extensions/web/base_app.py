"""
Layered Web Infrastructure for Snake Game AI Project
===================================================

This module provides the foundational web infrastructure used across all tasks and extensions.
It implements a layered architecture with clear inheritance hierarchy and enhanced naming
for maximum clarity and extensibility.

Architecture Layers:
1. BaseWebApp: Universal foundation for all Snake Game web applications
2. SimpleFlaskApp: Task-0 focused application with common conveniences  
3. BaseReplayApp: Universal base for all replay applications across the project

Design Philosophy:
- Universal Foundation: BaseWebApp works for Task-0 and all extensions
- Clear Hierarchy: Each layer adds specific functionality while maintaining compatibility
- Enhanced Naming: Clear, explicit naming that indicates purpose and scope
- Educational Value: Demonstrates layered architecture and inheritance patterns
- Extensibility: Easy for extensions to inherit and customize

Extension Pattern:
Extensions inherit from these base classes to create specialized web applications:
- extensions/heuristics/web/ → inherits from BaseReplayApp for pathfinding visualization
- extensions/supervised/web/ → inherits from BaseReplayApp for training metrics
- extensions/reinforcement/web/ → inherits from BaseReplayApp for reward visualization

Reference: docs/extensions-guideline/final-decision-10.md for SUPREME_RULES
"""

import os
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request

# Import utilities following SSOT principles
from utils.network_utils import random_free_port
from utils.print_utils import create_logger

# Create logger for this module
print_log = create_logger("WebApp")


class BaseWebApp:
    """
    Universal base class for all Snake Game web applications.
    
    This is the foundational class that provides core Flask infrastructure
    used by both Task-0 applications and all extension web applications.
    It focuses on universal functionality while remaining task-agnostic.
    
    Design Pattern: Template Method Pattern (Universal Web Foundation)
    Purpose: Provides universal Flask infrastructure for entire project
    Educational Value: Shows how to create truly universal base classes
    Usage: Base class for all web applications across all tasks and extensions
    
    Key Features:
    - Task-agnostic: No Task-0 specific functionality
    - Universal: Works for all tasks and extensions
    - Minimal: Only core Flask setup and routing
    - Extensible: Easy to inherit and customize
    - Educational: Clear base class design patterns
    
    Inheritance Hierarchy:
    BaseWebApp → SimpleFlaskApp (Task-0)
    BaseWebApp → ExtensionWebApp (extensions/common/web/)
    BaseWebApp → BaseReplayApp (universal replay)
    """
    
    def __init__(self, name: str = "SnakeGameWebApp", port: Optional[int] = None):
        """Initialize universal web application foundation.
        
        Args:
            name: Application name for display and logging
            port: Port number (None for auto-detection)
            
        Educational Value: Shows universal initialization patterns
        """
        self.name = name
        self.port = port or random_free_port()
        
        # Initialize Flask app with universal settings
        self.app = Flask(__name__)
        
        # Configure the application (template method)
        self.configure_app()
        
        # Set up universal routes (template method)
        self.setup_routes()
        
        print_log(f"BaseWebApp initialized: {name} on port {self.port}")
    
    def configure_app(self) -> None:
        """Configure Flask application with universal settings.
        
        Template Method: Subclasses can override for additional configuration
        Educational Value: Shows universal Flask configuration patterns
        """
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        print_log("Universal Flask configuration applied")
    
    def setup_routes(self) -> None:
        """Set up universal Flask routes used by all applications.
        
        Template Method: Common routes that all applications need
        Educational Value: Shows universal routing patterns for web applications
        """
        @self.app.route('/')
        def index():
            """Universal main page route."""
            template_name = self.get_template_name()
            app_data = self.get_app_data()
            return render_template(template_name, **app_data)
        
        @self.app.route('/api/health')
        def health():
            """Universal health check endpoint."""
            return jsonify({
                'status': 'healthy', 
                'app': self.name,
                'type': 'BaseWebApp'
            })
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Universal favicon route."""
            return self.app.send_static_file('favicon.ico')
        
        print_log("Universal routes configured")
    
    # Template methods that subclasses should implement
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get application data for template rendering.
        
        Template Method: Subclasses must implement this
        
        Returns:
            Dictionary with application data for template rendering
            
        Educational Value: Shows template method pattern implementation
        """
        return {
            'name': self.name, 
            'status': 'ready',
            'type': 'universal'
        }
    
    def get_template_name(self) -> str:
        """Get template name for main page.
        
        Template Method: Subclasses can override for custom templates
        
        Returns:
            Template filename
            
        Educational Value: Shows template selection patterns
        """
        return 'base.html'  # Universal base template
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = True):
        """Run the Flask application.
        
        Args:
            host: Host to bind to
            port: Port to bind to (uses self.port if None)
            debug: Enable debug mode
            
        Educational Value: Shows universal Flask application execution
        """
        actual_port = port or self.port
        print_log(f"Starting {self.name} on {host}:{actual_port}")
        self.app.run(host=host, port=actual_port, debug=debug)


class SimpleFlaskApp(BaseWebApp):
    """
    Simple Flask application for Task-0 applications.
    
    Extends BaseWebApp with Task-0 specific conveniences while maintaining
    compatibility with extension requirements. This class adds game-specific
    functionality that Task-0 applications commonly need.
    
    Design Pattern: Template Method Pattern (Task-0 Specific Layer)
    Purpose: Provides Task-0 conveniences while maintaining universal compatibility
    Educational Value: Shows how to add task-specific functionality to universal base
    Usage: Used by Task-0 web applications (human, LLM, basic replay)
    
    Key Features:
    - Game-specific: Adds common game functionality
    - API endpoints: Standard game control and state endpoints
    - Backwards compatible: Maintains compatibility with existing Task-0 code
    - Educational: Shows layered architecture implementation
    
    Inheritance: BaseWebApp → SimpleFlaskApp → Specific Game Apps
    """
    
    def __init__(self, name: str = "SnakeGame", port: Optional[int] = None):
        """Initialize simple Flask application for games.
        
        Args:
            name: Application name for display
            port: Port number (None for auto-detection)
            
        Educational Value: Shows task-specific initialization over universal base
        """
        super().__init__(name, port)
        print_log(f"SimpleFlaskApp initialized: {name}")
    
    def setup_routes(self) -> None:
        """Set up routes including universal routes plus game-specific routes.
        
        Educational Value: Shows how to extend universal routes with specific functionality
        """
        # Set up universal routes first
        super().setup_routes()
        
        # Add game-specific routes
        self.setup_game_routes()
        
        print_log("Game-specific routes added to universal routes")
    
    def setup_game_routes(self) -> None:
        """Set up game-specific Flask routes.
        
        Template Method: Game-specific routes that Task-0 applications use
        Educational Value: Shows game-specific routing patterns
        """
        @self.app.route('/api/state')
        def get_state():
            """API endpoint for game state."""
            state = self.get_api_state()
            return jsonify(state)
        
        @self.app.route('/api/move', methods=['POST'])
        def make_move():
            """API endpoint for making moves."""
            try:
                data = request.get_json() or {}
                # Add move action for backward compatibility
                if 'direction' in data and 'action' not in data:
                    data['action'] = 'move'
                response = self.handle_control(data)
                return jsonify(response)
            except Exception as e:
                print_log(f"Error in move endpoint: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/control', methods=['POST'])
        def control():
            """API endpoint for game controls."""
            try:
                data = request.get_json() or {}
                response = self.handle_control(data)
                return jsonify(response)
            except Exception as e:
                print_log(f"Error in control endpoint: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset():
            """API endpoint for game reset."""
            try:
                response = self.handle_control({'action': 'reset'})
                return jsonify(response)
            except Exception as e:
                print_log(f"Error in reset endpoint: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        print_log("Game-specific API routes configured")
    
    # Template methods that game applications should implement
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get current game state for API.
        
        Template Method: Game applications must implement this
        
        Returns:
            Dictionary with current game state
            
        Educational Value: Shows game state API patterns
        """
        return {'status': 'ready', 'type': 'simple_flask_app'}
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle game control commands.
        
        Template Method: Game applications must implement this
        
        Args:
            data: Control data from client
            
        Returns:
            Response dictionary
            
        Educational Value: Shows game control handling patterns
        """
        return {'status': 'not_implemented', 'type': 'simple_flask_app'}


class BaseReplayApp(BaseWebApp):
    """
    Universal base class for all replay applications across the project.
    
    Provides common replay functionality that can be extended by:
    - Task-0 basic replay (ROOT/web/replay_app.py)
    - Heuristic replay with pathfinding visualization (extensions/heuristics/web/)
    - ML replay with training metrics (extensions/supervised/web/)
    - RL replay with reward visualization (extensions/reinforcement/web/)
    
    Design Pattern: Template Method Pattern (Universal Replay Foundation)
    Purpose: Provides universal replay infrastructure with specialization points
    Educational Value: Shows how to create extensible replay system architecture
    Usage: Base class for all replay applications across all tasks and extensions
    
    Key Features:
    - Universal Replay: Core replay functionality for all tasks
    - Specialization Points: Clear extension points for task-specific features
    - Data Management: Consistent replay data handling patterns
    - API Consistency: Standard replay API across all extensions
    
    Inheritance: BaseWebApp → BaseReplayApp → Specific Replay Apps
    """
    
    def __init__(self, name: str = "SnakeGameReplay", log_dir: str = "", **config):
        """Initialize universal replay application.
        
        Args:
            name: Application name for display
            log_dir: Directory containing game logs
            **config: Additional configuration options
            
        Educational Value: Shows universal replay initialization patterns
        """
        super().__init__(name, config.get('port'))
        self.log_dir = log_dir
        self.config = config
        self.replay_engine = None
        
        # Initialize replay infrastructure (template method)
        self.setup_replay_infrastructure()
        
        print_log(f"BaseReplayApp initialized: {name} with log_dir: {log_dir}")
    
    def setup_routes(self) -> None:
        """Set up routes including universal routes plus replay-specific routes.
        
        Educational Value: Shows how to extend universal routes with replay functionality
        """
        # Set up universal routes first
        super().setup_routes()
        
        # Add replay-specific routes
        self.setup_replay_routes()
        
        print_log("Replay-specific routes added to universal routes")
    
    def setup_replay_routes(self) -> None:
        """Set up replay-specific Flask routes.
        
        Template Method: Universal replay routes that all replay apps use
        Educational Value: Shows universal replay routing patterns
        """
        @self.app.route('/api/replay/state')
        def get_replay_state():
            """API endpoint for replay state."""
            state = self.get_replay_state()
            return jsonify(state)
        
        @self.app.route('/api/replay/control', methods=['POST'])
        def replay_control():
            """API endpoint for replay controls."""
            try:
                data = request.get_json() or {}
                response = self.handle_replay_control(data)
                return jsonify(response)
            except Exception as e:
                print_log(f"Error in replay control endpoint: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/replay/info')
        def replay_info():
            """API endpoint for replay information."""
            info = self.get_replay_info()
            return jsonify(info)
        
        print_log("Universal replay API routes configured")
    
    def setup_replay_infrastructure(self) -> None:
        """Setup common replay infrastructure.
        
        Template Method: Subclasses can override for specialized replay setup
        Educational Value: Shows universal replay infrastructure patterns
        """
        print_log("Universal replay infrastructure initialized")
    
    # Template methods that replay applications should implement
    
    def get_replay_state(self) -> Dict[str, Any]:
        """Get current replay state.
        
        Template Method: Replay applications must implement this
        
        Returns:
            Dictionary with current replay state
            
        Educational Value: Shows replay state management patterns
        """
        return {
            'log_dir': self.log_dir,
            'status': 'ready',
            'type': 'base_replay_app'
        }
    
    def handle_replay_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay control commands.
        
        Template Method: Replay applications must implement this
        
        Args:
            data: Control data from client
            
        Returns:
            Response dictionary
            
        Educational Value: Shows replay control handling patterns
        """
        return {
            'status': 'not_implemented',
            'type': 'base_replay_app'
        }
    
    def get_replay_info(self) -> Dict[str, Any]:
        """Get replay information and metadata.
        
        Template Method: Replay applications can override for additional info
        
        Returns:
            Dictionary with replay information
            
        Educational Value: Shows replay metadata patterns
        """
        return {
            'log_dir': self.log_dir,
            'config': self.config,
            'type': 'base_replay_app'
        }


# =============================================================================
# Convenience Functions for Enhanced Developer Experience
# =============================================================================

def create_base_web_app(name: str = "SnakeGameWebApp", **config) -> BaseWebApp:
    """Create a universal base web application.
    
    Args:
        name: Application name
        **config: Configuration options
        
    Returns:
        BaseWebApp instance
        
    Educational Value: Shows factory function patterns for base classes
    """
    print_log(f"Creating base web app: {name}")
    return BaseWebApp(name, **config)


def create_simple_flask_app(name: str = "SnakeGame", **config) -> SimpleFlaskApp:
    """Create a simple Flask application for Task-0.
    
    Args:
        name: Application name
        **config: Configuration options
        
    Returns:
        SimpleFlaskApp instance
        
    Educational Value: Shows factory function patterns for Task-0 apps
    """
    print_log(f"Creating simple Flask app: {name}")
    return SimpleFlaskApp(name, **config)


def create_base_replay_app(name: str = "SnakeGameReplay", log_dir: str = "", **config) -> BaseReplayApp:
    """Create a universal base replay application.
    
    Args:
        name: Application name
        log_dir: Directory containing game logs
        **config: Configuration options
        
    Returns:
        BaseReplayApp instance
        
    Educational Value: Shows factory function patterns for replay apps
    """
    print_log(f"Creating base replay app: {name} with log_dir: {log_dir}")
    return BaseReplayApp(name, log_dir, **config) 