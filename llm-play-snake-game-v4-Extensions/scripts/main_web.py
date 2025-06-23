"""
Snake Game - Task 0 Main Web Interface with MVC Architecture
--------------------

Flask-based web application for LLM-driven Snake gameplay using the new MVC architecture.
This script demonstrates how Task 0 integrates with the MVC web framework.

Features:
- MVC architecture with role-based controllers
- Observer pattern for real-time updates
- Factory pattern for component creation
- Template Method pattern for request processing

This whole module is Task0 specific but uses the generic MVC framework.
"""

import sys
import pathlib
import logging
import argparse
import threading

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import MVC components
from web.models import LoggingObserver
from web.controllers import GamePlayController

# Import Task 0 components
from core.game_manager import GameManager
from llm.agent_llm import LLMSnakeAgent
from scripts.main import parse_arguments
from utils.network_utils import find_free_port

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# Import the new OOP controller classes
from core.game_controller import GameControllerAdapter


class MainController(GamePlayController):
    """
    Task 0 specific LLM controller that extends the base LLM controller.
    
    Adds Task 0 specific functionality while using the MVC framework.
    Demonstrates how task-specific controllers can extend the base framework.
    """
    
    def __init__(self, model_manager, view_renderer, game_manager: GameManager, **config):
        """
        Initialize Task 0 LLM controller.
        
        Args:
            model_manager: Game state model
            view_renderer: View rendering system
            game_manager: Task 0 GameManager instance
            **config: Controller configuration
        """
        super().__init__(model_manager, view_renderer, **config)
        self.game_manager = game_manager
        self.llm_agent = game_manager.agent
        
        # Task 0 specific state
        self.llm_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_response_time': 0.0,
            'last_response': None,
            'current_provider': getattr(game_manager.args, 'provider', 'unknown'),
            'current_model': getattr(game_manager.args, 'model', 'unknown')
        }
        
        logger.info("Initialized LLMController")
    
    def handle_state_request(self, context) -> dict:
        """Handle state requests with Task 0 specific enhancements."""
        # Get base state from parent
        base_state = super().handle_state_request(context)
        
        # Add Task 0 specific information
        base_state.update({
            "task_type": "task0_llm",
            "game_number": getattr(self.game_manager, 'game_count', 0) + 1,
            "round_count": getattr(self.game_manager, 'round_count', 0),
            "llm_stats": self.llm_stats.copy(),
            "planned_moves": getattr(self.game_manager.game, 'planned_moves', []),
            "llm_response": getattr(self.game_manager.game, 'processed_response', ''),
            "pause_between_moves": self.game_manager.get_pause_between_moves(),
            "running": getattr(self.game_manager, 'running', False)
        })
        
        return base_state
    
    def handle_control_request(self, context) -> dict:
        """Handle control requests with Task 0 specific commands."""
        command = context.data.get("command", "").lower()
        
        # Handle Task 0 specific commands
        if command == "pause":
            self.game_manager.running = False
            return {
                "status": "success",
                "message": "Game paused",
                "running": self.game_manager.running
            }
        
        elif command == "resume":
            self.game_manager.running = True
            return {
                "status": "success",
                "message": "Game resumed",
                "running": self.game_manager.running
            }
        
        # Delegate to parent for other commands
        return super().handle_control_request(context)
    
    def get_index_template_name(self) -> str:
        """Get template name for Task 0 LLM game page."""
        return "main.html"
    
    def get_index_template_context(self) -> dict:
        """Get template context for Task 0 LLM game page."""
        return {
            "controller_name": "Task 0 LLM Controller",
            "game_mode": "llm",
            "task_type": "task0",
            "llm_provider": self.llm_stats.get('current_provider', 'unknown'),
            "llm_model": self.llm_stats.get('current_model', 'unknown'),
            "features": [
                "LLM-driven gameplay",
                "Real-time state updates",
                "Game statistics tracking",
                "Pause/resume controls",
                "Performance monitoring"
            ]
        }


def create_mvc_application(game_args) -> tuple:
    """
    Create Task 0 MVC web application.
    
    Args:
        game_args: Parsed command line arguments
        
    Returns:
        Tuple of (Flask app, Controller, GameManager)
    """
    logger.info("Creating Task 0 MVC application...")
    
    # Create GameManager for Task 0
    game_manager = GameManager(game_args)
    game_manager.agent = LLMSnakeAgent(
        game_manager, 
        provider=game_args.provider, 
        model=game_args.model
    )
    
    # Create adapter for MVC compatibility
    game_controller_adapter = GameControllerAdapter(game_manager)
    
    # Create MVC components using factory
    from web.factories import ModelFactory, ViewRendererFactory, ControllerFactory
    
    model_factory = ModelFactory()
    view_factory = ViewRendererFactory()
    controller_factory = ControllerFactory()
    
    # Create model with live game state provider
    model = model_factory.create_live_game_model(game_controller_adapter, "llm")
    
    # Create view renderer
    view_renderer = view_factory.create_renderer(
        template_folder=str(REPO_ROOT / "web" / "templates"),
        static_folder=str(REPO_ROOT / "web" / "static")
    )
    
    # Register Task 0 specific controller
    controller_factory.register_controller_type("llm", MainController)
    
    # Create controller with Task 0 specific parameters
    controller = controller_factory.create_controller(
        "llm",
        model,
        view_renderer,
        game_manager=game_manager
    )
    
    # Add observers for monitoring
    logging_observer = LoggingObserver(logging.INFO)
    model.add_observer(logging_observer)
    
    # Create Flask app manually for custom routes
    from flask import Flask
    app = Flask(__name__, 
                template_folder=str(REPO_ROOT / "web" / "templates"),
                static_folder=str(REPO_ROOT / "web" / "static"))
    
    # Register MVC routes
    from web.controllers.base_controller import RequestType
    from flask import request
    
    @app.route('/')
    def index():
        return controller.handle_request(request, RequestType.INDEX_GET)
    
    @app.route('/api/state')
    def api_state():
        return controller.handle_request(request, RequestType.STATE_GET)
    
    @app.route('/api/control', methods=['POST'])
    def api_control():
        return controller.handle_request(request, RequestType.CONTROL_POST)
    
    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        return controller.handle_request(request, RequestType.RESET_POST)
    
    @app.route('/api/health')
    def api_health():
        return controller.handle_request(request, RequestType.HEALTH_GET)
    
    logger.info("Task 0 MVC application created successfully")
    return app, controller, game_manager


def run_game_manager_thread(game_manager: GameManager, game_args):
    """
    Run GameManager in background thread.
    
    Args:
        game_manager: GameManager instance
        game_args: Command line arguments
    """
    try:
        # Handle continuation if specified
        cont_dir = getattr(game_args, "continue_with_game_in_dir", None)
        if cont_dir:
            logger.info(f"Continuing from session: {cont_dir}")
            # Continuation logic would go here
            # For now, just run normally
        
        # Run the game manager
        game_manager.run()
        
    except Exception as e:
        logger.error(f"GameManager thread crashed: {e}", exc_info=True)


def main():
    """Main entry point for Task 0 MVC web interface."""

    # Parse command line arguments
    host_port_parser = argparse.ArgumentParser(add_help=False)
    host_port_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    host_port_parser.add_argument("--port", type=int, default=find_free_port(8000), help="Port number")
    host_port_args, remaining_argv = host_port_parser.parse_known_args()

    # Parse game arguments
    argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        game_args = parse_arguments()
    finally:
        sys.argv = argv_backup

    # The web interface is a GUI, so we should not force headless mode.
    # The GameManager uses the `use_gui` flag (derived from `no_gui`)
    # to enable features like pause_between_moves, which are essential for
    # the web experience.
    # game_args.no_gui = True # This was causing the no-pause issue

    print("\n🐍 Snake Game - Task 0 MVC Web Interface")
    print(f"🔗 URL: http://{host_port_args.host}:{host_port_args.port}")
    print(f"🤖 LLM: {game_args.provider}/{game_args.model}")
    print("📊 Architecture: MVC with Observer Pattern")
    print()
    
    try:
        # Create MVC application
        app, controller, game_manager = create_mvc_application(game_args)
        
        # Start GameManager in background thread
        game_thread = threading.Thread(
            target=run_game_manager_thread,
            args=(game_manager, game_args),
            daemon=True
        )
        game_thread.start()
        
        print("🎯 MVC Components:")
        print(f"   Controller: {controller.__class__.__name__}")
        print(f"   Model: {controller.model_manager.__class__.__name__}")
        print(f"   View: {controller.view_renderer.__class__.__name__}")
        print(f"   Observers: {controller.model_manager.get_observer_count()}")
        print()
        
        print("📡 API Endpoints:")
        print("   GET  /                 - Main game interface")
        print("   GET  /api/state        - Current game state")
        print("   POST /api/control      - Game commands (pause/resume)")
        print("   POST /api/reset        - Reset game")
        print("   GET  /api/health       - System health check")
        print()
        
        # Start Flask server
        logger.info(f"Starting Task 0 MVC web server at http://{host_port_args.host}:{host_port_args.port}")
        app.run(
            host=host_port_args.host,
            port=host_port_args.port,
            threaded=True,
            use_reloader=False,
            debug=False
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start Task 0 MVC web interface: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 