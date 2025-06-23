"""
Snake Game - Human Player Web Interface (MVC Architecture)
--------------------

Flask-based web application for human-driven Snake gameplay using the MVC framework.
This script demonstrates how to create a clean web interface using the new MVC architecture.

Features:
- MVC architecture with role-based controllers
- Factory pattern for component creation
- Clean separation of concerns
- Simplified codebase using framework components

This whole module is Task0 specific but uses the generic MVC framework.
"""

import sys
import pathlib
import logging
import argparse

# ------------------
# Ensure repository root is on sys.path **before** any local imports
# ------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root

# ------------------
# Ensure current working directory == repository root
# ------------------
REPO_ROOT = ensure_project_root()

from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE

# Import MVC components
from web.factories import create_web_application
from core.game_logic import GameLogic
from core.game_controller import BaseGameController
from utils.network_utils import find_free_port

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


class HumanGameControllerAdapter(BaseGameController):
    """
    Adapter to make GameLogic compatible with web MVC architecture for human play.
    
    This adapter wraps GameLogic to provide the BaseGameController interface
    expected by the web MVC framework for human player games.
    """
    
    def __init__(self, game_logic: GameLogic):
        """
        Initialize adapter for human play web interface.
        
        Args:
            game_logic: GameLogic instance for core game mechanics
        """
        # Create a mock game manager for the base class
        class MockGameManager:
            def __init__(self, game_logic):
                self.game = game_logic
                self.game_active = True
        
        self.game_logic = game_logic
        mock_manager = MockGameManager(game_logic)
        super().__init__(mock_manager, use_gui=False)
        
        logger.info("Initialized HumanGameControllerAdapter for web interface")
    
    def initialize_session(self) -> None:
        """Initialize web session - handled by external web framework."""
        logger.info("Human play web session initialization")
    
    def execute_main_loop(self) -> None:
        """Execute main loop - handled by external web framework."""
        logger.info("Main loop execution delegated to web framework")
    
    def make_move(self, direction: str) -> tuple[bool, bool]:
        """
        Execute a move through the GameLogic.
        
        Args:
            direction: Movement direction (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Tuple of (game_still_active, apple_eaten)
        """
        try:
            return self.game_logic.make_move(direction)
        except Exception as e:
            logger.error(f"Move execution failed: {e}")
            return False, False
    
    def reset_game(self) -> None:
        """Reset the game to initial state."""
        try:
            self.game_logic.reset()
            # Also reset the mock game manager's game_active flag
            self.game_manager.game_active = True
            logger.info(f"Game reset via human controller adapter. Game over: {self.game_logic.game_over}")
        except Exception as e:
            logger.error(f"Failed to reset game: {e}")
            raise
    
    def reset(self) -> None:
        """Reset method called by the web framework state provider."""
        self.reset_game()
    
    @property
    def game(self):
        """Get the game logic instance."""
        return self.game_logic
    
    @property
    def head_position(self):
        """Get snake head position."""
        return self.game_logic.head_position
    
    @property
    def snake_positions(self) -> list:
        """Get snake positions as list."""
        return self.game_logic.snake_positions.tolist()
    
    @property
    def apple_position(self) -> tuple:
        """Get apple position as tuple."""
        return tuple(self.game_logic.apple_position.tolist())
    
    @property
    def game_over(self) -> bool:
        """Get game over state."""
        return self.game_logic.game_over


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for human play web interface."""
    parser = argparse.ArgumentParser(
        description="Snake Game - Human Player Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port number for the web server (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Size of the game grid (default: {DEFAULT_GRID_SIZE})"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable Flask debug mode"
    )
    
    return parser


def main() -> None:
    """Main entry point for human play web interface."""
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Auto-detect free port if not specified
        port = args.port or find_free_port()
        
        logger.info("🐍 Starting Snake Game - Human Player Web Interface (MVC)")
        logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")
        logger.info(f"Server: http://{args.host}:{port}")
        
        # Create game logic for human play
        game_logic = GameLogic(
            grid_size=args.grid_size,
            use_gui=False  # Web interface, no pygame GUI
        )
        
        # Create adapter to make GameLogic compatible with MVC framework
        game_controller = HumanGameControllerAdapter(game_logic)
        
        # Create MVC web application using factory
        app, controller = create_web_application(
            game_controller=game_controller,
            game_mode="human",
            template_folder=str(REPO_ROOT / "web" / "templates"),
            static_folder=str(REPO_ROOT / "web" / "static")
        )
        
        # Configure Flask app
        app.config['DEBUG'] = args.debug
        
        logger.info("✅ MVC application created successfully")
        logger.info(f"🎮 Open http://{args.host}:{port} in your browser to play!")
        
        # Run Flask application
        app.run(
            host=args.host,
            port=port,
            debug=args.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start human play web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
