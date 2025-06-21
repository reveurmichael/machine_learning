"""
Snake Game - Human Player Web Interface (MVC Architecture)
=========================================================

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

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import MVC components
from web.factories import create_web_application
from core.game_controller import GameController
from utils.network_utils import find_free_port

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


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
        default=15,
        help="Size of the game grid (default: 15)"
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
        
        # Create game controller
        game_controller = GameController(
            grid_size=args.grid_size,
            use_gui=False  # Web interface, no pygame GUI
        )
        
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
