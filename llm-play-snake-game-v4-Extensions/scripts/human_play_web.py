"""
Snake Game - Human Player Web Interface (KISS Architecture)
--------------------

Flask-based web application for human-driven Snake gameplay using KISS principles.
This script demonstrates how Task-0 integrates with a simple web architecture
and serves as a foundation for Task 1-5 extensions.

Features:
- Simple Flask integration without complex MVC patterns
- Dynamic port allocation with network utilities
- KISS principles and elegant error handling
- Extensible foundation for future tasks
- Simple logging following SUPREME_RULES

Design Patterns Used:
    - Template Method Pattern: Simple Flask application lifecycle
    - Factory Pattern: Simple factory functions
    - Strategy Pattern: Pluggable game modes

Educational Goals:
    - Demonstrate simple web integration for Task-0
    - Show how future extensions can reuse this pattern
    - Illustrate KISS principles in web applications
    - Provide canonical example of Task-0 web interface

Extension Pattern for Future Tasks:
    Task-1 (Heuristics): Replace with pathfinding algorithms
    Task-2 (RL): Replace with RL agent and training monitoring
    Task-3 (Supervised): Replace with ML model evaluation
    Task-4 (Distillation): Replace with knowledge distillation
    Task-5 (Advanced): Combine multiple AI strategies
"""

import sys
import pathlib
import argparse

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import Task-0 components
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE

# Import simple web framework
from web.game_flask_app import HumanGameApp
from utils.network_utils import get_server_host_port

# Simple logging following SUPREME_RULES
print_log = lambda msg: print(f"[HumanWebApp] {msg}")


class HumanWebApp(HumanGameApp):
    """
    Task-0 Human Player Web Application.
    
    Extends HumanGameApp with human-specific configuration.
    Demonstrates how to specialize the simple Flask application for different game modes.
    
    Design Pattern: Template Method Pattern (Flask Application Lifecycle)
    Educational Value: Shows how to extend simple applications for specific modes
    Extension Pattern: Future tasks can extend this for their specific needs
    """
    
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE, **config):
        """
        Initialize human player web application.
        
        Args:
            grid_size: Size of the game grid
            **config: Additional configuration options
        """
        super().__init__(
            grid_size=grid_size,
            **config
        )
        print_log(f"Initialized for human play with {grid_size}x{grid_size} grid")
    
    def get_application_info(self) -> dict:
        """Get human-specific application information."""
        return {
            "name": "Task-0 Human Player",
            "task_name": "task0",
            "game_mode": "human",
            "grid_size": self.grid_size,
            "url": f"http://127.0.0.1:{getattr(self, 'port', 5000)}",
            "input_method": "keyboard",
            "features": [
                "Human player input",
                "Real-time game state",
                "Web-based interface",
                "Keyboard controls",
                "Score tracking"
            ]
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for human play web interface.
    
    Educational Value: Shows consistent argument handling across Task-0 scripts
    Extension Pattern: Future tasks can extend this with their specific arguments
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Human Player Web Interface (Task-0 Foundation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/human_play_web.py                    # Default settings
  python scripts/human_play_web.py --grid-size 15     # Larger grid
  python scripts/human_play_web.py --port 8080        # Specific port
  python scripts/human_play_web.py --debug            # Debug mode

Extension Pattern:
  Future tasks can copy this script and modify:
  - Replace with their algorithm/model
  - Add task-specific arguments
  - Maintain same elegant structure
        """
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Size of the game grid (default: {DEFAULT_GRID_SIZE})"
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
        "--debug", 
        action="store_true",
        help="Enable Flask debug mode for development"
    )
    
    return parser


def main() -> int:
    """
    Main entry point for human play web interface.
    
    Educational Value: Shows elegant application lifecycle management
    Extension Pattern: Future tasks can copy this exact pattern
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Get host and port using network utilities
        host, port = get_server_host_port(default_host=args.host, default_port=args.port)
        # Network utilities handle environment variables and port conflicts automatically
        
        print_log("🐍 Starting Snake Game - Human Player Web Interface")
        print_log("📊 Architecture: Task-0 KISS Framework")
        print_log("🎮 Mode: Human Player")
        print_log(f"📐 Grid: {args.grid_size}x{args.grid_size}")
        print_log(f"🌐 Server: http://{host}:{port}")
        print()
        
        # Create human game application using simple architecture
        app = HumanWebApp(
            grid_size=args.grid_size
        )
        
        # Show application info
        app_info = app.get_application_info()
        print_log("🎯 Application Information:")
        print_log(f"   Name: {app_info['name']}")
        print_log(f"   Task: {app_info['task_name']}")
        print_log(f"   Mode: {app_info.get('game_mode', 'unknown')}")
        print_log(f"   URL: {app_info['url']}")
        print()
        
        print_log("🎮 Controls:")
        print_log("   Arrow Keys: Move snake")
        print_log("   R: Reset game")
        print_log("   Ctrl+C: Stop server")
        print()
        
        print_log("🚀 Extension Pattern:")
        print_log("   Future tasks can copy this script structure")
        print_log("   Replace with task-specific components")
        print_log("   Maintain same elegant KISS architecture")
        print()
        
        # Start the application server
        print_log("✅ Starting web server...")
        app.run(host=host, port=port, debug=args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 Server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start human play web interface: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
