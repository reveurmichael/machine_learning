"""
Snake Game - Replay Web Interface (MVC Architecture)
--------------------

Flask-based web application for game replay using the MVC framework.
This script demonstrates how Task-0 integrates with the excellent web MVC architecture
for replay functionality and serves as a foundation for Task 1-5 extensions.

Features:
- Clean MVC architecture using web.factories
- Game replay with navigation controls
- Dynamic port allocation with network utilities
- KISS principles and elegant error handling
- Extensible foundation for future tasks
- Simple logging following SUPREME_RULES

Design Patterns Used:
    - Factory Pattern: Uses web.factories for consistent component creation
    - Template Method Pattern: Leverages BaseFlaskApp lifecycle
    - Strategy Pattern: Pluggable replay engines
    - Observer Pattern: Real-time updates via MVC framework

Educational Goals:
    - Demonstrate clean web MVC integration for Task-0 replay
    - Show how future extensions can reuse this pattern
    - Illustrate replay functionality in web applications
    - Provide canonical example of Task-0 replay interface

Extension Pattern for Future Tasks:
    Task-1 (Heuristics): Replay pathfinding algorithm decisions
    Task-2 (RL): Replay RL agent training and decision process
    Task-3 (Supervised): Replay ML model predictions
    Task-4 (Distillation): Replay knowledge distillation process
    Task-5 (Advanced): Replay complex AI strategy comparisons
"""

import sys
import pathlib
import argparse
from typing import Optional

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import replay components
from replay.replay_engine import ReplayEngine

# Import simple web framework
from web.game_flask_app import ReplayGameApp, create_replay_app
from utils.network_utils import get_server_host_port

# Simple logging following SUPREME_RULES
print_log = lambda msg: print(f"[ReplayWebApp] {msg}")


class ReplayWebApp(ReplayGameApp):
    """
    Task-0 Replay Web Application.
    
    Extends ReplayGameApp with replay-specific configuration.
    Demonstrates how to specialize the simple Flask application for replay mode.
    
    Design Pattern: Template Method Pattern (Flask Application Lifecycle)
    Educational Value: Shows how to extend simple applications for replay functionality
    Extension Pattern: Future tasks can extend this for their replay needs
    """
    
    def __init__(self, log_dir: str, game_number: int = 1, **config):
        """
        Initialize replay web application.
        
        Args:
            log_dir: Directory containing game logs to replay
            game_number: Game number to start replay from
            **config: Additional configuration options
        """
        super().__init__(
            log_dir=log_dir,
            game_number=game_number,
            **config
        )
        print_log(f"Initialized for replay from {log_dir}, starting game {game_number}")
    
    def get_application_info(self) -> dict:
        """Get replay-specific application information."""
        return {
            "name": "Task-0 Replay Viewer",
            "task_name": "task0",
            "game_mode": "replay",
            "log_directory": self.log_dir,
            "current_game": self.game_number,
            "url": f"http://127.0.0.1:{getattr(self, 'port', 5000)}",
            "features": [
                "Game replay viewer",
                "Navigation controls",
                "Step-by-step playback",
                "Game state analysis",
                "Performance metrics"
            ]
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for replay web interface.
    
    Educational Value: Shows consistent argument handling across Task-0 scripts
    Extension Pattern: Future tasks can extend this with their replay-specific arguments
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Replay Web Interface (Task-0 Foundation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/replay_web.py --log-dir logs/session_20250101  # Basic replay
  python scripts/replay_web.py --log-dir logs/session_20250101 --game 5  # Start from game 5
  python scripts/replay_web.py --log-dir logs/session_20250101 --port 8080  # Custom port
  python scripts/replay_web.py --log-dir logs/session_20250101 --debug  # Debug mode

Extension Pattern:
  Future tasks can copy this script and modify:
  - Add task-specific replay data formats
  - Customize replay visualization
  - Add algorithm-specific analysis
  - Maintain same elegant MVC structure

Replay Features:
  - Step-by-step game playback
  - Navigation controls (play/pause/seek)
  - Game state inspection
  - Performance analysis
        """
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing game logs to replay"
    )
    
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Game number to start replay from (default: 1)"
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
    Main entry point for replay web interface.
    
    Educational Value: Shows elegant application lifecycle management for replay
    Extension Pattern: Future tasks can copy this exact pattern for their replay needs
    
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
        
        print_log("🎬 Starting Snake Game - Replay Web Interface")
        print_log(f"📊 Architecture: Task-0 MVC Framework")
        print_log(f"🎮 Mode: Game Replay")
        print_log(f"📁 Log Directory: {args.log_dir}")
        print_log(f"🎯 Starting Game: {args.game}")
        print_log(f"🌐 Server: http://{host}:{port}")
        print()
        
        # Create replay application using elegant architecture
        app = ReplayWebApp(
            log_dir=args.log_dir,
            game_number=args.game
        )
        
        # Show application info
        app_info = app.get_application_info()
        print_log("🎯 Application Information:")
        print_log(f"   Name: {app_info['name']}")
        print_log(f"   Task: {app_info['task_name']}")
        print_log(f"   Mode: {app_info.get('game_mode', 'unknown')}")
        print_log(f"   Log Dir: {app_info.get('log_directory', 'unknown')}")
        print_log(f"   Current Game: {app_info.get('current_game', 'unknown')}")
        print_log(f"   URL: {app_info['url']}")
        print()
        
        print_log("🎮 Replay Controls:")
        print_log("   Play/Pause: Control replay playback")
        print_log("   Step Forward/Back: Navigate frame by frame")
        print_log("   Game Selection: Switch between different games")
        print_log("   Speed Control: Adjust playback speed")
        print_log("   Ctrl+C: Stop server")
        print()
        
        print_log("📊 Replay Features:")
        print_log("   Real-time state inspection")
        print_log("   Game statistics display")
        print_log("   Move-by-move analysis")
        print_log("   Performance metrics")
        print()
        
        print_log("🚀 Extension Pattern:")
        print_log("   Future tasks can copy this script structure")
        print_log("   Add task-specific replay visualizations")
        print_log("   Integrate algorithm-specific analysis")
        print_log("   Maintain same elegant MVC architecture")
        print()
        
        # Start the application server
        print_log("✅ Starting web server...")
        app.run(host=host, debug=args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 Server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start replay web interface: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
