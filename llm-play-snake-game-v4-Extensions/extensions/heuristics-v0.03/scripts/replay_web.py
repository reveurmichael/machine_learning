#!/usr/bin/env python3
"""
Heuristics Flask Web Replay Script
=================================

Flask-based web replay for heuristic algorithm games.
Follows Task-0 replay_web.py pattern with heuristic-specific enhancements.

Usage:
    python scripts/replay_web.py --log-dir logs/heuristics-bfs_20231201_120000 --game 1
    python scripts/replay_web.py --log-dir logs/heuristics-astar_20231201_120000 --game 2 --port 5001

Features:
- Flask web interface for replay visualization
- Algorithm-specific performance metrics display
- Real-time move progression with controls
- Compatible with all 7 heuristic algorithms
- Responsive web UI with heuristic insights
"""

import sys
from pathlib import Path
import argparse

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

# Add parent directory (heuristics-v0.03) to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import Task-0 utilities
from utils.network_utils import ensure_free_port, random_free_port

# Import heuristic-specific web replay
from replay_gui import HeuristicReplayGUI


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for heuristic web replay.
    
    Follows Task-0 replay_web.py argument pattern.
    """
    parser = argparse.ArgumentParser(
        description="Heuristic Web Replay - Web-based replay for heuristic algorithm games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start web replay for BFS game
    python scripts/replay_web.py --log-dir logs/heuristics-bfs_20231201_120000 --game 1
    
    # Start web replay on specific port
    python scripts/replay_web.py --log-dir logs/heuristics-astar_20231201_120000 --game 2 --port 5001
    
    # Start with custom move interval
    python scripts/replay_web.py --log-dir logs/heuristics-hamiltonian_20231201_120000 --game 1 --interval 1000
    
    # Enable debug mode
    python scripts/replay_web.py --log-dir logs/heuristics-bfs-safe-greedy_20231201_120000 --game 1 --debug
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing heuristic game logs"
    )
    
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Game number to replay (default: 1)"
    )
    
    # Optional web server settings
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for web server (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web server (default: 127.0.0.1)"
    )
    
    # Replay settings
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="Move interval in milliseconds (default: 500)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode"
    )
    
    # Heuristic-specific options
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        default=True,
        help="Show algorithm performance metrics (default: enabled)"
    )
    
    parser.add_argument(
        "--show-path-info",
        action="store_true",
        default=True,
        help="Show pathfinding information (default: enabled)"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise ValueError(f"Log directory does not exist: {args.log_dir}")
    
    if not log_dir.is_dir():
        raise ValueError(f"Log path is not a directory: {args.log_dir}")
    
    # Check for game file
    game_file = log_dir / f"game_{args.game}.json"
    if not game_file.exists():
        raise ValueError(f"Game file does not exist: {game_file}")
    
    if args.port is not None and (args.port < 1024 or args.port > 65535):
        raise ValueError("Port must be between 1024 and 65535")
    
    if args.interval < 100:
        raise ValueError("Move interval must be at least 100ms")


def setup_port(args: argparse.Namespace) -> int:
    """
    Set up web server port.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Port number to use
        
    Raises:
        RuntimeError: If no free port can be found
    """
    if args.port is not None:
        # Use specified port if available
        if ensure_free_port(args.port):
            return args.port
        else:
            raise RuntimeError(f"Port {args.port} is not available")
    else:
        # Find random free port
        port = random_free_port()
        if port is None:
            raise RuntimeError("No free ports available")
        return port


def main() -> None:
    """
    Main entry point for heuristic web replay.
    
    Creates and runs a Flask web server for replay visualization
    with heuristic-specific features and metrics.
    """
    try:
        # Parse and validate arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_arguments(args)
        
        # Set up port
        port = setup_port(args)
        
        print("🌐 Starting Heuristic Web Replay Server")
        print(f"📁 Log Directory: {args.log_dir}")
        print(f"🎯 Game Number: {args.game}")
        print(f"🌍 Server: http://{args.host}:{port}")
        print(f"⏱️  Move Interval: {args.interval}ms")
        print(f"🔧 Debug Mode: {args.debug}")
        print(f"📊 Show Metrics: {args.show_metrics}")
        print(f"🛤️  Show Path Info: {args.show_path_info}")
        
        # Create and configure web GUI
        web_gui = HeuristicReplayGUI(
            log_dir=str(Path(args.log_dir)),
            game_number=args.game,
            move_interval=args.interval,
            show_metrics=args.show_metrics,
            show_path_info=args.show_path_info
        )
        
        # Load game data
        if not web_gui.load_game_data():
            print(f"❌ Failed to load game {args.game}")
            sys.exit(1)
        
        print("✅ Game data loaded successfully")
        
        # Detect algorithm from log directory
        log_dir_name = Path(args.log_dir).name
        if "heuristics-" in log_dir_name:
            algorithm = log_dir_name.split("heuristics-")[1].split("_")[0].upper()
            print(f"🧠 Algorithm detected: {algorithm}")
        
        print("\n🚀 Starting web server...")
        print(f"📱 Open your browser and navigate to: http://{args.host}:{port}")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Run Flask application
        web_gui.run(
            host=args.host,
            port=port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Web server stopped by user")
        sys.exit(0)
        
    except ValueError as e:
        print(f"❌ Argument error: {e}")
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if 'args' in locals() and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 