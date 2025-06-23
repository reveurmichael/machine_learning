#!/usr/bin/env python3
"""
Heuristics PyGame Replay Script
==============================

PyGame-based replay for heuristic algorithm games.
Follows Task-0 replay.py pattern while adding heuristic-specific features.

Usage:
    python scripts/replay.py --log-dir logs/heuristics-bfs_20231201_120000 --game 1
    python scripts/replay.py --log-dir logs/heuristics-astar_20231201_120000 --game 2 --pause 0.5

Features:
- Reuses Task-0 PyGame replay infrastructure
- Algorithm-aware display with performance metrics
- Heuristic-specific console output
- Compatible with all 7 heuristic algorithms
"""

import pathlib
from extensions.common.path_utils import add_repo_root_to_sys_path
add_repo_root_to_sys_path(pathlib.Path(__file__))

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

import argparse
import sys
from pathlib import Path

# Add parent directory (heuristics-v0.03) to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Add root directory to Python path for base classes
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import Task-0 replay infrastructure
from replay.replay_engine import ReplayEngine
from gui.replay_gui import ReplayGUI

# Import heuristic-specific replay engine
from replay_engine import HeuristicReplayEngine


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for heuristic PyGame replay.
    
    Follows Task-0 replay.py argument pattern.
    """
    parser = argparse.ArgumentParser(
        description="Heuristic PyGame Replay - Replay heuristic algorithm games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replay BFS game with default settings
    python scripts/replay.py --log-dir logs/heuristics-bfs_20231201_120000 --game 1
    
    # Replay A* game with slower playback
    python scripts/replay.py --log-dir logs/heuristics-astar_20231201_120000 --game 2 --pause 1.0
    
    # Replay with auto-advance (no manual control)
    python scripts/replay.py --log-dir logs/heuristics-hamiltonian_20231201_120000 --game 1 --auto
    
    # Replay without GUI (console only)
    python scripts/replay.py --log-dir logs/heuristics-bfs-safe-greedy_20231201_120000 --game 1 --no-gui
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
    
    # Optional replay settings
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause between moves in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-advance through moves without user input"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (console output only)"
    )
    
    # Heuristic-specific options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with algorithm details"
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
    
    if args.pause < 0:
        raise ValueError("Pause duration must be non-negative")


def main() -> None:
    """
    Main entry point for heuristic PyGame replay.
    
    Creates and runs either a heuristic-specific or Task-0 compatible
    replay engine based on the log directory contents.
    """
    try:
        # Parse and validate arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_arguments(args)
        
        if args.verbose:
            print("🎮 Starting Heuristic PyGame Replay")
            print(f"📁 Log Directory: {args.log_dir}")
            print(f"🎯 Game Number: {args.game}")
            print(f"⏱️  Pause Duration: {args.pause}s")
            print(f"🤖 Auto Advance: {args.auto}")
            print(f"🖥️  GUI Enabled: {not args.no_gui}")
        
        # Determine replay engine type
        log_dir = Path(args.log_dir)
        if "heuristics-" in log_dir.name:
            # Use heuristic-specific replay engine
            if args.verbose:
                print("🧠 Using heuristic-specific replay engine")
            
            replay_engine = HeuristicReplayEngine(
                log_dir=str(log_dir),
                pause_between_moves=args.pause,
                auto_advance=args.auto,
                use_gui=not args.no_gui
            )
        else:
            # Fallback to Task-0 replay engine
            if args.verbose:
                print("🔄 Using Task-0 compatible replay engine")
            
            replay_engine = ReplayEngine(
                log_dir=str(log_dir),
                pause_between_moves=args.pause,
                auto_advance=args.auto,
                use_gui=not args.no_gui
            )
        
        # Load game data
        if not replay_engine.load_game_data(args.game):
            print(f"❌ Failed to load game {args.game}")
            sys.exit(1)
        
        # Set up GUI if enabled
        if not args.no_gui:
            gui = ReplayGUI()
            replay_engine.set_gui(gui)
            
            if args.verbose:
                print("🖼️  PyGame GUI initialized")
        
        # Run replay
        replay_engine.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Replay interrupted by user")
        sys.exit(1)
        
    except ValueError as e:
        print(f"❌ Argument error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if 'args' in locals() and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 