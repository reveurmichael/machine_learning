"""
Heuristics v0.04 CLI Entry Point
----------------

CLI interface for heuristics v0.04 (moved from root to scripts/ folder).
This provides a clean command-line interface while supporting the new
web-based interface as the primary entry point.

Usage:
    python scripts/main.py --algorithm bfs --max-games 5
    python scripts/main.py --help

Features:
- Extends BaseGameManager for session management
- Multiple heuristic algorithms: BFS, BFS-Safe-Greedy
- Generates game_N.json and summary.json files
- No GUI dependencies (headless by default)
- Organized code structure with agents/ package
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from utils.path_utils import ensure_project_root
from utils.print_utils import print_error, print_info, print_warning, print_success

# Import the extension components using relative imports (extension-specific)
# Add parent directory to sys.path to enable relative imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from game_manager import HeuristicGameManager
from agents import get_available_algorithms, DEFAULT_ALGORITHM

# Get available algorithms from agents package
AVAILABLE_ALGORITHMS = get_available_algorithms()

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for v0.04 multi-algorithm support with automatic dataset updates.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Heuristics v0.04 - Multi-Algorithm Snake Game Agents with Automatic Dataset Updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Core game settings
    parser.add_argument(
        "--algorithm",
        type=str,
        default=DEFAULT_ALGORITHM,
        choices=AVAILABLE_ALGORITHMS,
        help=f"Heuristic algorithm to use. Available: {', '.join(AVAILABLE_ALGORITHMS)} (default: {DEFAULT_ALGORITHM})"
    )
    
    parser.add_argument(
        "--max-games",
        type=int,
        default=5,
        help="Maximum number of games to play (default: 5)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
        help="Maximum steps per game (default: 800)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Size of the game grid (default: 10)"
    )
    
    # v0.02 specific settings
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (shows algorithm details)"
    )
    
    # Force headless mode (no GUI support in heuristics)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=True,
        help="Disable GUI (always true for heuristics extensions)"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.max_games <= 0:
        raise ValueError("max-games must be positive")
        
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
        
    if args.grid_size < 4:
        raise ValueError("grid-size must be at least 4")
        
    if args.grid_size > 20:
        raise ValueError("grid-size should not exceed 20 (performance reasons)")
        
    # Ensure heuristics extensions are always headless
    args.no_gui = True


def main() -> None:
    """Main entry point for heuristics v0.04 with automatic dataset updates."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Display configuration
    if args.verbose:
        print_info("Heuristics v0.04 - Multi-Algorithm Snake Game Agents")
        print_info("=" * 50)
        print_info(f"Algorithm: {args.algorithm}")
        print_info(f"Grid size: {args.grid_size}x{args.grid_size}")
        print_info(f"Max games: {args.max_games}")
        print_info(f"Max steps per game: {args.max_steps}")
        print_info("")
    
    # Create and initialize game manager
    game_manager = HeuristicGameManager(args)
    game_manager.initialize()
    
    # Run the games
    game_manager.run()
    print_success("Heuristics v0.04 execution completed successfully!")


if __name__ == "__main__":
    ensure_project_root()
    main() 