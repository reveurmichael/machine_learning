#!/usr/bin/env python3
"""
Heuristics v0.03 Main Entry Point - Hamiltonian Cycle
====================================================

This script provides the main entry point for running Hamiltonian cycle
algorithm on the Snake game. It demonstrates the most advanced Snake
algorithm that guarantees infinite survival.

Usage:
    python -m extensions.heuristics-v0.03.main --algorithm Hamiltonian --max-games 5
    python -m extensions.heuristics-v0.03.main --help

Features:
- Extends BaseGameManager for session management
- Uses Hamiltonian cycle algorithm with 100% survival guarantee
- Pre-computed cycle generation for any grid size
- Conservative shortcutting for efficiency improvements
- Generates game_N.json and summary.json files
- No GUI dependencies (headless by default)
- Compatible with Task-0 log format
"""

import argparse
import sys
from pathlib import Path

# Add the root directory to the Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import the components
from game_manager import HeuristicGameManager


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser compatible with Task-0 format.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Heuristics v0.03 - Hamiltonian Cycle Snake Game Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 5 games with Hamiltonian algorithm
    python -m extensions.heuristics-v0.03.main --algorithm Hamiltonian --max-games 5
    
    # Run 10 games with custom step limit (may run infinitely!)
    python -m extensions.heuristics-v0.03.main --max-games 10 --max-steps 1000
    
    # Quick test run
    python -m extensions.heuristics-v0.03.main --max-games 1
        """
    )
    
    # Core game settings
    parser.add_argument(
        "--algorithm",
        type=str,
        default="Hamiltonian",
        choices=["Hamiltonian", "HAMILTON", "CYCLE"],  # Hamiltonian cycle variations
        help="Heuristic algorithm to use (default: Hamiltonian)"
    )
    
    parser.add_argument(
        "--max-games",
        type=int,
        default=3,
        help="Maximum number of games to play (default: 3)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per game (default: 1000, may run infinitely with Hamiltonian!)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Size of the game grid (default: 10)"
    )
    
    # Logging settings
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom log directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Compatibility with Task-0 arguments
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=True,
        help="Disable GUI (always true for heuristics)"
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
        
    if args.grid_size < 5:
        raise ValueError("grid-size must be at least 5")
        
    # Ensure heuristics are always headless
    args.no_gui = True


def main() -> None:
    """
    Main entry point for the heuristics extension.
    
    This function:
    1. Parses command line arguments
    2. Creates and initializes the HeuristicGameManager
    3. Runs the game session
    4. Handles cleanup and error reporting
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Validate arguments
        validate_arguments(args)
        
        # Create and initialize game manager
        game_manager = HeuristicGameManager(args)
        game_manager.initialize()
        
        # Run the game session
        game_manager.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(1)
        
    except ValueError as e:
        print(f"❌ Argument error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 