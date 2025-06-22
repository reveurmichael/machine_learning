#!/usr/bin/env python3
"""
Heuristics v0.01 - Simple BFS Snake Agent
=========================================

Minimal proof of concept for extending base classes.
"""

import argparse
import sys
import os
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from game_manager import HeuristicGameManager


def create_argument_parser() -> argparse.ArgumentParser:
    """Create simple argument parser."""
    parser = argparse.ArgumentParser(
        description="BFS Snake Game Agent - Proof of Concept",
        epilog="""
Examples:
    python main.py --max-games 3
    python main.py --max-games 5 --max-steps 500
        """
    )
    
    parser.add_argument("--max-games", type=int, default=3, help="Number of games (default: 3)")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per game (default: 800)")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size (default: 10)")
    
    return parser


def main() -> None:
    """Main entry point."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Force headless mode
        setattr(args, "no_gui", True)
        
        # Run BFS session
        game_manager = HeuristicGameManager(args)
        game_manager.initialize()
        game_manager.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if "HEURISTIC_DEBUG" in os.environ:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 