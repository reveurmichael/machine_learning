#!/usr/bin/env python3
"""
Heuristics v0.01 - Simple BFS Snake Agent
=========================================

Minimal proof of concept extension.
"""

import sys
import pathlib

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.path_utils import ensure_project_root_on_path
ensure_project_root_on_path()

import argparse
from game_manager import HeuristicGameManager


def main() -> None:
    """Run BFS heuristic agent."""
    parser = argparse.ArgumentParser(description="BFS Snake Agent")
    parser.add_argument("--max-games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per game")
    
    args = parser.parse_args()
    
    manager = HeuristicGameManager(args)
    manager.initialize()
    manager.run()


if __name__ == "__main__":
    main() 