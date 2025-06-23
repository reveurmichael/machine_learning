#!/usr/bin/env python3
"""
Heuristics v0.01 - Simple BFS Snake Agent
=========================================

Minimal proof of concept extension.
"""

import sys
import os
import pathlib

def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    current = start.resolve()
    for _ in range(10):
        if (current / "config").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("Could not locate repository root containing 'config/' folder")

project_root = _find_repo_root(pathlib.Path(__file__))
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

import argparse
from game_manager import HeuristicGameManager


def main() -> None:
    """Run BFS heuristic agent."""
    parser = argparse.ArgumentParser(description="BFS Snake Agent")
    parser.add_argument("--max-games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per game")

    
    args = parser.parse_args()
    
    # Enforce head-less mode: heuristics extensions v0.01 have no GUI.
    args.no_gui = True
    
    manager = HeuristicGameManager(args)
    manager.initialize()
    manager.run()


if __name__ == "__main__":
    main() 