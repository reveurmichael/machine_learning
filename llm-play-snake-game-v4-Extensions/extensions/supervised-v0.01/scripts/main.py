from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add extension dir and project root to sys.path for local imports
_current_dir = Path(__file__).resolve().parent
_ext_dir = _current_dir.parent
sys.path.insert(0, str(_ext_dir))
sys.path.insert(0, str(_ext_dir.parent.parent))

from agents import create, get_available_algorithms, DEFAULT_ALGORITHM  # type: ignore  # noqa: E402
from game_manager import SupervisedGameManager  # type: ignore  # noqa: E402

DEFAULT_GRID_SIZE = 10
DEFAULT_MAX_GAMES = 10
DEFAULT_MAX_STEPS = 500

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised v0.01 runner")
    parser.add_argument("--algorithm", type=str, default=DEFAULT_ALGORITHM)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    alg = args.algorithm
    if alg not in get_available_algorithms():
        print(f"Unknown algorithm '{alg}', defaulting to {DEFAULT_ALGORITHM}")
        alg = DEFAULT_ALGORITHM

    agent = create(alg)

    gm_args = argparse.Namespace(
        algorithm=alg,
        grid_size=args.grid_size,
        max_games=args.max_games,
        max_steps=args.max_steps,
        verbose=args.verbose,
        no_gui=True,
    )

    manager = SupervisedGameManager(gm_args, agent=agent)
    manager.initialize()
    manager.run()
    print("Supervised v0.01 completed")


if __name__ == "__main__":
    main()