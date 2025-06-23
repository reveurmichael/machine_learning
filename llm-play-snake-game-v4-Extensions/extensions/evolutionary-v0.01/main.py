"""Command-line entry for evolutionary-v0.01.

Usage (headless):

    python -m extensions.evolutionary_v0_01.main --max-games 10 --max-steps 500
"""

from __future__ import annotations

import argparse
from extensions.common.path_utils import setup_extension_paths
from game_manager import EvolutionaryGameManager

setup_extension_paths()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary GA Snake (v0.01)")
    parser.add_argument("--max-games", type=int, default=5, help="Number of games to run")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per game")
    parser.add_argument("--grid-size", type=int, default=10, help="Board side length (N×N)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser


def main() -> None:  # noqa: D401
    args = _build_arg_parser().parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    manager = EvolutionaryGameManager(args)
    manager.initialize()
    manager.run()


if __name__ == "__main__":
    main() 