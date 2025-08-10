from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.path_utils import ensure_project_root
from utils.print_utils import print_info, print_success

_current_dir = Path(__file__).resolve().parent
_ext_dir = _current_dir.parent
sys.path.insert(0, str(_ext_dir))
sys.path.insert(0, str(_ext_dir.parent.parent))

from agents import create, get_available_algorithms, DEFAULT_ALGORITHM  # type: ignore  # noqa: E402
from game_manager import RLV02GameManager  # type: ignore  # noqa: E402

def create_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL v0.02 runner")
    p.add_argument("--algorithm", type=str, default=DEFAULT_ALGORITHM)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--max-games", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    ensure_project_root()
    args = create_argument_parser().parse_args()
    alg = args.algorithm
    if alg not in get_available_algorithms():
        print_info(f"Unknown algorithm '{alg}', defaulting to {DEFAULT_ALGORITHM}")
        alg = DEFAULT_ALGORITHM
    agent = create(alg, model_path=args.model_path)
    gm_args = argparse.Namespace(
        algorithm=alg,
        grid_size=args.grid_size,
        max_games=args.max_games,
        max_steps=args.max_steps,
        verbose=args.verbose,
        no_gui=True,
    )
    manager = RLV02GameManager(gm_args, agent=agent)
    manager.initialize()
    manager.run()
    print_success("RL v0.02 completed")


if __name__ == "__main__":
    main()