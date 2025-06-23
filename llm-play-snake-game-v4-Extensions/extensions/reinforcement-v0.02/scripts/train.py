#!/usr/bin/env python3
"""CLI – Reinforcement Learning v0.02

Train various RL algorithms via a single command-line interface.
Only *head-less* mode is supported – replay & GUI arrive in v0.03.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from importlib import import_module  # noqa: E402

# Add repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root  # type: ignore

ensure_project_root()

# Safe import even with hyphen in package name
RLGameManager = import_module("extensions.reinforcement-v0.02.game_manager").RLGameManager


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reinforcement Learning v0.02 – Training CLI")
    p.add_argument("--algorithm", choices=["DQN", "PPO", "A3C", "SAC"], default="DQN")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "logs/extensions/models"))
    p.add_argument("--no-gui", action="store_true", help="Always head-less in v0.02 (flag kept for parity)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 72)
    print(f"RL v0.02 – Training {args.algorithm} on grid {args.grid_size}x{args.grid_size}")
    print("=" * 72)

    manager = RLGameManager(args)
    manager.run()


if __name__ == "__main__":
    main() 