#!/usr/bin/env python3
"""CLI – Reinforcement Learning v0.03

Same flags as v0.02 but imports the upgraded manager (with dashboard hooks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from importlib import import_module

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.path_utils import ensure_project_root  # type: ignore

ensure_project_root()

RLGameManager = import_module("extensions.reinforcement-v0.03.game_manager").RLGameManager


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL v0.03 – Training CLI")
    p.add_argument("--algorithm", choices=["DQN", "PPO", "A3C", "SAC"], default="DQN")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "logs/extensions/models"))
    p.add_argument("--no-gui", action="store_true", default=True)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manager = RLGameManager(args)
    manager.run()


if __name__ == "__main__":
    main() 