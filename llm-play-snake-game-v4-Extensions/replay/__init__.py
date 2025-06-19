"""
Replay package initialization.
This file exposes the replay engine and utility functions for use by the application.
"""

from replay.replay_engine import ReplayEngine, BaseReplayEngine
from replay.replay_utils import load_game_json, parse_game_data
from replay.replay_data import ReplayDataLLM

__all__ = [
    'BaseReplayEngine',
    'ReplayEngine',
    'load_game_json',
    'parse_game_data',
    'ReplayDataLLM',
]

# --------------------------------
# Expose the CLI parser from ``scripts/replay.py`` so external modules can do
# ``from replay import parse_arguments`` without caring about the script path.
# --------------------------------

try:
    from scripts.replay import parse_arguments  # type: ignore

    __all__.append("parse_arguments")
except ModuleNotFoundError:  # pragma: no cover – Task-0 runs via scripts/*
    # In environments where ``scripts`` is not on sys.path (unlikely because
    # Task-0 entry points chdir to repo root) we silently skip the re-export.
    pass

 