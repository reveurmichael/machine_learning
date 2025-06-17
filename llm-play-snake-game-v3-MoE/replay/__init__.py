"""
Replay package initialization.
This file exposes the replay engine and utility functions for use by the application.
"""

from replay.replay_engine import ReplayEngine
from replay.replay_utils import load_game_json, parse_game_data

__all__ = [
    'ReplayEngine',
    'load_game_json',
    'parse_game_data',
]

# ------------------------------------------------------------------
# Make the CLI parser from the *script* version (top-level replay.py)
# available when users import the *package* ``replay``.
#
# This avoids duplicating the full argument-definition block while
# eliminating the ImportError hit by ``replay_web.py``.
# ------------------------------------------------------------------

import importlib.util as _importlib_util
import pathlib as _pathlib

# Compute absolute path to the *top-level* replay.py script (one directory up
# from the ``replay`` package directory).
_script_path = (_pathlib.Path(__file__).resolve().parent.parent) / 'replay.py'

_spec = _importlib_util.spec_from_file_location('replay_script', str(_script_path))
if _spec and _spec.loader:  # pragma: no cover – defensive
    _replay_script = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_replay_script)
    parse_arguments = _replay_script.parse_arguments  # type: ignore[attr-defined]
    # Make it part of the public API
    __all__.append('parse_arguments')

 