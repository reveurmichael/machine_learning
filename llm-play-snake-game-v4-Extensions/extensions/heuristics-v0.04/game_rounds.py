"""
Heuristic Game Rounds - Round management for heuristic algorithms
----------------

This module provides access to the base round management system.
The base architecture is already perfectly prepared for all tasks.

Design Philosophy:
- Use BaseRoundManager directly (inherits all generic round functionality)
- No extension needed - base class is already perfect for heuristics
- Maintains compatibility with the base round management system
- Keeps heuristics extension self-contained and standalone
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

from core.game_rounds import BaseRoundManager

# Use BaseRoundManager directly - no extension needed
# The base class is already perfectly prepared for all tasks per round.md guidelines
HeuristicRoundManager = BaseRoundManager 