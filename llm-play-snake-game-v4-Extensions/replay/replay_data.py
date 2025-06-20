from __future__ import annotations

"""Dataclasses used by the *replay* package.

Separated into its own module so that both GUI and non-GUI components can
import the lightweight containers without pulling in Pygame or file-I/O code.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["BaseReplayData", "ReplayData"]


# Base record: strict minimum for pure visual replay 
# This class is NOT Task0 specific. It's generic.

@dataclass(slots=True)
class BaseReplayData:
    """Minimal subset of *game_N.json* required for vanilla playback."""

    apple_positions: List[List[int]]
    moves: List[str]
    game_end_reason: Optional[str]


# Extended record: LLM-specific extras (plans, metadata, raw JSON)
# This class is Task0 specific.

@dataclass(slots=True)
class ReplayData(BaseReplayData):
    """Extended replay data used by the Task-0 LLM GUI overlay."""

    planned_moves: List[str]
    primary_llm: str
    secondary_llm: str
    timestamp: Optional[str]
    llm_response: Optional[str]
    full_json: Dict[str, Any] 
    