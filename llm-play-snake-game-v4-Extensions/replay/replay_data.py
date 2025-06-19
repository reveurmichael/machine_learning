from __future__ import annotations

"""Dataclasses used by the *replay* package.

Separated into its own module so that both GUI and non-GUI components can
import the lightweight containers without pulling in Pygame or file-I/O code.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["ReplayData", "ReplayDataLLM"]


# --- Base record: strict minimum for pure visual replay ---------------------

@dataclass(slots=True)
class ReplayData:
    """Minimal subset of *game_N.json* required for vanilla playback."""

    apple_positions: List[List[int]]
    moves: List[str]
    game_end_reason: Optional[str]


# --- Extended record: LLM-specific extras (plans, metadata, raw JSON) -------


@dataclass(slots=True)
class ReplayDataLLM(ReplayData):
    """Extended replay data used by the Task-0 LLM GUI overlay."""

    planned_moves: List[str]
    primary_llm: str
    secondary_llm: str
    timestamp: Optional[str]
    llm_response: Optional[str]
    full_json: Dict[str, Any] 