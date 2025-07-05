"""
Heuristic Game Rounds - Round management for heuristic algorithms
----------------

This module provides heuristic-specific round management functionality
while maintaining compatibility with the base round management system.

Design Philosophy:
- Extends BaseRoundManager (inherits all generic round functionality)
- Adds heuristic-specific methods like game_state recording
- Maintains same interface as base RoundManager for compatibility
- Keeps heuristics extension self-contained and standalone
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import Dict, Any, Optional
from core.game_rounds import RoundManager


class HeuristicRoundManager(RoundManager):
    """
    Heuristics-specific round manager that extends the base RoundManager.
    
    Adds heuristic-specific functionality like game_state recording
    while maintaining compatibility with the base round management.
    
    Design Patterns:
    - Template Method: Inherits base round management structure
    - Strategy Pattern: Different heuristic algorithms share same round management
    """
    
    def record_game_state(self, game_state: dict) -> None:
        """
        Store a snapshot of the game state in the current round's data.
        
        This method is called by the game manager to record the current
        game state before applying moves, ensuring every round has a
        valid game_state for dataset generation.
        
        Args:
            game_state: Dictionary containing the current game state
                       (snake position, apple position, score, etc.)
        """
        current_round_dict = self._get_or_create_round_data(self.round_buffer.number)
        current_round_dict['game_state'] = dict(game_state)  # Store a copy to avoid mutation

    def sync_round_data(self) -> None:
        """
        Synchronize the in-progress round buffer with the persistent mapping.
        
        Override the base method to ensure one move per round for heuristics.
        Instead of extending the moves list, we replace it to maintain
        the one-move-per-round principle.
        """
        if not self.round_buffer:
            return

        current_round_dict = self._get_or_create_round_data(self.round_buffer.number)
        current_round_dict.update({
            "round": self.round_buffer.number,
            "apple_position": self.round_buffer.apple_position,
        })
        current_round_dict["planned_moves"] = list(self.round_buffer.planned_moves)
        
        # For heuristics: replace moves instead of extending (one move per round)
        if self.round_buffer.moves:
            current_round_dict["moves"] = list(self.round_buffer.moves)
            self.round_buffer.moves.clear()
        
        # Fail fast if round 1 is missing after sync
        if 1 not in self.rounds_data and '1' not in self.rounds_data:
            raise RuntimeError("[SSOT] Round 1 missing in rounds_data after sync_round_data!") 