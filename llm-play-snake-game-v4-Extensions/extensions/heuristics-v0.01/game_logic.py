"""
Heuristic Game Logic - Simple BFS game logic
==========================================

Minimal extension of BaseGameLogic for BFS pathfinding.
"""

from __future__ import annotations
import time
from typing import List, Optional, TYPE_CHECKING

from config.ui_constants import GRID_SIZE
from core.game_logic import BaseGameLogic

if TYPE_CHECKING:
    from agent_bfs import BFSAgent


class HeuristicGameLogic(BaseGameLogic):
    """
    Simple game logic for BFS algorithm.
    Uses BaseGameData directly - no custom data tracking needed.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = False) -> None:
        """Initialize with BFS agent support."""
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        self.agent: Optional[BFSAgent] = None
    
    def set_agent(self, agent: BFSAgent) -> None:
        """Set the BFS agent."""
        self.agent = agent
    
    def plan_next_moves(self) -> List[str]:
        """Get next move from BFS agent."""
        if not self.agent:
            return ["NO_PATH_FOUND"]
        
        try:
            move = self.agent.get_move(self)
            return [move] if move and move != "NO_PATH_FOUND" else ["NO_PATH_FOUND"]
        except Exception:
            return ["NO_PATH_FOUND"]
    
    def get_next_planned_move(self) -> str:
        """
        Get the next planned move, generating new plan if needed.
        
        Overrides the base method to use heuristic planning instead of
        LLM-based planning.
        
        Returns:
            Next move direction or "NO_PATH_FOUND"
        """
        # Check if we need a new plan
        if not self.planned_moves:
            self.planned_moves = self.plan_next_moves()
        
        # Get next move from plan
        if self.planned_moves:
            return self.planned_moves.pop(0)
        else:
            return "NO_PATH_FOUND"
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the current heuristic algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        return {
            "algorithm_name": "BFS",
            "agent_type": type(self.agent).__name__ if self.agent else "None",
            "has_agent": self.agent is not None
        }
    
    def get_state_snapshot(self) -> dict:
        """
        Get current game state snapshot for agent decision making.
        
        Provides a clean interface for heuristic agents to access game state
        without coupling to internal game logic structure.
        
        Returns:
            Dictionary containing current game state
        """
        return {
            "head_position": self.head_position.tolist(),
            "snake_positions": self.snake_positions.tolist(),
            "apple_position": self.apple_position.tolist(),
            "grid_size": self.grid_size,
            "score": self.game_state.score,
            "steps": self.game_state.steps,
            "current_direction": self.current_direction,
            "snake_length": len(self.snake_positions)
        } 