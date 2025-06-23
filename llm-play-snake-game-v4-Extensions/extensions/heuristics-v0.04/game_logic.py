"""
Heuristic Game Logic - Game logic for heuristic algorithms
========================================================

This module extends BaseGameLogic to provide heuristic-specific game logic
while maintaining compatibility with the base game engine.

Design Philosophy:
- Extends BaseGameLogic (inherits all core game mechanics)
- Uses HeuristicGameData for data tracking
- Integrates heuristic agents (BFS, DFS, etc.)
- No GUI dependencies (headless by default)
"""

from __future__ import annotations

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

import time
from typing import List, Optional, TYPE_CHECKING

from config.ui_constants import GRID_SIZE
from core.game_logic import BaseGameLogic
from game_data import HeuristicGameData

if TYPE_CHECKING:
    from bfs_agent import BFSAgent


class HeuristicGameLogic(BaseGameLogic):
    """
    Game logic for heuristic algorithms.
    
    Extends BaseGameLogic with heuristic-specific functionality while
    maintaining the same core game mechanics and interface.
    
    Design Patterns:
    - Template Method: Inherits base game logic structure
    - Strategy Pattern: Pluggable heuristic algorithms
    - Factory Pattern: Uses HeuristicGameData for data container
    """
    
    # Use heuristic-specific data container
    GAME_DATA_CLS = HeuristicGameData
    
    # Type annotations to help pylint understand inheritance
    game_state: HeuristicGameData
    planned_moves: List[str]
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = False) -> None:
        """
        Initialize heuristic game logic.
        
        Args:
            grid_size: Size of the game grid
            use_gui: Whether to use GUI (default False for heuristics)
        """
        # Heuristics are headless by default
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        
        # Heuristic-specific initialization
        self.agent: Optional[BFSAgent] = None
        self.algorithm_name: str = "v0.02-BFS-Safe-Greedy"
        
        # Ensure we have the correct data type
        # Note: game_state is initialized in super().__init__(), so we can safely access it here
        if not isinstance(self.game_state, HeuristicGameData):
            self.game_state = HeuristicGameData()
            self.game_state.reset()
    
    def set_agent(self, agent: BFSAgent) -> None:
        """
        Set the heuristic agent for pathfinding.
        
        Args:
            agent: Heuristic agent instance (BFS, DFS, etc.)
        """
        self.agent = agent
        self.algorithm_name = getattr(agent, 'algorithm_name', 'Unknown')
        
        # Update game data with algorithm info
        if isinstance(self.game_state, HeuristicGameData):
            self.game_state.algorithm_name = self.algorithm_name
    
    def plan_next_moves(self) -> List[str]:
        """
        Plan next moves using the heuristic agent.
        
        This method implements the planning logic for heuristic algorithms,
        replacing the LLM-specific planning in Task-0.
        
        Returns:
            List of planned moves (typically single move for heuristics)
        """
        if not self.agent:
            return ["NO_PATH_FOUND"]
        
        try:
            # Record start time for performance tracking
            start_time = time.time()
            
            # Get move from heuristic agent (try v0.04 method first, fallback to v0.03)
            if hasattr(self.agent, 'get_move_with_explanation'):
                move, explanation = self.agent.get_move_with_explanation(self)
                # Store explanation for JSONL dataset generation
                if isinstance(self.game_state, HeuristicGameData):
                    self.game_state.record_move_explanation(explanation)
            else:
                # Fallback for agents that don't support explanations yet
                move = self.agent.get_move(self)
                if isinstance(self.game_state, HeuristicGameData):
                    fallback_explanation = f"Move {move} chosen by {self.algorithm_name} algorithm."
                    self.game_state.record_move_explanation(fallback_explanation)
            
            # Record search time
            search_time = time.time() - start_time
            
            # Track pathfinding attempt
            if isinstance(self.game_state, HeuristicGameData):
                success = move not in [None, "NO_PATH_FOUND"]
                path_length = 1 if success else 0  # Heuristics typically return single moves
                
                self.game_state.record_pathfinding_attempt(
                    success=success,
                    path_length=path_length,
                    search_time=search_time,
                    nodes_explored=1  # Simplified - could be enhanced with actual node count
                )
            
            # Return planned moves
            if move is None or move == "NO_PATH_FOUND":
                return ["NO_PATH_FOUND"]
            else:
                return [move]
                
        except Exception as e:
            print(f"Heuristic planning error: {e}")
            
            # Track failed attempt
            if isinstance(self.game_state, HeuristicGameData):
                self.game_state.record_pathfinding_attempt(success=False)
                self.game_state.last_move_explanation = f"Error in {self.algorithm_name}: {str(e)}"
                
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
        # Note: planned_moves is initialized in super().__init__(), so we can safely access it here
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
            "algorithm_name": self.algorithm_name,
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