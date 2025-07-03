"""
Heuristic Game Logic - Core game mechanics for heuristic algorithms v0.04
--------------------------------------------------------------------------

This module implements the game logic specifically designed for heuristic
algorithms, extending the base game logic with features needed for
pathfinding algorithms.

v0.04 Enhancement: Supports explanation generation for LLM fine-tuning
datasets while maintaining compatibility with existing heuristic agents.

Design Patterns:
- Inheritance: Extends BaseGameLogic from core framework
- Strategy Pattern: Different heuristic algorithms can be plugged in
- Observer Pattern: Game state changes notify interested components
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
from typing import TYPE_CHECKING, List, Optional

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from core.game_logic import BaseGameLogic
from config.ui_constants import GRID_SIZE
from utils.print_utils import print_error

if TYPE_CHECKING:
    pass

# Import extension-specific components using relative imports
from game_data import HeuristicGameData

if TYPE_CHECKING:
    from agents.agent_bfs import BFSAgent


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
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        """
        Initialize heuristic game logic with pathfinding capabilities.
        
        Args:
            grid_size: Size of the game grid (default from config)
            use_gui: Whether to use GUI (default True, can be disabled for headless)
        """
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        
        # Heuristic-specific attributes
        self.agent: Optional[BFSAgent] = None
        # Default algorithm name before an agent is set
        self.algorithm_name: str = "BFS-Safe-Greedy"
        
        # Ensure we have the correct data type and grid_size is set
        # Note: game_state is initialized in super().__init__(), so we can safely access it here
        if not isinstance(self.game_state, HeuristicGameData):
            self.game_state = HeuristicGameData()
            self.game_state.reset()
        
        # Ensure game_state has grid_size for JSON output
        if isinstance(self.game_state, HeuristicGameData):
            self.game_state.grid_size = grid_size
    
    def set_agent(self, agent: BFSAgent) -> None:
        """
        Set the heuristic agent for pathfinding.
        
        Args:
            agent: Heuristic agent instance (BFS, DFS, etc.)
        """
        self.agent = agent
        self.algorithm_name = getattr(agent, 'algorithm_name', 'Unknown')
        
        # Update game data with algorithm info and grid_size
        if isinstance(self.game_state, HeuristicGameData):
            self.game_state.algorithm_name = self.algorithm_name
            self.game_state.grid_size = self.grid_size  # Set actual grid size
    
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

                    # If the explanation is a dictionary with a metrics key,
                    # extract it so that dataset generation can access a
                    # *flat* metrics list without reparsing the explanation.
                    metrics_payload = {}
                    if isinstance(explanation, dict):
                        # New schema – metrics stored separately inside dict
                        metrics_payload = explanation.get("metrics", {})
                    self.game_state.record_move_metrics(metrics_payload)
            else:
                # Fallback for agents that don't support explanations yet
                move = self.agent.get_move(self)
                if isinstance(self.game_state, HeuristicGameData):
                    fallback_explanation = f"Move {move} chosen by {self.algorithm_name} algorithm."
                    self.game_state.record_move_explanation(fallback_explanation)
                    self.game_state.record_move_metrics({})
            
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
            print_error(f"Heuristic planning error: {e}")
            
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
