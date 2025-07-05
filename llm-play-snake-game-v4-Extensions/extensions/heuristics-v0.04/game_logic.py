"""
Heuristic Game Logic - Core game mechanics for heuristic algorithms v0.04
----------------

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
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from core.game_logic import BaseGameLogic
from config.ui_constants import GRID_SIZE

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
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Get move from heuristic agent with explanation support
        move = self._get_agent_move()
        
        # Record search performance
        search_time = time.time() - start_time
        self._record_pathfinding_attempt(move, search_time)
        
        # Generate planned moves
        planned_moves = [move] if move and move != "NO_PATH_FOUND" else ["NO_PATH_FOUND"]
        
        # Record planned moves in round manager for proper rounds_data population
        if hasattr(self.game_state, 'round_manager') and self.game_state.round_manager:
            self.game_state.round_manager.record_planned_moves(planned_moves)
            # Sync round data immediately to ensure planned moves are recorded
            self.game_state.round_manager.sync_round_data()
        
        # Return planned moves
        return planned_moves
    
    def _get_agent_move(self) -> str:
        """Get move from agent with explanation support."""
        if hasattr(self.agent, 'get_move_with_explanation'):
            move, explanation = self.agent.get_move_with_explanation(self)
            self._store_explanation(explanation)
            return move
        else:
            # Standard move generation for agents without explanation support
            move = self.agent.get_move(self)
            self._store_explanation(f"Move {move} chosen by {self.algorithm_name} algorithm.")
            return move
    
    def _store_explanation(self, explanation) -> None:
        """Store move explanation and metrics for dataset generation."""
        if not isinstance(self.game_state, HeuristicGameData):
            return
            
        self.game_state.record_move_explanation(explanation)
        
        # Extract metrics if explanation is a dictionary
        metrics = explanation.get("metrics", {}) if isinstance(explanation, dict) else {}
        self.game_state.record_move_metrics(metrics)
            
    def _record_pathfinding_attempt(self, move: str, search_time: float, error: str = None) -> None:
        """Record pathfinding attempt for statistics."""
        if not isinstance(self.game_state, HeuristicGameData):
            return
            
        success = move not in [None, "NO_PATH_FOUND"]
        path_length = 1 if success else 0
                
        self.game_state.record_pathfinding_attempt(
            success=success,
            path_length=path_length,
            search_time=search_time,
            nodes_explored=1  # Simplified - could be enhanced with actual node count
        )
            
        if error:
            self.game_state.last_move_explanation = f"Error in {self.algorithm_name}: {error}"
    
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
            move = self.planned_moves.pop(0)
            
            # Note: The move will be recorded in the round buffer by the base make_move() method
            # which calls game_state.record_move(), which in turn calls round_buffer.add_move()
            # No need to duplicate this here to avoid double-recording
            
            return move
        else:
            return "NO_PATH_FOUND"
    
    def get_next_planned_move_with_state(self, recorded_game_state: dict) -> str:
        """
        Get the next planned move using a recorded game state for SSOT compliance.
        This method ensures that the agent generates explanations using the same
        game state that is recorded for dataset generation, eliminating coordinate mismatches.
        
        PRE-EXECUTION: The recorded_game_state contains the game state BEFORE the move is executed.
        This includes: head_position, apple_position, snake_positions, score, steps.
        The agent will make decisions based on this pre-move state.
        
        Args:
            recorded_game_state: The game state that was recorded for this round (PRE-MOVE state)
        Returns:
            Next move direction or "NO_PATH_FOUND"
        """
        if not self.agent:
            return "NO_PATH_FOUND"

        # Use the provided state dict directly for the agent
        # PRE-EXECUTION: Agent receives pre-move state and must make decision based on current positions
        # The agent will calculate: valid_moves, path_to_apple, manhattan_distance, etc. from pre-move state
        move, explanation = self.agent.get_move_with_explanation(recorded_game_state)
        self._store_explanation(explanation)

        # Generate planned moves
        planned_moves = [move] if move and move != "NO_PATH_FOUND" else ["NO_PATH_FOUND"]

        # Record planned moves in round manager
        if hasattr(self.game_state, 'round_manager') and self.game_state.round_manager:
            self.game_state.round_manager.record_planned_moves(planned_moves)
            # Note: The actual move will be recorded by the base make_move() method
            # No need to duplicate this here to avoid double-recording

        # Update planned_moves
        self.planned_moves = planned_moves

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
        
        PRE-EXECUTION: This method returns the current game state BEFORE any move is executed.
        This includes: head_position, apple_position, snake_positions, score, steps.
        All values are from the current state and will be used for agent decision making.
        
        Returns:
            Dictionary containing current game state (PRE-MOVE state)
        """
        # SSOT Fix: Use the actual head_position from the game logic
        # PRE-EXECUTION: head_position is the current head position before any move
        # The game logic sets head_position = snake_positions[-1] (last element)
        head_pos = self.head_position.tolist() if hasattr(self, 'head_position') else [0, 0]
        
        return {
            "head_position": head_pos,  # PRE-MOVE: current head position
            "snake_positions": self.snake_positions.tolist(),  # PRE-MOVE: current snake body positions
            "apple_position": self.apple_position.tolist(),  # PRE-MOVE: current apple position
            "grid_size": self.grid_size,
            "score": self.game_state.score,  # PRE-MOVE: current score
            "steps": self.game_state.steps,  # PRE-MOVE: current step count
            "current_direction": self.current_direction,  # PRE-MOVE: current direction
            "snake_length": len(self.snake_positions)  # PRE-MOVE: current snake length
        }
    
    def get_recorded_state_snapshot(self, recorded_state: dict) -> dict:
        """
        Get game state snapshot from recorded state for dataset consistency.
        
        This ensures agents use the same state that gets recorded in the dataset,
        preventing coordinate mismatches between explanations and recorded data.
        
        PRE-EXECUTION: The recorded_state contains the game state BEFORE the move is executed.
        This method returns a clean snapshot of that pre-move state for agent use.
        
        Args:
            recorded_state: Recorded game state from dataset_game_states (PRE-MOVE state)
            
        Returns:
            Dictionary containing recorded game state (PRE-MOVE state)
        """
        return {
            "head_position": recorded_state.get("head_position", [0, 0]),  # PRE-MOVE: recorded head position
            "snake_positions": recorded_state.get("snake_positions", []),  # PRE-MOVE: recorded snake positions
            "apple_position": recorded_state.get("apple_position", [0, 0]),  # PRE-MOVE: recorded apple position
            "grid_size": recorded_state.get("grid_size", self.grid_size),
            "score": recorded_state.get("score", 0),  # PRE-MOVE: recorded score
            "steps": recorded_state.get("steps", 0),  # PRE-MOVE: recorded step count
            "current_direction": recorded_state.get("current_direction", "UP"),  # PRE-MOVE: recorded direction
            "snake_length": len(recorded_state.get("snake_positions", []))  # PRE-MOVE: recorded snake length
        }

    def _generate_apple(self) -> NDArray[np.int_]:
        """Generate a new apple position not occupied by the snake."""
        apple = super()._generate_apple()
        return apple 

    def make_move(self, direction_key: str) -> Tuple[bool, bool]:
        """
        Make a move in the given direction and update the game state.
        
        POST-EXECUTION: This method actually executes the move and updates the game state.
        After this call, the following values will be updated:
        - head_position: new head position after the move
        - snake_positions: updated snake body positions after the move
        - score: updated score (increased if apple was eaten)
        - steps: incremented step count
        - apple_position: new apple position if the previous one was eaten
        
        Args:
            direction_key: The direction to move (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Tuple of (apple_eaten, game_over) status after the move
        """
        result = super().make_move(direction_key)
        return result 
