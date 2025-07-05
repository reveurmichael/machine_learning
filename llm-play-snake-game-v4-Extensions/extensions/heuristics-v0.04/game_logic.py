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
        
        Args:
            recorded_game_state: The game state that was recorded for this round
            
        Returns:
            Next move direction or "NO_PATH_FOUND"
        """
        if not self.agent:
            return "NO_PATH_FOUND"
        
        # Temporarily set the game state to the recorded state for explanation generation
        original_snapshot = self.get_state_snapshot()
        
        # Create a temporary game state snapshot using the recorded state
        temp_snapshot = self.get_recorded_state_snapshot(recorded_game_state)
        
        # Get move from agent using the recorded game state
        if hasattr(self.agent, 'get_move_with_explanation'):
            # Create a temporary game logic instance with the recorded state
            temp_game = self._create_temp_game_logic(temp_snapshot)
            move, explanation = self.agent.get_move_with_explanation(temp_game)
            self._store_explanation(explanation)
        else:
            # Standard move generation for agents without explanation support
            move = self.agent.get_move(self)
            self._store_explanation(f"Move {move} chosen by {self.algorithm_name} algorithm.")
        
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
    
    def _create_temp_game_logic(self, game_state_snapshot: dict):
        """
        Create a temporary game logic instance with the given game state snapshot.
        
        This allows the agent to generate explanations using the recorded game state
        without affecting the actual game state.
        
        Args:
            game_state_snapshot: The game state snapshot to use
            
        Returns:
            A temporary game logic instance with the given state
        """
        # Create a temporary game logic instance
        temp_game = HeuristicGameLogic(grid_size=game_state_snapshot.get('grid_size', 10), use_gui=False)
        
        # Set the agent
        if self.agent:
            temp_game.set_agent(self.agent)
        
        # Override the get_state_snapshot method to return our recorded state
        def get_recorded_snapshot():
            return game_state_snapshot
        
        temp_game.get_state_snapshot = get_recorded_snapshot
        
        return temp_game
    
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
        # SSOT Fix: Use snake_positions[0] as the head position to match coordinate system
        # According to coordinate-system.md: snake_positions[0] is the HEAD, snake_positions[-1] is the TAIL
        head_pos = self.snake_positions[0].tolist() if len(self.snake_positions) > 0 else [0, 0]
        
        return {
            "head_position": head_pos,
            "snake_positions": self.snake_positions.tolist(),
            "apple_position": self.apple_position.tolist(),
            "grid_size": self.grid_size,
            "score": self.game_state.score,
            "steps": self.game_state.steps,
            "current_direction": self.current_direction,
            "snake_length": len(self.snake_positions)
        }
    
    def get_recorded_state_snapshot(self, recorded_state: dict) -> dict:
        """
        Get game state snapshot from recorded state for dataset consistency.
        
        This ensures agents use the same state that gets recorded in the dataset,
        preventing coordinate mismatches between explanations and recorded data.
        
        Args:
            recorded_state: Recorded game state from dataset_game_states
            
        Returns:
            Dictionary containing recorded game state
        """
        return {
            "head_position": recorded_state.get("head_position", [0, 0]),
            "snake_positions": recorded_state.get("snake_positions", []),
            "apple_position": recorded_state.get("apple_position", [0, 0]),
            "grid_size": recorded_state.get("grid_size", self.grid_size),
            "score": recorded_state.get("score", 0),
            "steps": recorded_state.get("steps", 0),
            "current_direction": recorded_state.get("current_direction", "UP"),
            "snake_length": len(recorded_state.get("snake_positions", []))
        } 
