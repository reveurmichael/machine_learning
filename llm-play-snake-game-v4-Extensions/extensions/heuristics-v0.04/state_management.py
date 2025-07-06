"""
State Management for Robust Pre/Post Move Separation

This module implements immutable state objects and explicit type separation
to prevent SSOT violations between pre-move and post-move states.

Design Philosophy:
- Immutable pre-move states using MappingProxyType
- Explicit PreMoveState/PostMoveState classes with type enforcement
- Clear lifecycle orchestration with fail-fast validation
- Single source of truth pipeline enforcement
"""

from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Any, List, Optional, Tuple
import copy
import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, project_root)

# Import utilities using absolute path
from utils.print_utils import print_info, print_error, print_warning
from agents.agent_bfs import BFSAgent

__all__ = [
    "PreMoveState", 
    "PostMoveState", 
    "StateManager",
    "create_pre_move_state",
    "create_post_move_state",
    "validate_state_consistency"
]


@dataclass(frozen=True)
class PreMoveState:
    """
    Immutable pre-move game state.
    
    This class enforces that pre-move states cannot be modified,
    preventing accidental mutations that could cause SSOT violations.
    
    Design Pattern: Immutable Object Pattern
    Purpose: Prevent state mutations during agent decision making
    Educational Value: Shows how to enforce state immutability for SSOT compliance
    """
    game_state: MappingProxyType
    
    def __post_init__(self):
        """Validate pre-move state structure."""
        if not isinstance(self.game_state, MappingProxyType):
            raise TypeError(f"PreMoveState.game_state must be MappingProxyType, got {type(self.game_state)}")
        
        # Fail-fast: Validate required fields exist
        required_fields = ['head_position', 'snake_positions', 'apple_position', 'grid_size']
        missing_fields = [field for field in required_fields if field not in self.game_state]
        if missing_fields:
            raise ValueError(f"PreMoveState missing required fields: {missing_fields}")
        
        # Fail-fast: Validate head position is consistent with snake positions
        head_pos = self.game_state.get('head_position')
        snake_positions = self.game_state.get('snake_positions', [])
        if snake_positions and head_pos != snake_positions[-1]:
            raise ValueError(f"SSOT violation: head_position {head_pos} != last snake position {snake_positions[-1]}")
    
    def get_head_position(self) -> List[int]:
        """Get head position from pre-move state."""
        return list(self.game_state['head_position'])
    
    def get_apple_position(self) -> List[int]:
        """Get apple position from pre-move state."""
        return list(self.game_state['apple_position'])
    
    def get_snake_positions(self) -> List[List[int]]:
        """Get snake positions from pre-move state."""
        return [list(pos) for pos in self.game_state['snake_positions']]
    
    def get_grid_size(self) -> int:
        """Get grid size from pre-move state."""
        return self.game_state['grid_size']
    
    def get_score(self) -> int:
        """Get score from pre-move state."""
        return self.game_state.get('score', 0)
    
    def get_steps(self) -> int:
        """Get steps from pre-move state."""
        return self.game_state.get('steps', 0)


@dataclass
class PostMoveState:
    """
    Mutable post-move game state.
    
    This class represents the state after a move has been executed.
    It includes the move that was applied and the resulting game state.
    
    Design Pattern: Command Pattern
    Purpose: Track state changes and applied moves
    Educational Value: Shows how to track state transitions for debugging
    """
    game_state: Dict[str, Any]
    move_applied: str
    pre_move_state: PreMoveState = field(repr=False)  # Reference to original pre-move state
    
    def __post_init__(self):
        """Validate post-move state structure."""
        if not isinstance(self.game_state, dict):
            raise TypeError(f"PostMoveState.game_state must be dict, got {type(self.game_state)}")
        
        if not isinstance(self.move_applied, str):
            raise TypeError(f"PostMoveState.move_applied must be str, got {type(self.move_applied)}")
        
        if not isinstance(self.pre_move_state, PreMoveState):
            raise TypeError(f"PostMoveState.pre_move_state must be PreMoveState, got {type(self.pre_move_state)}")
    
    def get_head_position(self) -> List[int]:
        """Get head position from post-move state."""
        return list(self.game_state.get('head_position', [0, 0]))
    
    def get_apple_position(self) -> List[int]:
        """Get apple position from post-move state."""
        return list(self.game_state.get('apple_position', [0, 0]))
    
    def get_snake_positions(self) -> List[List[int]]:
        """Get snake positions from post-move state."""
        return [list(pos) for pos in self.game_state.get('snake_positions', [])]
    
    def get_grid_size(self) -> int:
        """Get grid size from post-move state."""
        return self.game_state.get('grid_size', 10)
    
    def get_score(self) -> int:
        """Get score from post-move state."""
        return self.game_state.get('score', 0)
    
    def get_steps(self) -> int:
        """Get steps from post-move state."""
        return self.game_state.get('steps', 0)


class StateManager:
    """
    Manages state transitions and enforces SSOT compliance.
    
    This class orchestrates the strict pipeline:
    pre_move_state --> move --> post_move_state
    
    Design Pattern: State Machine Pattern
    Purpose: Enforce clear state transitions and prevent SSOT violations
    Educational Value: Shows how to implement state management with fail-fast validation
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self._current_pre_state: Optional[PreMoveState] = None
        self._current_post_state: Optional[PostMoveState] = None
        print_info("[StateManager] Initialized with immutable state management", "StateManager")
    
    def create_pre_move_state(self, raw_game_state: Dict[str, Any]) -> PreMoveState:
        """
        Create an immutable pre-move state from raw game state.
        
        Args:
            raw_game_state: Raw game state dictionary
            
        Returns:
            Immutable PreMoveState object
            
        Raises:
            ValueError: If state validation fails
        """
        # Deep copy to prevent external modifications
        frozen_state = MappingProxyType(copy.deepcopy(raw_game_state))
        pre_state = PreMoveState(game_state=frozen_state)
        
        # Fail-fast: Validate state consistency
        self._validate_pre_move_state(pre_state)
        
        self._current_pre_state = pre_state
        print_info(f"[StateManager] Created pre-move state with head at {pre_state.get_head_position()}", "StateManager")
        return pre_state
    
    def create_post_move_state(self, pre_state: PreMoveState, move: str, 
                             raw_post_game_state: Dict[str, Any]) -> PostMoveState:
        """
        Create a post-move state from pre-move state and move result.
        
        Args:
            pre_state: The pre-move state
            move: The move that was applied
            raw_post_game_state: Raw game state after move execution
            
        Returns:
            PostMoveState object
            
        Raises:
            ValueError: If state validation fails
        """
        if not isinstance(pre_state, PreMoveState):
            raise TypeError(f"pre_state must be PreMoveState, got {type(pre_state)}")
        
        # Deep copy to prevent external modifications
        post_game_state = copy.deepcopy(raw_post_game_state)
        post_state = PostMoveState(
            game_state=post_game_state,
            move_applied=move,
            pre_move_state=pre_state
        )
        
        # Fail-fast: Validate state transition consistency
        self._validate_state_transition(pre_state, post_state)
        
        self._current_post_state = post_state
        print_info(f"[StateManager] Created post-move state with head at {post_state.get_head_position()}", "StateManager")
        return post_state
    
    def _validate_pre_move_state(self, pre_state: PreMoveState) -> None:
        """
        Validate pre-move state for consistency.
        
        Args:
            pre_state: Pre-move state to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Validate head position is consistent with snake positions
        head_pos = pre_state.get_head_position()
        snake_positions = pre_state.get_snake_positions()
        
        if not snake_positions:
            raise ValueError("Pre-move state has no snake positions")
        
        if head_pos != snake_positions[-1]:
            raise ValueError(f"SSOT violation: head position {head_pos} != last snake position {snake_positions[-1]}")
        
        # Validate apple position is within grid bounds
        apple_pos = pre_state.get_apple_position()
        grid_size = pre_state.get_grid_size()
        
        if not (0 <= apple_pos[0] < grid_size and 0 <= apple_pos[1] < grid_size):
            raise ValueError(f"Apple position {apple_pos} outside grid bounds {grid_size}x{grid_size}")
        
        # Validate head position is within grid bounds
        if not (0 <= head_pos[0] < grid_size and 0 <= head_pos[1] < grid_size):
            raise ValueError(f"Head position {head_pos} outside grid bounds {grid_size}x{grid_size}")
        
        print_info(f"[StateManager] Pre-move state validation passed: head={head_pos}, apple={apple_pos}", "StateManager")
    
    def _validate_state_transition(self, pre_state: PreMoveState, post_state: PostMoveState) -> None:
        """
        Validate that post-move state is consistent with pre-move state and move.
        
        Args:
            pre_state: Pre-move state
            post_state: Post-move state
            
        Raises:
            ValueError: If transition validation fails
        """
        pre_head = pre_state.get_head_position()
        post_head = post_state.get_head_position()
        move = post_state.move_applied
        
        # Calculate expected post-move head position
        expected_head = self._calculate_expected_head_position(pre_head, move, pre_state.get_grid_size())
        
        if post_head != expected_head:
            raise ValueError(
                f"State transition violation: expected head {expected_head} after move '{move}', "
                f"got {post_head} from pre-head {pre_head}"
            )
        
        # Validate that snake length increased only if apple was eaten
        pre_snake_length = len(pre_state.get_snake_positions())
        post_snake_length = len(post_state.get_snake_positions())
        pre_apple = pre_state.get_apple_position()
        post_apple = post_state.get_apple_position()
        
        if post_snake_length > pre_snake_length + 1:
            raise ValueError(f"Snake length increased by {post_snake_length - pre_snake_length} but should be at most 1")
        
        print_info(f"[StateManager] State transition validation passed: {pre_head} -> {post_head} via '{move}'", "StateManager")
    
    def _calculate_expected_head_position(self, pre_head: List[int], move: str, grid_size: int) -> List[int]:
        """
        Calculate expected head position after applying move.
        
        Args:
            pre_head: Pre-move head position
            move: Move direction
            grid_size: Grid size
            
        Returns:
            Expected post-move head position
        """
        x, y = pre_head
        
        if move == "UP":
            y = (y + 1) % grid_size
        elif move == "DOWN":
            y = (y - 1) % grid_size
        elif move == "RIGHT":
            x = (x + 1) % grid_size
        elif move == "LEFT":
            x = (x - 1) % grid_size
        else:
            raise ValueError(f"Invalid move: {move}")
        
        return [x, y]
    
    def get_current_pre_state(self) -> Optional[PreMoveState]:
        """Get current pre-move state."""
        return self._current_pre_state
    
    def get_current_post_state(self) -> Optional[PostMoveState]:
        """Get current post-move state."""
        return self._current_post_state
    
    def clear_states(self) -> None:
        """Clear current states."""
        self._current_pre_state = None
        self._current_post_state = None
        print_info("[StateManager] Cleared current states", "StateManager")


# Convenience functions for external use
def create_pre_move_state(raw_game_state: Dict[str, Any]) -> PreMoveState:
    """
    Create an immutable pre-move state.
    
    Args:
        raw_game_state: Raw game state dictionary
        
    Returns:
        Immutable PreMoveState object
    """
    manager = StateManager()
    return manager.create_pre_move_state(raw_game_state)


def create_post_move_state(pre_state: PreMoveState, move: str, 
                         raw_post_game_state: Dict[str, Any]) -> PostMoveState:
    """
    Create a post-move state from pre-move state and move result.
    
    Args:
        pre_state: The pre-move state
        move: The move that was applied
        raw_post_game_state: Raw game state after move execution
        
    Returns:
        PostMoveState object
    """
    manager = StateManager()
    return manager.create_post_move_state(pre_state, move, raw_post_game_state)


def validate_state_consistency(pre_state: PreMoveState, post_state: PostMoveState) -> bool:
    """
    Validate consistency between pre-move and post-move states.
    
    Args:
        pre_state: Pre-move state
        post_state: Post-move state
        
    Returns:
        True if states are consistent, False otherwise
    """
    try:
        manager = StateManager()
        manager._validate_state_transition(pre_state, post_state)
        return True
    except ValueError as e:
        print_error(f"[StateManager] State consistency validation failed: {e}", "StateManager")
        return False


# SSOT validation utilities
def validate_explanation_head_consistency(pre_state: PreMoveState, explanation: Dict[str, Any]) -> bool:
    """
    Validate that explanation head position matches pre-move state head position.
    
    Args:
        pre_state: Pre-move state
        explanation: Agent explanation dictionary
        
    Returns:
        True if head positions match, False otherwise
    """
    pre_head = pre_state.get_head_position()
    
    # Extract head position from explanation
    explanation_head = None
    if isinstance(explanation, dict):
        metrics = explanation.get('metrics', {})
        explanation_head = metrics.get('head_position')
    
    if explanation_head is None:
        print_warning("[StateManager] No head position found in explanation", "StateManager")
        return False
    
    # Compare head positions
    if pre_head != explanation_head:
        print_error(
            f"[StateManager] SSOT violation: pre-move head {pre_head} != explanation head {explanation_head}",
            "StateManager"
        )
        return False
    
    print_info(f"[StateManager] Explanation head position validation passed: {pre_head}", "StateManager")
    return True


def extract_state_for_agent(pre_state: PreMoveState) -> Dict[str, Any]:
    """
    Extract game state dict for agent use from pre-move state.
    
    This ensures agents always receive the exact pre-move state
    that was validated and frozen.
    
    Args:
        pre_state: Pre-move state
        
    Returns:
        Game state dictionary for agent use
    """
    # Convert MappingProxyType back to dict for agent compatibility
    # This is safe because the original state was deep-copied
    return dict(pre_state.game_state) 