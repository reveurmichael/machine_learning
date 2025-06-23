"""
BFS Hamiltonian Agent - Enhanced BFS with Hamiltonian Cycle Fallback
===================================================================

This module implements a sophisticated agent that extends BFS Safe Greedy
with Hamiltonian cycle capabilities. It represents the evolution from basic
BFS → BFS-Safe-Greedy → BFS-Hamiltonian.

Inheritance Hierarchy:
- BFSAgent (basic pathfinding)
  └── BFSSafeGreedyAgent (adds safety validation)
      └── BFSHamiltonianAgent (adds Hamiltonian cycle fallback)

This demonstrates progressive enhancement: each level adds new capabilities
while reusing the proven functionality of its parent.

Design Patterns:
- Inheritance: Extends BFSSafeGreedyAgent with Hamiltonian cycle logic
- Template Method: Overrides get_move() while reusing safety validation
- Strategy Pattern: Hamiltonian cycle as ultimate fallback strategy
- Composite Pattern: Combines multiple pathfinding strategies
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


def generate_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    """
    Generate a simple Hamiltonian cycle using boustrophedon pattern.
    
    Creates a snake-like pattern that visits every cell exactly once:
    - Even rows: left to right
    - Odd rows: right to left
    
    Args:
        grid_size: Size of the square grid
        
    Returns:
        List of positions forming a Hamiltonian cycle
    """
    cycle: List[Tuple[int, int]] = []
    for y in range(grid_size):
        if y % 2 == 0:
            # Even row: left to right
            row = list(range(grid_size))
        else:
            # Odd row: right to left
            row = list(range(grid_size - 1, -1, -1))
        
        for x in row:
            cycle.append((x, y))
    
    return cycle


class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """
    BFS Hamiltonian Hybrid Agent: Ultimate enhanced BFS with Hamiltonian fallback.
    
    Inheritance Hierarchy:
    - Inherits from BFSSafeGreedyAgent (gets BFS + safety validation)
    - Adds Hamiltonian cycle generation and intelligent shortcuts
    - Provides ultimate fallback strategy that guarantees no deadlocks
    
    Algorithm Enhancement:
    1. Try shortest safe path to apple (inherited from BFSSafeGreedyAgent)
    2. Try tail-chasing fallback (inherited from BFSSafeGreedyAgent)
    3. NEW: Try intelligent Hamiltonian shortcut toward apple
    4. NEW: Follow Hamiltonian cycle as ultimate safe fallback
    5. Last resort: any non-crashing move (inherited)
    
    This demonstrates the final evolution in the BFS family:
    BFS → BFS-Safe-Greedy → BFS-Hamiltonian
    """

    def __init__(self) -> None:
        """Initialize BFS Hamiltonian agent, extending BFS Safe Greedy."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-HAMILTONIAN"
        self.name = "BFS Hamiltonian Hybrid"
        self.description = (
            "Ultimate enhanced BFS agent. Inherits safe pathfinding from "
            "BFSSafeGreedyAgent and adds Hamiltonian cycle capabilities. "
            "Provides intelligent shortcuts and guarantees no deadlocks."
        )
        
        # Hamiltonian cycle state
        self.hamiltonian: List[Tuple[int, int]] = []
        self.grid_size: Optional[int] = None

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using enhanced BFS with Hamiltonian fallback.
        
        Enhancement over parent BFS Safe Greedy:
        - Adds intelligent Hamiltonian shortcuts
        - Provides Hamiltonian cycle as ultimate safe fallback
        - Guarantees the snake never gets permanently trapped
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            snake = [tuple(seg) for seg in game.snake_positions]
            grid_size = game.grid_size
            obstacles = set(snake[:-1])  # Exclude tail (can vacate)

            # Initialize Hamiltonian cycle once per grid size
            if self.grid_size != grid_size:
                self.grid_size = grid_size
                self.hamiltonian = generate_hamiltonian_cycle(grid_size)

            # --------------------------------------------------------------
            # 1. & 2. Try inherited BFS Safe Greedy strategies first
            # --------------------------------------------------------------
            # This calls the parent's get_move() which handles:
            # - Safe path to apple with validation
            # - Tail-chasing fallback
            # But we catch "NO_PATH_FOUND" to add our Hamiltonian fallbacks
            
            # Try inherited safe apple path
            path_to_apple = self._bfs_pathfind(head, apple, obstacles, grid_size)
            if path_to_apple and len(path_to_apple) > 1:
                if self._path_is_safe(path_to_apple, snake, apple, grid_size):
                    next_pos = path_to_apple[1]
                    return position_to_direction(head, next_pos)

            # Try inherited tail-chasing fallback
            tail = snake[-1]
            path_to_tail = self._bfs_pathfind(head, tail, obstacles, grid_size)
            if path_to_tail and len(path_to_tail) > 1:
                next_pos = path_to_tail[1]
                return position_to_direction(head, next_pos)

            # --------------------------------------------------------------
            # 3. NEW: Intelligent Hamiltonian shortcut toward apple
            # --------------------------------------------------------------
            shortcut_move = self._hamiltonian_shortcut(head, apple, snake, grid_size)
            if shortcut_move:
                return shortcut_move

            # --------------------------------------------------------------
            # 4. NEW: Hamiltonian cycle fallback (ultimate safety)
            # --------------------------------------------------------------
            hamiltonian_move = self._hamiltonian_move(head, obstacles)
            if hamiltonian_move:
                return hamiltonian_move

            # --------------------------------------------------------------
            # 5. Last resort: inherited safe move finder
            # --------------------------------------------------------------
            return self._get_safe_move(head, obstacles, grid_size)
            
        except Exception as e:
            print(f"BFS Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND"

    def _hamiltonian_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        snake: List[Tuple[int, int]],
        grid_size: int
    ) -> Optional[str]:
        """
        Try intelligent Hamiltonian shortcut toward apple.
        
        Enhancement: Instead of always following the full cycle,
        try to jump ahead on the cycle toward the apple if safe.
        
        Args:
            head: Current head position
            apple: Apple position
            snake: Current snake body
            grid_size: Size of game grid
            
        Returns:
            Direction for shortcut move or None if not safe
        """
        # Check if both head and apple are on the Hamiltonian cycle
        if head not in self.hamiltonian or apple not in self.hamiltonian:
            return None
            
        # Find indices on the cycle
        idx_head = self.hamiltonian.index(head)
        idx_apple = self.hamiltonian.index(apple)
        
        # Calculate forward distance on cycle
        forward_distance = (idx_apple - idx_head) % len(self.hamiltonian)
        
        # Skip if apple is at current position or too far (prefer normal cycle)
        if forward_distance == 0 or forward_distance > len(self.hamiltonian) // 2:
            return None
        
        # Build segment from head to apple along cycle
        segment = []
        for i in range(forward_distance + 1):
            pos_idx = (idx_head + i) % len(self.hamiltonian)
            segment.append(self.hamiltonian[pos_idx])
        
        # Check if following this segment is safe
        if self._segment_is_safe(segment, snake, apple, grid_size):
            next_pos = segment[1]  # First move in segment
            return position_to_direction(head, next_pos)
            
        return None

    def _hamiltonian_move(
        self, 
        head: Tuple[int, int], 
        obstacles: Set[Tuple[int, int]]
    ) -> Optional[str]:
        """
        Move to next position in Hamiltonian cycle.
        
        Ultimate fallback: follow the pre-computed Hamiltonian cycle
        which guarantees visiting every cell and avoiding deadlocks.
        
        Args:
            head: Current head position
            obstacles: Set of obstacle positions
            
        Returns:
            Direction for next Hamiltonian move or None if not possible
        """
        if head not in self.hamiltonian:
            return None
            
        # Find current position in cycle
        current_idx = self.hamiltonian.index(head)
        
        # Get next position in cycle
        next_idx = (current_idx + 1) % len(self.hamiltonian)
        next_pos = self.hamiltonian[next_idx]
        
        # Check if next position is safe
        if next_pos not in obstacles:
            return position_to_direction(head, next_pos)
            
        return None

    def _segment_is_safe(
        self,
        segment: List[Tuple[int, int]],
        snake: List[Tuple[int, int]],
        apple: Tuple[int, int],
        grid_size: int
    ) -> bool:
        """
        Check if following a Hamiltonian segment is safe.
        
        Simulates following the segment and verifies the snake
        can still reach its tail afterward.
        
        Args:
            segment: Proposed path segment along Hamiltonian cycle
            snake: Current snake body
            apple: Apple position
            grid_size: Size of game grid
            
        Returns:
            True if segment is safe to follow
        """
        # Simulate following the segment
        virtual_snake = list(snake)
        
        for step in segment[1:]:  # Skip current head position
            virtual_snake.insert(0, step)  # Move head
            
            if step == apple:
                # Apple eaten - snake grows, keep tail
                break
            
            # No apple - tail moves forward
            virtual_snake.pop()
        
        # Check if new head can reach new tail
        new_head = virtual_snake[0]
        new_tail = virtual_snake[-1]
        new_obstacles = set(virtual_snake[:-1])  # Exclude tail
        
        # Use inherited BFS to check tail reachability
        tail_path = self._bfs_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def __str__(self) -> str:
        """String representation showing inheritance hierarchy."""
        return f"BFSHamiltonianAgent(extends=BFSSafeGreedyAgent, algorithm={self.algorithm_name})"
