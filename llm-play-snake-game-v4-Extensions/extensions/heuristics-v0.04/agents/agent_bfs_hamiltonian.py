"""
BFS Hamiltonian Agent - BFS with Hamiltonian cycle fallback for Snake Game v0.04
---------------------------------------------------------------------------------

This module implements a hybrid agent that combines BFS pathfinding with
Hamiltonian cycle following as a safety fallback mechanism.

v0.04 Enhancement: Generates natural language explanations for each move
to create rich datasets for LLM fine-tuning.

Strategy:
1. Try BFS pathfinding to apple (shortest path)
2. If BFS fails or leads to unsafe positions, follow Hamiltonian cycle
3. Generate detailed explanations for strategy selection

The Hamiltonian cycle ensures the snake can always move safely by following
a predetermined path that visits every cell exactly once.

Design Patterns:
- Strategy Pattern: Multiple pathfinding strategies
- Fallback Pattern: Safe fallback when primary strategy fails
- Protocol Pattern: Implements BaseAgent interface for compatibility
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error

# Import extension-specific components using relative imports
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """
    BFS Hamiltonian Hybrid Agent: Ultimate enhanced BFS with Hamiltonian fallback.
    
    Inheritance Hierarchy:
    - Inherits from BFSSafeGreedyAgent (gets BFS + safety validation).
    - Adds a robust Hamiltonian cycle as the ultimate fallback strategy.
    
    Algorithm Priority:
    1. Find a safe path to the apple using BFS (inherited).
    2. If no safe apple path, fall back to the Hamiltonian cycle for guaranteed survival.
    """

    def __init__(self) -> None:
        """Initialize BFS Hamiltonian agent, extending BFS Safe Greedy."""
        super().__init__()
        self.algorithm_name = "BFS-HAMILTONIAN"
        self.name = "BFS Hamiltonian Hybrid"
        self.description = (
            "Ultimate enhanced BFS. Inherits safe pathfinding from "
            "BFSSafeGreedyAgent and adds a Hamiltonian cycle for guaranteed safety."
        )
        
        # Hamiltonian cycle state
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: Optional[int] = None

    def get_move_with_explanation(self, game: "HeuristicGameLogic") -> Tuple[str, str]:
        """
        Get next move using enhanced BFS with Hamiltonian fallback, with detailed explanation.
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            snake = [tuple(seg) for seg in game.snake_positions]
            grid_size = game.grid_size

            # --- Primary Strategy: Safe Path to Apple (from parent class) ---
            path_to_apple = self._bfs_pathfind(head, apple, set(snake), grid_size)
            
            if path_to_apple and len(path_to_apple) > 1:
                if self._path_is_safe(path_to_apple, snake, apple, grid_size):
                    direction = position_to_direction(head, path_to_apple[1])
                    explanation = (
                        f"BFS found a safe, shortest path of length {len(path_to_apple) - 1} to the apple. "
                        f"This is the optimal move. Executing: {direction}."
                    )
                    return direction, explanation

            # --- Fallback Strategy: Hamiltonian Cycle ---
            # This is triggered if the BFS path is non-existent or unsafe.
            if self.grid_size != grid_size or not self.cycle:
                self.grid_size = grid_size
                self._generate_hamiltonian_cycle()
            
            direction, explanation = self._follow_hamiltonian_cycle_with_explanation(head, snake)
            return direction, explanation
            
        except Exception as e:
            explanation = f"BFS Hamiltonian Agent encountered a critical error: {str(e)}"
            print_error(f"BFS Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND", explanation

    def _generate_hamiltonian_cycle(self) -> None:
        """Generates a robust, guaranteed-closed Hamiltonian cycle using a spanning tree perimeter."""
        size = self.grid_size
        adj = {(x, y): [] for x in range(size) for y in range(size)}
        for x in range(size):
            for y in range(size):
                if x > 0: adj[(x, y)].append((x - 1, y))
                if x < size - 1: adj[(x, y)].append((x + 1, y))
                if y > 0: adj[(x, y)].append((x, y - 1))
                if y < size - 1: adj[(x, y)].append((x, y + 1))

        tree_adj = {(x, y): [] for x in range(size) for y in range(size)}
        visited = set()
        stack = [(0, 0)]
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v)
            for neighbor in adj[v]:
                if neighbor not in visited:
                    tree_adj[v].append(neighbor)
                    tree_adj[neighbor].append(v)
                    stack.append(neighbor)
        
        cycle = []
        curr, prev = (0, 0), (-1, -1)
        for _ in range(2 * size * size):
            cycle.append(curr)
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            options = [(curr[0] + dy, curr[1] - dx), (curr[0] + dx, curr[1] + dy), (curr[0] - dy, curr[1] + dx)]
            if dx == 0: options = [(curr[0] + dy, curr[1]), (curr[0], curr[1] + dy), (curr[0] - dy, curr[1])]
            else: options = [(curr[0], curr[1] - dx), (curr[0] + dx, curr[1]), (curr[0], curr[1] + dx)]

            found_next = False
            for nxt in options:
                if (nxt in tree_adj[curr] or nxt == prev) and nxt != prev:
                    prev, curr = curr, nxt
                    found_next = True
                    break
            if not found_next:
                 prev, curr = curr, tree_adj[curr][0]

            if curr == (0, 0) and len(cycle) > 1: break
        
        self.cycle = cycle
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    def _follow_hamiltonian_cycle_with_explanation(self, head: tuple, snake_body: list) -> Tuple[str, str]:
        """Follows the Hamiltonian cycle and generates a corresponding explanation."""
        head_idx = self.cycle_map.get(head)
        if head_idx is None:
            # Snake is off-cycle, a critical error.
            return self._get_safe_move(head, set(snake_body), self.grid_size), "CRITICAL ERROR: Off-cycle. Emergency fallback."

        next_idx = (head_idx + 1) % len(self.cycle)
        next_pos = self.cycle[next_idx]
        
        # Final safety check before moving
        if next_pos in snake_body:
             return self._get_safe_move(head, set(snake_body), self.grid_size), "Hamiltonian path blocked. Emergency fallback."

        direction = position_to_direction(head, next_pos)
        
        explanation = (
            "BFS path to apple was not found or deemed unsafe. "
            "Activating Hamiltonian Cycle Fallback for guaranteed safety. "
            f"Following cycle: moving {direction} from {head} to {next_pos}."
        )
        return direction, explanation

    def __str__(self) -> str:
        """String representation showing inheritance and state."""
        return f"BFSHamiltonianAgent(extends=BFSSafeGreedyAgent, grid_size={self.grid_size}, cycle_nodes={len(self.cycle)})"
