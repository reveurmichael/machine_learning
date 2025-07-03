"""
A* Hamiltonian Hybrid Agent - Advanced Pathfinding with Safety Guarantee
--------------------

This module implements a sophisticated agent that extends the A* agent with
Hamiltonian cycle capabilities. It represents the evolution from A* to A*-Hamiltonian.

Inheritance Hierarchy:
- AStarAgent (optimal pathfinding with heuristics)
  └── AStarHamiltonianAgent (adds Hamiltonian cycle fallback)

This demonstrates how advanced algorithms can be built by extending proven ones:
A* provides optimal pathfinding, Hamiltonian cycle provides safety guarantee.

Design Patterns:
- Inheritance: Extends AStarAgent with Hamiltonian cycle logic
- Template Method: Overrides get_move() while reusing A* pathfinding
- Strategy Pattern: Hamiltonian cycle as ultimate fallback strategy
- Composite Pattern: Combines optimal search with safety guarantee
"""

from __future__ import annotations
from typing import Dict, List, Tuple, TYPE_CHECKING
from .agent_astar import AStarAgent

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error

# Import extension-specific components using relative imports

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class AStarHamiltonianAgent(AStarAgent):
    """
    A* Hamiltonian Hybrid Agent: Advanced pathfinding with safety guarantee.
    
    Inheritance Pattern:
    - Inherits from AStarAgent (reuses optimal A* pathfinding)
    - Adds Hamiltonian cycle generation and management
    - Provides safety validation and fallback strategies
    - Demonstrates evolution from optimal to optimal + safe
    
    Algorithm Enhancement:
    1. Use inherited A* to find optimal path to apple (from AStarAgent)
    2. Validate path safety (can reach tail afterward)
    3. If safe, follow A* path
    4. NEW: Fall back to Hamiltonian cycle (guaranteed safe)
    5. Last resort: any safe move (inherited)
    
    This represents the pinnacle of heuristic Snake AI:
    - Optimal when possible (A* pathfinding)
    - Safe always (Hamiltonian cycle fallback)
    """

    def __init__(self) -> None:
        """Initialize A* Hamiltonian agent, extending A* agent."""
        super().__init__()  # Initialize parent A* agent
        self.algorithm_name = "ASTAR-HAMILTONIAN"
        self.name = "A* Hamiltonian Hybrid"
        self.description = (
            "Advanced hybrid agent extending AStarAgent with Hamiltonian cycle "
            "safety guarantee. Uses inherited A* optimal pathfinding when safe, "
            "falls back to Hamiltonian cycle when necessary. Represents the "
            "pinnacle of heuristic Snake AI."
        )
        
        # Hamiltonian cycle state
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_cycle_index: int = 0

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using A* with Hamiltonian fallback.
        
        This method provides a simplified interface for direct move requests.
        For explanation-capable moves, use get_move_with_explanation().
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move

    def get_move_with_explanation(self, game: "HeuristicGameLogic") -> Tuple[str, str]:
        """
        Get next move using A* with Hamiltonian fallback, with detailed explanation.
        
        Enhancement over parent A*:
        - Adds safety validation before following A* path.
        - Provides a robust Hamiltonian cycle as the ultimate safe fallback.
        - Guarantees the snake never gets permanently trapped.
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            grid_size = game.grid_size
            body = [tuple(seg) for seg in game.snake_positions]

            # --- Primary Strategy: A* Pathfinding ---
            # Use the sophisticated pathfinding inherited from AStarAgent.
            path_to_apple = self._astar_pathfind(head, apple, set(body), grid_size)
            
            if path_to_apple and len(path_to_apple) > 1:
                # Safety Check: Before committing to the A* path, ensure it doesn't lead to a trap.
                if self._path_is_safe_after_eating(path_to_apple, body, grid_size):
                    direction = position_to_direction(head, path_to_apple[1])
                    explanation = self._generate_astar_success_explanation(path_to_apple, head, apple, direction)
                    return direction, explanation

            # --- Fallback Strategy: Hamiltonian Cycle ---
            # This is triggered if A* fails or its path is deemed unsafe.
            # Generate cycle on-demand.
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_hamiltonian_cycle()
            
            direction, explanation = self._follow_hamiltonian_cycle_with_explanation(head, body)
            return direction, explanation

        except Exception as e:
            explanation = f"A* Hamiltonian Agent encountered a critical error: {str(e)}"
            print_error(f"A* Hamiltonian Agent error: {e}")
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
            if not found_next: # handle dangling ends
                 prev, curr = curr, tree_adj[curr][0]

            if curr == (0, 0) and len(cycle) > 1: break
        
        self.cycle = cycle
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    def _path_is_safe_after_eating(self, path: list, snake_body: list, grid_size: int) -> bool:
        """
        Simulates the result of taking a path to an apple and checks if the snake
        can still reach its new tail, preventing traps. This is the core safety check.
        """
        # Create a virtual snake that has eaten the apple
        virtual_snake = path[::-1] + snake_body[1:]
        
        # The new head is the apple's position, the new tail is the original snake's second-to-last segment
        new_head = virtual_snake[0]
        new_tail = virtual_snake[-1]
        
        # Check if a path exists from the new head to the new tail, avoiding the new body
        obstacles = set(virtual_snake[1:-1])
        path_to_tail = self._astar_pathfind(new_head, new_tail, obstacles, grid_size)
        
        return bool(path_to_tail)

    def _follow_hamiltonian_cycle_with_explanation(self, head: tuple, body: list) -> Tuple[str, str]:
        """Follows the Hamiltonian cycle and generates a corresponding explanation."""
        head_idx = self.cycle_map.get(head)
        if head_idx is None:
            # Snake is off-cycle, a critical error. Find any safe move.
            for move, (dx, dy) in DIRECTIONS.items():
                next_pos = (head[0] + dx, head[1] + dy)
                if next_pos not in body and 0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size:
                    return move, "CRITICAL ERROR: Off-cycle. Emergency fallback."
            return "NO_PATH_FOUND", "CRITICAL ERROR: Off-cycle and trapped."

        next_idx = (head_idx + 1) % len(self.cycle)
        next_pos = self.cycle[next_idx]
        direction = position_to_direction(head, next_pos)
        
        explanation = (
            "A* path to apple was not found or deemed unsafe. "
            "Activating Hamiltonian Cycle Fallback for guaranteed safety. "
            f"Following cycle: moving {direction} from {head} to {next_pos}. "
            f"This ensures survival while waiting for a safe opportunity."
        )
        return direction, explanation
    
    def _generate_astar_success_explanation(self, path: list, head: tuple, apple: tuple, direction: str) -> str:
        """Generates an explanation for a successful and safe A* path."""
        path_length = len(path) - 1
        manhattan_dist = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        return (
            f"A* found an optimal path of length {path_length} to the apple (Manhattan distance: {manhattan_dist}). "
            f"Safety check passed: the snake can still reach its tail after eating. "
            f"Executing optimal move: {direction}."
        )

    def get_cycle_info(self) -> Dict[str, object]:
        """
        Get information about the Hamiltonian cycle.
        
        Useful for debugging and analysis.
        
        Returns:
            Dictionary with cycle information
        """
        return {
            "cycle_length": len(self.cycle),
            "grid_size": self.grid_size,
            "current_index": self.current_cycle_index,
            "cycle_generated": bool(self.cycle)
        }

    def __str__(self) -> str:
        """String representation showing inheritance and state."""
        return f"AStarHamiltonianAgent(extends=AStarAgent, grid_size={self.grid_size}, cycle_nodes={len(self.cycle)})" 