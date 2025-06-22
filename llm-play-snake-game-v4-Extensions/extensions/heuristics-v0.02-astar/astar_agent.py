"""
A* Pathfinding Agent for Snake Game
===================================

This module implements the A* (A-star) pathfinding algorithm for Snake gameplay.
A* is an informed search algorithm that uses a heuristic function to guide the
search towards the goal, making it more efficient than uninformed search like BFS.

The algorithm uses f(n) = g(n) + h(n) where:
- g(n) = actual cost from start to node n
- h(n) = heuristic cost from node n to goal (Manhattan distance)
- f(n) = estimated total cost of path through node n

Key Features:
- Manhattan distance heuristic (admissible for grid-based movement)
- Priority queue for optimal node exploration
- Collision detection integration
- Same interface as BFS agent for easy substitution

Design Patterns:
- Strategy Pattern: Interchangeable pathfinding algorithm
- Template Method: Consistent agent interface
"""

import heapq
from typing import List, Tuple, Optional, Set, Dict
from config.game_constants import DIRECTIONS


class AStarAgent:
    """
    A* pathfinding agent for Snake game.
    
    Implements the A* algorithm with Manhattan distance heuristic to find
    optimal paths from the snake head to the apple position while avoiding
    collisions with walls and the snake's own body.
    
    The agent uses a priority queue to explore the most promising nodes first,
    making it more efficient than BFS while still guaranteeing optimal paths.
    
    Design Patterns:
    - Strategy Pattern: Pluggable pathfinding algorithm
    - Command Pattern: get_move() returns direction commands
    """
    
    def __init__(self):
        """Initialize the improved A* pathfinding agent."""
        self.name = "A*"
        self.description = (
            "A* with Manhattan heuristic, predictive tail modelling, dead-end "
            "avoidance and multi-layer safety fallbacks"
        )
    
    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """Compute the next move with tail prediction & layered safety."""
        try:
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            grid_size = game.grid_size

            body_list = [tuple(pos) for pos in game.snake_positions]

            # ----------------------------------------------------------
            # Edge-case: head already occupies the apple tile
            # ----------------------------------------------------------
            if head_pos == apple_pos:
                return self._get_any_safe_direction(head_pos, set(body_list), grid_size)

            # ----------------------------------------------------------
            # Tail modelling – will the snake grow this tick?
            # ----------------------------------------------------------
            tail = body_list[-1]
            will_grow = apple_pos == tail  # tail remains stationary when eating

            obstacles: set[Tuple[int, int]] = set(body_list[:-1]) if not will_grow else set(body_list)

            future_tail = None if will_grow else self._predict_tail_movement(body_list)

            # ----------------------------------------------------------
            # Primary path planning with A*
            # ----------------------------------------------------------
            path = self._astar_pathfind(
                head_pos,
                apple_pos,
                obstacles,
                grid_size,
                future_tail=future_tail,
            )

            if path and len(path) >= 2:
                next_pos = path[1]

                # Dead-end detection – avoid moves that trap the snake
                if future_tail and self._is_future_dead_end(
                    next_pos, apple_pos, obstacles, grid_size, future_tail
                ):
                    return self._get_alternate_safe_move(head_pos, obstacles, grid_size, next_pos)

                return self._get_direction(head_pos, next_pos)

            # ------------------------------------------------------
            # Fallback layers – try any safe move, then evasion
            # ------------------------------------------------------
            move = self._get_any_safe_direction(head_pos, obstacles, grid_size)
            if move != "NO_PATH_FOUND":
                return move

            return self._emergency_evasion(head_pos, grid_size)

        except Exception as e:  # pragma: no cover – defensive catch-all
            print(f"A* Agent error: {e}")
            return self._emergency_evasion(tuple(game.head_position), game.grid_size)
    
    def _astar_pathfind(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_size: int,
        future_tail: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm implementation.
        
        Uses Manhattan distance as heuristic function and priority queue
        for optimal node exploration.
        
        Args:
            start: Starting position (snake head)
            goal: Goal position (apple)
            obstacles: Set of obstacle positions (snake body)
            grid_size: Size of the game grid
            
        Returns:
            List of positions forming the optimal path, or empty list if no path
        """
        # Priority queue: (f_score, position)
        start_h = self._manhattan_distance(start, goal)
        open_set: List[Tuple[int, Tuple[int, int]]] = [(start_h, start)]

        # Track best-known g(s)
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: Set[Tuple[int, int]] = set()

        while open_set:
            f_score, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            current_g = g_scores[current]

            for dx, dy in DIRECTIONS.values():
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid_position(neighbor, grid_size):
                    continue

                if (
                    neighbor in obstacles
                    and neighbor != goal
                    and not (future_tail and neighbor == future_tail)
                ):
                    continue

                tentative_g = current_g + 1

                if tentative_g < g_scores.get(neighbor, 1_000_000):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_neighbor = tentative_g + self._manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_neighbor, neighbor))
        
        # No path found
        return []
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Manhattan distance is admissible for grid-based movement, guaranteeing
        optimal paths when used as a heuristic in A*.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Manhattan distance between the positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_valid_position(self, pos: Tuple[int, int], grid_size: int) -> bool:
        """
        Check if position is within grid boundaries.
        
        Args:
            pos: Position to check
            grid_size: Size of the game grid
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        return 0 <= x < grid_size and 0 <= y < grid_size
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the optimal path from came_from mapping.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: Goal node to trace back from
            
        Returns:
            List of positions forming the complete path from start to goal
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # Reverse to get path from start to goal
        return path[::-1]
    
    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        Convert position difference to direction string.
        
        Uses the same coordinate system as the base game for consistency.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Map coordinate differences to directions
        # Based on DIRECTIONS = {"UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0)}
        if dx == 0 and dy == 1:
            return "UP"
        elif dx == 0 and dy == -1:
            return "DOWN"
        elif dx == 1 and dy == 0:
            return "RIGHT"
        elif dx == -1 and dy == 0:
            return "LEFT"
        else:
            return "NO_PATH_FOUND"  # Invalid move 

    def _get_any_safe_direction(self, head: Tuple[int, int], obstacles: Set[Tuple[int, int]], grid_size: int) -> str:
        """Return any safe direction when already on the apple (edge-case)."""
        for direction, (dx, dy) in DIRECTIONS.items():
            n = (head[0] + dx, head[1] + dy)
            if self._is_valid_position(n, grid_size) and n not in obstacles:
                return direction
        return "NO_PATH_FOUND" 

    # ------------------------------------------------------------------
    # Tail prediction & dead-end detection helpers
    # ------------------------------------------------------------------

    def _predict_tail_movement(self, body: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return the coordinate the tail is expected to vacate next tick."""
        if len(body) < 2:
            return None
        tail, before_tail = body[-1], body[-2]
        dx, dy = tail[0] - before_tail[0], tail[1] - before_tail[1]
        return (tail[0] + dx, tail[1] + dy)

    def _is_future_dead_end(
        self,
        next_pos: Tuple[int, int],
        apple: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_size: int,
        future_tail: Tuple[int, int],
    ) -> bool:
        """Heuristic: true if moving to `next_pos` leaves no path to apple nor space."""
        temp_obstacles = obstacles - {future_tail}
        # 1) Is there still a path to the apple?
        if self._astar_pathfind(next_pos, apple, temp_obstacles, grid_size):
            return False
        # 2) Does the position have *any* free neighbor (post-move)?
        for dx, dy in DIRECTIONS.values():
            n = (next_pos[0] + dx, next_pos[1] + dy)
            if self._is_valid_position(n, grid_size) and n not in temp_obstacles:
                return False
        return True

    # ------------------------------------------------------------------
    # Layer-2 safety: choose alt move with most breathing room
    # ------------------------------------------------------------------

    def _get_alternate_safe_move(
        self,
        head: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_size: int,
        avoid_pos: Tuple[int, int],
    ) -> str:
        candidates: list[tuple[str, int]] = []
        for dir_name, (dx, dy) in DIRECTIONS.items():
            n = (head[0] + dx, head[1] + dy)
            if n == avoid_pos:
                continue
            if self._is_valid_position(n, grid_size) and n not in obstacles:
                candidates.append((dir_name, self._count_adjacent_free(n, obstacles, grid_size)))
        if not candidates:
            return self._get_any_safe_direction(head, obstacles, grid_size)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _count_adjacent_free(
        self, pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], grid_size: int
    ) -> int:
        return sum(
            1
            for dx, dy in DIRECTIONS.values()
            if self._is_valid_position((pos[0] + dx, pos[1] + dy), grid_size)
            and (pos[0] + dx, pos[1] + dy) not in obstacles
        )

    # ------------------------------------------------------------------
    # Layer-3 safety helpers
    # ------------------------------------------------------------------

    def _get_any_safe_direction(
        self, head: Tuple[int, int], obstacles: Set[Tuple[int, int]], grid_size: int
    ) -> str:
        options: list[tuple[str, int]] = []
        for dir_name, (dx, dy) in DIRECTIONS.items():
            n = (head[0] + dx, head[1] + dy)
            if self._is_valid_position(n, grid_size) and n not in obstacles:
                options.append((dir_name, self._count_adjacent_free(n, obstacles, grid_size)))
        if not options:
            # Last-ditch: just stay within bounds if possible
            for dir_name, (dx, dy) in DIRECTIONS.items():
                n = (head[0] + dx, head[1] + dy)
                if self._is_valid_position(n, grid_size):
                    return dir_name
            return "NO_PATH_FOUND"
        options.sort(key=lambda x: x[1], reverse=True)
        return options[0][0]

    def _emergency_evasion(self, head: Tuple[int, int], grid_size: int) -> str:
        for dir_name, (dx, dy) in DIRECTIONS.items():
            n = (head[0] + dx, head[1] + dy)
            if self._is_valid_position(n, grid_size):
                return dir_name
        return "NO_PATH_FOUND" 