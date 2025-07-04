"""
CSV Utilities for Heuristics Extensions

This module provides CSV-specific formatting utilities for generating
CSV entries from game states and agent explanations.

Design Philosophy:
- Single responsibility: Only handles CSV formatting
- Agent-agnostic: Works with any agent and game state
- Consistent schema: Standardized CSV output format across all agents

Usage:
    from extensions.common.utils.csv_utils import create_csv_record
    
    # Create a CSV record
    csv_record = create_csv_record(game_state, move, step_number, metrics)
"""

from typing import Dict, Any

def create_csv_record(game_state: Dict[str, Any], move: str, step_number: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a CSV record from game state and move data.
    
    Args:
        game_state: Game state dictionary
        move: Move made by the agent
        step_number: Step number in the game
        metrics: Agent metrics dictionary
        
    Returns:
        CSV record dictionary
    """
    # Extract basic game state
    head_pos = game_state.get('snake_head', [0, 0])
    apple_pos = game_state.get('apple_position', [0, 0])
    snake_length = game_state.get('snake_length', 1)
    grid_size = game_state.get('grid_size', 10)
    
    # Defensive: ensure both are lists/tuples of length 2 with ints
    def safe_pos(pos, name):
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                return [int(pos[0]), int(pos[1])]
            except Exception:
                pass
        # If invalid, log warning and return [0, 0]
        from utils.print_utils import print_warning
        print_warning(f"[CSVUtils] Invalid {name} position: {pos}, using [0, 0]")
        return [0, 0]
    
    head_pos = safe_pos(head_pos, 'head')
    apple_pos = safe_pos(apple_pos, 'apple')
    
    # Calculate apple direction flags
    apple_dir_up = 1 if apple_pos[1] < head_pos[1] else 0
    apple_dir_down = 1 if apple_pos[1] > head_pos[1] else 0
    apple_dir_left = 1 if apple_pos[0] < head_pos[0] else 0
    apple_dir_right = 1 if apple_pos[0] > head_pos[0] else 0
    
    # Calculate danger flags (simplified)
    danger_straight = 0  # Would need more complex logic for actual danger detection
    danger_left = 0
    danger_right = 0
    
    # Calculate free space (simplified)
    free_space_up = max(0, head_pos[1])  # Distance to top wall
    free_space_down = max(0, grid_size - 1 - head_pos[1])  # Distance to bottom wall
    free_space_left = max(0, head_pos[0])  # Distance to left wall
    free_space_right = max(0, grid_size - 1 - head_pos[0])  # Distance to right wall
    
    # Create CSV record
    csv_record = {
        'game_id': 1,  # Would need to be passed in or calculated
        'step_in_game': step_number,
        'head_x': head_pos[0],
        'head_y': head_pos[1],
        'apple_x': apple_pos[0],
        'apple_y': apple_pos[1],
        'snake_length': snake_length,
        'apple_dir_up': apple_dir_up,
        'apple_dir_down': apple_dir_down,
        'apple_dir_left': apple_dir_left,
        'apple_dir_right': apple_dir_right,
        'danger_straight': danger_straight,
        'danger_left': danger_left,
        'danger_right': danger_right,
        'free_space_up': free_space_up,
        'free_space_down': free_space_down,
        'free_space_left': free_space_left,
        'free_space_right': free_space_right,
        'target_move': move
    }
    
    return csv_record 