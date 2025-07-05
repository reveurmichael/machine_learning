"""
JSONL Prompt Formatting Utilities for Heuristics Extensions

This module provides JSONL-specific prompt formatting utilities for generating
rich prompts for LLM fine-tuning datasets.

Design Philosophy:
- Single responsibility: Only handles JSONL prompt formatting
- Agent-agnostic: Works with any agent and game state
- Rich prompts: Generate detailed, educational prompts for fine-tuning
- Consistent format: Standardized prompt structure across all agents

Usage:
    # Self-referential import removed - this file contains the function
    
    # Format a prompt for JSONL dataset entry
    prompt = format_prompt_for_jsonl(game_state, agent_name)
"""

from typing import Dict, List, Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.print_utils import print_warning


def format_prompt_for_jsonl(game_state: Dict[str, Any], agent_name: str) -> str:
    """
    Format a rich prompt for JSONL dataset entry.
    
    Args:
        game_state: Game state dictionary
        agent_name: Name of the agent
        
    Returns:
        Formatted prompt string
    """
    # Extract game state information
    grid_size = game_state.get('grid_size', 10)
    score = game_state.get('score', 0)
    steps = game_state.get('steps', 0)
    snake_length = game_state.get('snake_length', 1)
    
    # Robustly extract head_pos and apple_pos
    head_pos = game_state.get('head_position', [0, 0])
    apple_pos = game_state.get('apple_position', [0, 0])
    
    # Defensive: ensure both are lists/tuples of length 2 with ints
    def safe_pos(pos, name):
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                return [int(pos[0]), int(pos[1])]
            except Exception:
                pass
        # If invalid, log warning and return [0, 0]
        print_warning(f"[PromptFormatter] Invalid {name} position: {pos}, using [0, 0]")
        return [0, 0]
    
    head_pos = safe_pos(head_pos, 'head')
    apple_pos = safe_pos(apple_pos, 'apple')
    
    snake_positions = game_state.get('snake_positions', [])
    
    # Generate board visualization
    board_viz = generate_board_visualization(snake_positions, apple_pos, grid_size)
    
    # Calculate Manhattan distance to apple
    manhattan_distance = abs(head_pos[0] - apple_pos[0]) + abs(head_pos[1] - apple_pos[1])
    
    # SSOT: Get valid moves from ssot_utils
    from ssot_utils import ssot_calculate_valid_moves
    valid_moves = ssot_calculate_valid_moves(head_pos, snake_positions, grid_size)
    
    # Format the prompt
    prompt = f"""### Instruction:
You are an expert Snake game AI. Your task is to analyze the provided game state and determine the single best move from the list of valid moves. Your decision should be based on the logic of the specified heuristic algorithm.

### Input:
**Algorithm:** {agent_name.lower()}
**Game State:**
- Grid Size: {grid_size}x{grid_size}
- Score: {score}
- Steps: {steps}
- Snake Length: {snake_length}
- Head Position: {head_pos}
- Apple Position: {apple_pos}

**Board:**
{board_viz}

**Strategic Context:**
- Manhattan Distance to Apple: {manhattan_distance}
- Valid Moves: {valid_moves}

### Task:
Based on the {agent_name.lower()} logic, what is the optimal next move? Provide the move and a detailed, step-by-step explanation of the reasoning."""
    
    return prompt


def generate_board_visualization(snake_positions: List, apple_position: List, grid_size: int) -> str:
    """
    Generate a text-based board visualization.
    
    Args:
        snake_positions: List of snake positions
        apple_position: Apple position
        grid_size: Size of the game grid
        
    Returns:
        Board visualization string
    """
    # Create empty board
    board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Place snake body segments (exclude head if more than one segment)
    if len(snake_positions) > 1:
        for pos in snake_positions[:-1]:  # All except head (head is at index -1)
            if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                board[pos[1]][pos[0]] = 'S'
    
    if snake_positions:
        head_pos = snake_positions[-1]  # Head is at index -1 (last element)
        if 0 <= head_pos[0] < grid_size and 0 <= head_pos[1] < grid_size:
            board[head_pos[1]][head_pos[0]] = 'H'
    
    # Place apple - handle different data types safely
    try:
        if isinstance(apple_position, (list, tuple)) and len(apple_position) >= 2:
            apple_x, apple_y = apple_position[0], apple_position[1]
            if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                board[apple_y][apple_x] = 'A'
    except (IndexError, TypeError, KeyError):
        # Skip apple placement if position is invalid
        pass
    
    # Convert to string
    board_lines = []
    for row in board:
        board_lines.append(' '.join(row))
    
    return '\n'.join(board_lines)


# SSOT: Valid moves calculation is implemented in ssot_utils.py
# Do not reimplement here - use ssot_calculate_valid_moves from ssot_utils


def _get_next_position(current_pos: List[int], move: str, grid_size: int) -> List[int]:
    """
    Calculate the next position based on current position and move.
    
    Args:
        current_pos: Current position [x, y]
        move: Move direction (UP, DOWN, LEFT, RIGHT)
        grid_size: Size of the game grid
        
    Returns:
        Next position [x, y] if within bounds, None otherwise
        
    Note: 
        Uses universal coordinate system from docs/extensions-guideline/coordinate-system.md:
        - UP: (0, 1) - Move up (increase Y)
        - DOWN: (0, -1) - Move down (decrease Y)
        - LEFT: (-1, 0) - Move left (decrease X)
        - RIGHT: (1, 0) - Move right (increase X)
    """
    x, y = current_pos
    
    if move == 'UP':
        new_pos = [x, y + 1]  # UP increases Y (bottom-left origin)
    elif move == 'DOWN':
        new_pos = [x, y - 1]  # DOWN decreases Y
    elif move == 'LEFT':
        new_pos = [x - 1, y]  # LEFT decreases X
    elif move == 'RIGHT':
        new_pos = [x + 1, y]  # RIGHT increases X
    else:
        return None
    
    # Check bounds
    if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
        return new_pos
    else:
        return None 