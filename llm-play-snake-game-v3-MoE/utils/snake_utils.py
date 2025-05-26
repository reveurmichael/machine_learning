"""
Snake game utility functions.
Provides utility functions for snake game mechanics.
"""

import traceback
import numpy as np
from pathlib import Path
import json
from colorama import Fore

def filter_invalid_reversals(moves, current_direction=None):
    """Filter out invalid reversal moves from a sequence.
    
    Args:
        moves: List of move directions
        current_direction: Current direction of the snake
        
    Returns:
        Filtered list of moves with invalid reversals removed
    """
    if not moves or len(moves) <= 1:
        return moves
        
    filtered_moves = []
    last_direction = current_direction if current_direction else moves[0]
    
    for move in moves:
        # Skip if this move would be a reversal of the last direction
        if (last_direction == "UP" and move == "DOWN") or \
           (last_direction == "DOWN" and move == "UP") or \
           (last_direction == "LEFT" and move == "RIGHT") or \
           (last_direction == "RIGHT" and move == "LEFT"):
            print(f"Filtering out invalid reversal move: {move} after {last_direction}")
        else:
            filtered_moves.append(move)
            last_direction = move
    
    # If all moves were filtered out, return empty list
    if not filtered_moves:
        print("All moves were invalid reversals. Not moving.")
        
    return filtered_moves

def calculate_move_differences(head_pos, apple_pos):
    """Calculate the expected move differences based on head and apple positions.
    
    Args:
        head_pos: Position of the snake's head as (x, y)
        apple_pos: Position of the apple as (x, y)
        
    Returns:
        String describing the expected move differences with actual numbers
    """
    head_x, head_y = head_pos
    apple_x, apple_y = apple_pos
    
    # Calculate horizontal differences
    x_diff_text = ""
    if head_x <= apple_x:
        x_diff = apple_x - head_x
        x_diff_text = f"#RIGHT - #LEFT = {x_diff} (= {apple_x} - {head_x})"
    else:
        x_diff = head_x - apple_x
        x_diff_text = f"#LEFT - #RIGHT = {x_diff} (= {head_x} - {apple_x})"
    
    # Calculate vertical differences
    y_diff_text = ""
    if head_y <= apple_y:
        y_diff = apple_y - head_y
        y_diff_text = f"#DOWN - #UP = {y_diff} (= {apple_y} - {head_y})"
    else:
        y_diff = head_y - apple_y
        y_diff_text = f"#UP - #DOWN = {y_diff} (= {head_y} - {apple_y})"
    
    return f"{x_diff_text}, and {y_diff_text}"

def extract_apple_positions(log_dir, game_number):
    """Extract apple positions from a game summary file.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to extract apple positions for
        
    Returns:
        List of apple positions as [x, y] arrays
    """
    log_dir_path = Path(log_dir)
    json_summary_file = log_dir_path / f"game{game_number}_summary.json"
    apple_positions = []
    
    if not json_summary_file.exists():
        print(f"No JSON summary file found for game {game_number}")
        return apple_positions
    
    try:
        with open(json_summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Extract apple positions from JSON
        if 'apple_positions' in summary_data and summary_data['apple_positions']:
            for pos in summary_data['apple_positions']:
                apple_positions.append(np.array([pos['x'], pos['y']]))
        
        print(f"Extracted {len(apple_positions)} apple positions from game {game_number} JSON summary")
    
    except Exception as e:
        print(f"Error extracting apple positions from JSON summary: {e}")
    
    return apple_positions

def validate_move(move, current_direction):
    """Validate if a move is valid given the current direction.
    
    Args:
        move: Move direction to validate
        current_direction: Current direction of the snake
        
    Returns:
        True if move is valid, False otherwise
    """
    if not move or not current_direction:
        return False
        
    # Check if move is a valid direction
    if move not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        return False
        
    # Check if move is not a reversal
    if (current_direction == "UP" and move == "DOWN") or \
       (current_direction == "DOWN" and move == "UP") or \
       (current_direction == "LEFT" and move == "RIGHT") or \
       (current_direction == "RIGHT" and move == "LEFT"):
        return False
        
    return True

def get_next_position(current_pos, move):
    """Calculate the next position based on current position and move.
    
    Args:
        current_pos: Current position as [x, y]
        move: Move direction
        
    Returns:
        Next position as [x, y]
    """
    x, y = current_pos
    
    if move == "UP":
        return [x, y - 1]
    elif move == "DOWN":
        return [x, y + 1]
    elif move == "LEFT":
        return [x - 1, y]
    elif move == "RIGHT":
        return [x + 1, y]
    else:
        return current_pos

def check_collision(position, snake_positions, board_size):
    """Check if a position collides with the snake or board boundaries.
    
    Args:
        position: Position to check as [x, y]
        snake_positions: List of snake body positions
        board_size: Size of the game board
        
    Returns:
        Tuple of (collision_type, is_collision)
        collision_type: Type of collision ("wall" or "self")
        is_collision: True if collision occurred, False otherwise
    """
    x, y = position
    
    # Check wall collision
    if x < 0 or x >= board_size[0] or y < 0 or y >= board_size[1]:
        return "wall", True
        
    # Check self collision
    if position in snake_positions:
        return "self", True
        
    return None, False

def calculate_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions.
    
    Args:
        pos1: First position as [x, y]
        pos2: Second position as [x, y]
        
    Returns:
        Manhattan distance between positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_available_moves(current_pos, current_direction, board_size, snake_positions):
    """Get list of available valid moves from current position.
    
    Args:
        current_pos: Current position as [x, y]
        current_direction: Current direction of the snake
        board_size: Size of the game board
        snake_positions: List of snake body positions
        
    Returns:
        List of valid move directions
    """
    available_moves = []
    
    for move in ["UP", "DOWN", "LEFT", "RIGHT"]:
        if validate_move(move, current_direction):
            next_pos = get_next_position(current_pos, move)
            collision_type, is_collision = check_collision(next_pos, snake_positions, board_size)
            if not is_collision:
                available_moves.append(move)
                
    return available_moves

def format_body_cells(body_positions):
    """Format the snake body cells for prompt representation.
    
    Args:
        body_positions: List of (x, y) positions representing the snake's body
        
    Returns:
        String representation of body cells
    """
    # Skip the head (first position) if it's included in body_positions
    if len(body_positions) > 1:
        body_cells = body_positions[1:]
    else:
        body_cells = []
        
    formatted_cells = []
    for x, y in body_cells:
        formatted_cells.append(f"({x},{y})")
    
    if formatted_cells:
        return "[" + ", ".join(formatted_cells) + "]"
    else:
        return "[]"  # Empty body 