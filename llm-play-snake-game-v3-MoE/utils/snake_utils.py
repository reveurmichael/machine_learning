"""
Snake game utility functions.
Provides utility functions for snake game mechanics.
"""

import traceback
import numpy as np
from pathlib import Path
import json

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
        head_pos: Position of the snake's head as [x, y]
        apple_pos: Position of the apple as [x, y]
        
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
        y_diff_text = f"#UP - #DOWN = {y_diff} (= {apple_y} - {head_y})"
    else:
        y_diff = head_y - apple_y
        y_diff_text = f"#DOWN - #UP = {y_diff} (= {head_y} - {apple_y})"
    
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