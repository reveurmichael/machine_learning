"""
Snake game utility functions.
Provides snake game mechanics like collision detection and move validation.
"""

import traceback
from utils.json_utils import extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays
import numpy as np

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

def parse_llm_response(response, processed_response_func, game_instance):
    """Parse the LLM's response to extract multiple sequential moves.
    
    Args:
        response: Text response from the LLM in JSON format
        processed_response_func: Function to process the response for display
        game_instance: The game instance with all necessary attributes
        
    Returns:
        The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
        or None if no valid moves were found
    """
    try:
        # Store the raw response for display
        game_instance.last_llm_response = response
        
        # Process the response for display
        game_instance.processed_response = processed_response_func(response)
        
        # Reset planned moves
        game_instance.planned_moves = []
        
        # Print raw response snippet for debugging
        print(f"Parsing LLM response: '{response[:50]}...'")
        
        # Method 1: Try to extract from JSON code block
        json_data = extract_json_from_code_block(response)
        
        # Method 2: Try to extract JSON from regular text if code block fails
        if not json_data or "moves" not in json_data or not json_data["moves"]:
            json_data = extract_json_from_text(response)
            
        # Extract moves from JSON if found
        if json_data and "moves" in json_data and isinstance(json_data["moves"], list):
            valid_moves = [move.upper() for move in json_data["moves"] 
                         if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            if valid_moves:
                game_instance.planned_moves = valid_moves
                print(f"Found {len(game_instance.planned_moves)} moves in JSON: {game_instance.planned_moves}")

        # Method 3: Try finding arrays if other methods failed
        if not game_instance.planned_moves:
            array_moves = extract_moves_from_arrays(response)
            if array_moves:
                game_instance.planned_moves = array_moves
                print(f"Found {len(game_instance.planned_moves)} moves in array format: {game_instance.planned_moves}")
        
        # If we still have no moves, leave planned_moves empty
        if not game_instance.planned_moves:
            print("No valid directions found. Not moving.")
        else:
            # Filter out invalid reversal moves if we have moves
            current_direction = game_instance._get_current_direction_key()
            game_instance.planned_moves = filter_invalid_reversals(game_instance.planned_moves, current_direction)
        
        # Get the next move from the sequence (or None if empty)
        if game_instance.planned_moves:
            next_move = game_instance.planned_moves.pop(0)
            return next_move
        else:
            return None
            
    except Exception as e:
        print(f"Error in parse_llm_response: {e}")
        traceback.print_exc()
        
        # Ensure the game can continue even if parsing fails
        game_instance.last_llm_response = response
        
        # Store an error message as processed response
        try:
            game_instance.processed_response = f"ERROR: Failed to parse response: {str(e)}\n\n{response[:200]}..."
        except:
            game_instance.processed_response = "ERROR: Failed to parse response and display error details"
            
        # Clear planned moves
        game_instance.planned_moves = []
        
        return None 

def is_collision(snake_head, snake_positions, grid_size):
    """Check if a collision has occurred.
    
    Args:
        snake_head: Current position of the snake head
        snake_positions: List of all positions the snake occupies
        grid_size: Size of the game grid
        
    Returns:
        Boolean indicating if a collision has occurred and collision type
    """
    # Get position values
    head_x, head_y = snake_head
    
    # Check wall collision
    if (head_x < 0 or head_x >= grid_size or 
        head_y < 0 or head_y >= grid_size):
        return True, "wall"
    
    # Check self collision (skip head position which is at index 0)
    if len(snake_positions) > 1 and any(
        head_x == x and head_y == y for x, y in snake_positions[1:]):
        return True, "self"
    
    return False, None

def generate_apple(snake_positions, grid_size, apple_pos=None):
    """Generate a new apple position that doesn't overlap with the snake.
    
    Args:
        snake_positions: List of all positions the snake occupies
        grid_size: Size of the game grid
        apple_pos: Predefined apple position (for replays)
        
    Returns:
        Coordinates of the new apple as a numpy array
    """
    # If a position is provided (for replay), use it
    if apple_pos is not None:
        return apple_pos
    
    # Generate random position
    while True:
        apple = np.array([
            np.random.randint(0, grid_size),
            np.random.randint(0, grid_size)
        ])
        
        # Check if the apple overlaps with the snake
        if not any(apple[0] == x and apple[1] == y for x, y in snake_positions):
            return apple

def update_snake(snake_positions, direction, apple_pos):
    """Update the snake's position based on the direction and apple position.
    
    Args:
        snake_positions: List of all positions the snake occupies
        direction: Direction to move the snake
        apple_pos: Current position of the apple
        
    Returns:
        Updated snake positions and boolean indicating if apple was eaten
    """
    # Make a copy of the snake positions
    new_positions = snake_positions.copy()
    
    # Get current head position
    head_x, head_y = new_positions[0]
    
    # Calculate new head position based on direction
    if direction == "UP":
        head_y -= 1
    elif direction == "DOWN":
        head_y += 1
    elif direction == "LEFT":
        head_x -= 1
    elif direction == "RIGHT":
        head_x += 1
    
    # Insert new head position
    new_positions.insert(0, [head_x, head_y])
    
    # Check if apple was eaten
    apple_eaten = False
    if head_x == apple_pos[0] and head_y == apple_pos[1]:
        apple_eaten = True
    else:
        # Remove tail if no apple was eaten
        new_positions.pop()
    
    return new_positions, apple_eaten 