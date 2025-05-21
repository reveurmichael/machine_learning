"""
Utility module for snake-specific functions in the Snake game.
Provides helper functions for snake movement and game rules.
"""

def filter_invalid_reversals(moves, current_direction):
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
    last_direction = current_direction
    
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
        x_diff_text = f"the number of RIGHT moves, minus, the number of LEFT moves, should be equal to {x_diff} (= {apple_x} - {head_x})"
    else:
        x_diff = head_x - apple_x
        x_diff_text = f"the number of LEFT moves, minus, the number of RIGHT moves, should be equal to {x_diff} (= {head_x} - {apple_x})"
    
    # Calculate vertical differences
    y_diff_text = ""
    if head_y <= apple_y:
        y_diff = apple_y - head_y
        y_diff_text = f"the number of UP moves, minus, the number of DOWN moves, should be equal to {y_diff} (= {apple_y} - {head_y})"
    else:
        y_diff = head_y - apple_y
        y_diff_text = f"the number of DOWN moves, minus, the number of UP moves, should be equal to {y_diff} (= {head_y} - {apple_y})"
    
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
    from json_utils import extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays
    
    # Store the raw response for display
    game_instance.last_llm_response = response
    
    # Process the response for display
    game_instance.processed_response = processed_response_func(response)
    
    # Clear previous planned moves
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