"""
Utility module for LLM response processing.
Handles parsing, extracting, and processing responses from language models.
"""

import traceback
from colorama import Fore
from utils.json_utils import extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays

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
            from utils.snake_utils import filter_invalid_reversals
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

def handle_llm_response(response, next_move, error_steps, empty_steps, consecutive_empty_steps, max_empty_moves):
    """Handle the common logic for LLM response processing.
    
    Args:
        response: The LLM response text
        next_move: The parsed next move (or None)
        error_steps: Count of error steps
        empty_steps: Count of empty steps
        consecutive_empty_steps: Count of consecutive empty steps
        max_empty_moves: Maximum allowed consecutive empty moves
        
    Returns:
        Tuple of (error_steps, empty_steps, consecutive_empty_steps, game_active)
    """
    game_active = True
    
    # Check for empty moves with ERROR in reasoning
    if not next_move and "ERROR" in response:
        error_steps += 1
        consecutive_empty_steps = 0  # Reset consecutive empty steps if ERROR occurs
        print(Fore.YELLOW + f"⚠️ ERROR in LLM response. Continuing with next round.")
    elif not next_move:
        empty_steps += 1
        consecutive_empty_steps += 1
        print(Fore.YELLOW + f"⚠️ Empty move (consecutive: {consecutive_empty_steps})")
        # Check if we've reached max consecutive empty moves
        if consecutive_empty_steps >= max_empty_moves:
            print(Fore.RED + f"❌ Game over! {max_empty_moves} consecutive empty moves without ERROR.")
            game_active = False
    else:
        consecutive_empty_steps = 0  # Reset on valid move
        
    return error_steps, empty_steps, consecutive_empty_steps, game_active 