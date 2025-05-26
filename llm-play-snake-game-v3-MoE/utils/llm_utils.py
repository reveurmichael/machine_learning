"""
Utility module for LLM response processing.
Handles parsing, extracting, and processing responses from language models.
"""

import traceback
from colorama import Fore
from utils.json_utils import extract_valid_json
from utils.snake_utils import filter_invalid_reversals

def parse_llm_response(response_text, current_direction_key, process_response_for_display_func):
    """Parse the LLM's response to extract multiple sequential moves.
    
    Args:
        response_text: Text response from the LLM.
        current_direction_key: The current direction of the snake to filter reversals.
        process_response_for_display_func: Function to process the raw response for display.
        
    Returns:
        A tuple (planned_moves, processed_display_response_text, error_message).
        planned_moves: A list of extracted and filtered moves.
        processed_display_response_text: The response formatted for display.
        error_message: An error message string if parsing failed, else None.
    """
    planned_moves = []
    error_message = None
    
    # Process the response for display first
    processed_display_response_text = process_response_for_display_func(response_text)
    
    try:
        print(f"Parsing LLM response: '{response_text[:50]}...'")
        
        # Try to extract valid JSON data
        json_data = extract_valid_json(response_text)
        
        # Extract moves from JSON if found
        if json_data and "moves" in json_data and isinstance(json_data["moves"], list):
            valid_moves = [move.upper() for move in json_data["moves"] 
                         if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            if valid_moves:
                planned_moves = valid_moves
                print(f"Found {len(planned_moves)} moves in JSON: {planned_moves}")
        
        if not planned_moves:
            print("No valid directions found in response.")
        else:
            # Filter out invalid reversal moves
            planned_moves = filter_invalid_reversals(planned_moves, current_direction_key)
            if not planned_moves:
                print("All extracted moves were invalid reversals.")
                error_message = "All extracted moves were invalid reversals."
    
    except Exception as e:
        error_message = f"Error parsing LLM response: {str(e)}"
        print(f"{Fore.RED}{error_message}")
        traceback.print_exc()
    
    return planned_moves, processed_display_response_text, error_message

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