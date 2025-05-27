"""
Utility module for LLM response processing and prompt preparation.
Handles parsing, extracting, and processing responses from language models.
"""

import json
import traceback
import time
from colorama import Fore
from config import PROMPT_TEMPLATE_TEXT_PRIMARY_LLM, PROMPT_TEMPLATE_TEXT_SECONDARY_LLM
from utils.json_utils import extract_valid_json, extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays, validate_json_format

def prepare_snake_prompt(head_position, body_positions, apple_position, current_direction):
    """Prepare a prompt for the primary LLM to determine the next snake move.
    
    Args:
        head_position: [x, y] position of the snake's head
        body_positions: List of [x, y] positions for the snake's body
        apple_position: [x, y] position of the apple
        current_direction: Current direction of movement
        
    Returns:
        Formatted prompt string
    """
    # Get head position in (x, y) format for prompt
    head_x, head_y = head_position
    head_pos = f"({head_x},{head_y})"
    
    # Get current direction string
    direction_str = current_direction if current_direction else "NONE"
    
    # Format body cells
    body_cells_str = format_body_cells_str(body_positions)
    
    # Get apple position
    apple_x, apple_y = apple_position
    apple_pos = f"({apple_x},{apple_y})"
    
    # Calculate the expected move differences
    # Use the function now in game_logic but imported here since we need it for prompt preparation
    from core.game_logic import GameLogic
    game_logic = GameLogic(use_gui=False)  # Create a temporary instance
    move_differences = game_logic.calculate_move_differences(head_position, apple_position)
    
    # Create a prompt from the template text using string replacements
    prompt = PROMPT_TEMPLATE_TEXT_PRIMARY_LLM
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_CURRENT_DIRECTION", direction_str)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells_str)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_ON_THE_TOPIC_OF_MOVES_DIFFERENCE", move_differences)
    
    return prompt

def format_body_cells_str(body_positions):
    """Format the snake body cells as a string representation.
    
    Args:
        body_positions: List of [x, y] coordinates of the snake segments
        
    Returns:
        String representation of body cells in format: "[(x1,y1), (x2,y2), ...]"
    """
    body_cells = []
    
    # Format each position as a tuple string
    for x, y in body_positions:
        body_cells.append(f"({x},{y})")
        
    return "[" + ", ".join(body_cells) + "]"

def parse_and_format(llm_client, llm_response, head_pos=None, apple_pos=None, body_cells=None):
    """Parse the output from the primary LLM and convert it to the required JSON format.
    
    Args:
        llm_client: The LLM client to use for secondary LLM
        llm_response: The raw response from the primary LLM
        head_pos: Optional head position string in format "(x, y)"
        apple_pos: Optional apple position string in format "(x, y)"
        body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
        
    Returns:
        A tuple containing (formatted_response, parser_prompt) where:
          - formatted_response: The properly formatted JSON response
          - parser_prompt: The prompt that was sent to the secondary LLM
    """
    # Check if the primary LLM response contains valid JSON (for logging purposes only)
    first_json_valid = False
    json_data = extract_valid_json(llm_response)
    if json_data and validate_json_format(json_data):
        first_json_valid = True
        print("Primary LLM response contains valid JSON, but will still use secondary LLM for consistency")
    
    # Create the prompt for the secondary LLM
    parser_prompt = create_parser_prompt(llm_response, head_pos, apple_pos, body_cells)
    
    # Get response from the secondary LLM
    print("Using secondary LLM to parse and format response")
    formatted_response = llm_client.generate_text_with_secondary_llm(parser_prompt)
    
    # Extract JSON from the secondary LLM's response
    json_data = extract_valid_json(formatted_response)
    
    # If we don't have valid JSON from the secondary LLM, create a fallback response
    if not json_data or not validate_json_format(json_data):
        print("Warning: Secondary LLM failed to generate valid JSON, using fallback")
        fallback_data = {
            "moves": [],
            "reasoning": "ERROR: Could not generate valid moves from LLM response"
        }
        return json.dumps(fallback_data), parser_prompt
        
    return json.dumps(json_data), parser_prompt

def create_parser_prompt(llm_response, head_pos=None, apple_pos=None, body_cells=None):
    """Create a prompt for the secondary LLM to parse the output of the primary LLM.
    
    Args:
        llm_response: The raw response from the primary LLM
        head_pos: Optional head position string in format "(x, y)"
        apple_pos: Optional apple position string in format "(x, y)"
        body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
        
    Returns:
        Prompt for the secondary LLM
    """
    # Use string replacement for the prompt template
    parser_prompt = PROMPT_TEMPLATE_TEXT_SECONDARY_LLM.replace("TEXT_TO_BE_REPLACED_FIRST_LLM_RESPONSE", llm_response)
    
    # Replace head and apple position placeholders if provided
    if head_pos:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
    if apple_pos:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
    if body_cells:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells)
        
    return parser_prompt

def check_llm_health(llm_client, max_retries=2, retry_delay=2):
    """Check if the LLM is accessible and responding by sending a simple test query.
    
    Args:
        llm_client: The LLM client to check
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        A tuple containing (is_healthy, response) where:
          - is_healthy: Boolean indicating if the LLM is healthy
          - response: The response from the LLM or an error message
    """
    test_prompt = "Hello, are you there? Please respond with 'Yes, I am here.'"
    
    for attempt in range(max_retries):
        try:
            print(f"Health check attempt {attempt+1}/{max_retries} for {llm_client.provider} LLM...")
            response = llm_client.generate_response(test_prompt)
            
            # Check if we got a response that contains the expected text or something reasonable
            if response and isinstance(response, str) and len(response) > 5 and "ERROR" not in response:
                print(Fore.GREEN + f"✅ {llm_client.provider} LLM health check passed!")
                return True, response
            else:
                print(Fore.YELLOW + f"⚠️ {llm_client.provider} LLM returned an unusual response: {response}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        except Exception as e:
            error_msg = f"Error connecting to {llm_client.provider} LLM: {str(e)}"
            print(Fore.RED + f"❌ {error_msg}")
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    return False, f"Failed to get a valid response from {llm_client.provider} LLM after {max_retries} attempts"

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
        
        # If response is empty or contains an error
        if not response or "ERROR" in response.upper():
            print("Error in LLM response or empty response")
            return None
        
        # Parse JSON from the response
        try:
            # Try to extract JSON from the response
            json_data = extract_valid_json(response)
            
            # If we couldn't extract JSON, try looking for code blocks
            if not json_data:
                # First try to extract from a code block
                json_data = extract_json_from_code_block(response)
                
                # If still no JSON, try to extract using regex
                if not json_data:
                    json_data = extract_json_from_text(response)
                    
                    # If still no JSON, try to extract move arrays
                    if not json_data:
                        json_data = extract_moves_from_arrays(response)
            
            # If we have JSON data, get the moves
            if json_data and 'moves' in json_data:
                moves = json_data.get('moves', [])
                
                # Store the reasoning for display
                reasoning = json_data.get('reasoning', '')
                if reasoning:
                    game_instance.processed_response = reasoning
                
                # If we have moves, return the first one and store the rest
                if moves and len(moves) > 0:
                    # Get the current direction to check for reversals
                    current_direction = game_instance._get_current_direction_key() if game_instance.current_direction is not None else None
                    
                    # Filter out any invalid reversals in the planned moves
                    # Use the filter_invalid_reversals function from game_controller.py
                    game_instance.planned_moves = game_instance.filter_invalid_reversals(moves[1:], current_direction)
                    
                    # Return the first move
                    return moves[0]
            
            # If we didn't get moves, check if reasoning contains "ERROR"
            if json_data and 'reasoning' in json_data and "ERROR" in json_data['reasoning']:
                # This is an explicit error state from the LLM
                print("LLM reported an error state")
                game_instance.game_state.record_error_move()
                return None
                
            # No valid moves, return None (empty move)
            return None
            
        except Exception as e:
            print(f"Error parsing JSON from LLM response: {e}")
            traceback.print_exc()
            
            # Record error in game state
            game_instance.game_state.record_error_move()
            
            return None
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        traceback.print_exc()
        
        # Record error in game state
        game_instance.game_state.record_error_move()
        
        return None

def handle_llm_response(response, next_move, error_steps, empty_steps, consecutive_empty_steps, max_empty_moves):
    """Handle LLM response and update game statistics.
    
    Args:
        response: Text response from the LLM
        next_move: The next move to make (or None)
        error_steps: Count of error steps
        empty_steps: Count of empty steps
        consecutive_empty_steps: Count of consecutive empty steps
        max_empty_moves: Maximum allowed consecutive empty moves
        
    Returns:
        Tuple of (error_steps, empty_steps, consecutive_empty_steps, game_active)
    """
    # Default to game continuing
    game_active = True
    
    # If response contains an error marker, reset consecutive empty steps
    if response and "ERROR" in response.upper():
        consecutive_empty_steps = 0
        error_steps += 1
        print(f"Error in LLM response. Error steps: {error_steps}")
    
    # If no valid move, count as empty step
    if next_move is None:
        empty_steps += 1
        consecutive_empty_steps += 1
        print(f"No valid move. Empty steps: {empty_steps}, Consecutive: {consecutive_empty_steps}")
        
        # Check if we've reached maximum consecutive empty moves
        if consecutive_empty_steps >= max_empty_moves:
            print(f"Game over! Maximum consecutive empty moves ({max_empty_moves}) reached.")
            game_active = False
    else:
        # Reset consecutive empty steps on valid move
        consecutive_empty_steps = 0
    
    return error_steps, empty_steps, consecutive_empty_steps, game_active 