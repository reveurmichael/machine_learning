"""
Response parsing system for language models.
Functions for extracting structured data from LLM responses, handling both 
primary and secondary LLM outputs in the Snake game context.
"""

from utils.json_utils import extract_valid_json, extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays
from utils.moves_utils import normalize_directions


def parse_and_format(llm_client, llm_response, parser_options=None):
    """Parse an LLM response and format it for use by the game.
    
    Core parsing function that extracts structured move data from LLM responses.
    Works with both primary and secondary LLM responses.
    
    Args:
        llm_client: LLM client used to generate the response
        llm_response: Raw response from the LLM
        parser_options: Dictionary containing optional parsing parameters:
            - game_state: GameData instance to track statistics
            - head_pos: Current head position [x, y]
            - apple_pos: Current apple position [x, y]
            - body_cells: List of body cell positions [[x1, y1], [x2, y2], ...]
        
    Returns:
        Dictionary with parsed data or None if parsing failed
    """
    try:
        game_state = None
        if parser_options and 'game_state' in parser_options:
            game_state = parser_options['game_state']
        
        # Parse JSON directly from response
        parsed_data = extract_valid_json(llm_response, game_state, attempt_id=0)
        
        if parsed_data and "moves" in parsed_data:
            # Record success and return data if it includes a 'moves' field
            parsed_data["moves"] = normalize_directions(parsed_data["moves"])
            return parsed_data
        return None
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None

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
            print("Attempting to extract JSON from LLM response...")

            # Try to extract JSON from the response
            json_data = extract_valid_json(response, game_instance.game_state, attempt_id=0)

            # If we couldn't extract JSON, try looking for code blocks
            if not json_data:
                print("Direct JSON extraction failed, trying code block extraction...")
                # First try to extract from a code block
                json_data = extract_json_from_code_block(response)

                # If still no JSON, try to extract using regex
                if not json_data:
                    print("Code block extraction failed, trying text extraction...")
                    json_data = extract_json_from_text(response)

                    # If still no JSON, try to extract move arrays
                    if not json_data:
                        print("Text extraction failed, trying move arrays extraction...")
                        json_data = extract_moves_from_arrays(response)

            # If we have JSON data, get the moves
            if json_data and 'moves' in json_data:
                print(f"Successfully extracted moves: {json_data['moves']}")
                moves = json_data.get('moves', [])

                # Store the reasoning for display
                reasoning = json_data.get('reasoning', '')
                if reasoning:
                    game_instance.processed_response = reasoning

                # If we have moves, return the first one and store the rest
                if moves and len(moves) > 0:
                    print(f"First move: {moves[0]}, Planned moves: {moves[1:] if len(moves) > 1 else []}")

                    # Get the current direction to check for reversals
                    current_direction = None
                    if hasattr(game_instance, 'get_current_direction_key'):
                        current_direction = game_instance.get_current_direction_key()
                    elif game_instance.current_direction is not None:
                        # Get direction using current_direction attribute
                        current_direction = game_instance._get_current_direction_key()

                    # Filter out any invalid reversals in the planned moves
                    filtered_moves = game_instance.filter_invalid_reversals(moves[1:], current_direction)
                    if len(filtered_moves) < len(moves[1:]):
                        print(f"Filtered out {len(moves[1:]) - len(filtered_moves)} invalid reversal moves")

                    game_instance.planned_moves = filtered_moves

                    # Return the first move
                    return moves[0]
                else:
                    print("Moves list is empty")
            elif json_data:
                print(f"JSON data found but no 'moves' key. Keys: {json_data.keys()}")
            else:
                print("No valid JSON data found")

            # If we didn't get moves, check if reasoning contains "SOMETHING_IS_WRONG"
            if (
                json_data
                and "reasoning" in json_data
                and "SOMETHING_IS_WRONG" in json_data["reasoning"]
            ):
                # This is an explicit error state from the LLM
                print("LLM reported an SOMETHING_IS_WRONG state")
                game_instance.game_state.record_something_is_wrong_move()
                return None

            # No valid moves, return None (empty move)
            return None

        except Exception as e:
            print(f"Error parsing JSON from LLM response: {e}")

            # Record error in game state
            game_instance.game_state.record_something_is_wrong_move()

            return None
    except Exception as e:
        print(f"Error processing LLM response: {e}")

        # Record error in game state
        game_instance.game_state.record_something_is_wrong_move()

        return None

def handle_llm_response(llm_client, response, parser_input=None, game_state=None):
    """Handle a response from an LLM, extracting and processing moves.
    
    Args:
        llm_client: LLM client used to generate the response
        response: Raw response from the LLM
        parser_input: Optional tuple of (head_pos, apple_pos, body_cells) for parsing
        game_state: Optional GameData instance to track statistics
        
    Returns:
        Parsed JSON data with moves, or None if parsing failed
    """
    # Unpack parser input if provided
    head_pos, apple_pos, body_cells = parser_input if parser_input else (None, None, None)
    
    # Parse and format the response
    return parse_and_format(llm_client, response, {'game_state': game_state, 'head_pos': head_pos, 'apple_pos': apple_pos, 'body_cells': body_cells}) 
