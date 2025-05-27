"""
Utility module for LLM response processing, prompt preparation, and interaction.
Handles parsing, extracting, and processing responses from language models,
as well as interactions between the game and LLMs.
"""

import json
import traceback
import time
import os
from colorama import Fore
from config import PROMPT_TEMPLATE_TEXT_PRIMARY_LLM, PROMPT_TEMPLATE_TEXT_SECONDARY_LLM
from utils.json_utils import extract_valid_json, extract_json_from_code_block, extract_json_from_text, extract_moves_from_arrays, validate_json_format
from utils.file_utils import save_to_file
from utils.log_utils import format_parsed_llm_response
from datetime import datetime

def format_raw_llm_response(response, request_time, response_time, model_name=None, provider=None):
    """Format the raw LLM response with metadata for saving to a file.
    
    Args:
        response: The raw response from the LLM
        request_time: Timestamp when the request was sent
        response_time: Duration of the response in seconds
        model_name: Name of the model used (optional)
        provider: Name of the provider used (optional)
        
    Returns:
        Formatted response with metadata
    """
    # Format the metadata section
    metadata = [
        f"Time: {request_time}",
        f"Response time: {response_time:.2f} seconds",
    ]
    
    if model_name:
        metadata.append(f"Model: {model_name}")
    
    if provider:
        metadata.append(f"Provider: {provider}")
    
    # Format the full response
    formatted_response = "=== METADATA ===\n"
    formatted_response += "\n".join(metadata)
    formatted_response += "\n\n=== RESPONSE ===\n"
    formatted_response += response
    
    return formatted_response

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

def parse_and_format(llm_client, llm_response, game_state=None, head_pos=None, apple_pos=None, body_cells=None):
    """Parse the output from the primary LLM and convert it to the required JSON format.
    
    Args:
        llm_client: The LLM client to use for secondary LLM
        llm_response: The raw response from the primary LLM
        game_state: Optional game state object
        head_pos: Optional head position string in format "(x, y)"
        apple_pos: Optional apple position string in format "(x, y)"
        body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
        
    Returns:
        Properly formatted JSON response as a dictionary
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
    
    # If game_state is provided, record parser usage
    if game_state:
        game_state.record_parser_usage()
    
    # Extract JSON from the secondary LLM's response
    json_data = extract_valid_json(formatted_response)
    
    # If we don't have valid JSON from the secondary LLM, create a fallback response
    if not json_data or not validate_json_format(json_data):
        print("Warning: Secondary LLM failed to generate valid JSON, using fallback")
        fallback_data = {
            "moves": [],
            "reasoning": "ERROR: Could not generate valid moves from LLM response"
        }
        return fallback_data
        
    return json_data

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
                print(Fore.GREEN + f"✅ {llm_client.provider}/{llm_client.model} LLM health check passed!")
                return True, response
            else:
                print(Fore.YELLOW + f"⚠️ {llm_client.provider}/{llm_client.model} LLM returned an unusual response: {response}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        except Exception as e:
            error_msg = f"Error connecting to {llm_client.provider}/{llm_client.model} LLM: {str(e)}"
            print(Fore.RED + f"❌ {error_msg}")
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    return False, f"Failed to get a valid response from {llm_client.provider}/{llm_client.model} LLM after {max_retries} attempts"

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

def handle_llm_response(llm_client, response, parser_input=None, game_state=None):
    """Handle LLM response and process it for game use.
    
    Args:
        llm_client: The LLM client
        response: Text response from the LLM
        parser_input: Tuple of (head_pos, apple_pos, body_cells) for parser
        game_state: The game state object for recording stats
        
    Returns:
        Parsed output from the LLM
    """
    # Unpack parser input if provided
    head_pos, apple_pos, body_cells = parser_input if parser_input else (None, None, None)
    
    # Parse and format the response
    return parse_and_format(llm_client, response, game_state, head_pos, apple_pos, body_cells)

def get_llm_response(game_manager):
    """Get a response from the LLM based on the current game state.
    
    Args:
        game_manager: The GameManager instance
        
    Returns:
        Tuple of (next_move, game_active)
    """
    from datetime import datetime
    
    # Start tracking LLM communication time
    game_manager.game.game_state.record_llm_communication_start()
    
    # Get game state
    game_state = game_manager.game.get_state_representation()
    
    # Format prompt for LLM
    prompt = game_state
    
    # Log the prompt with updated filename format
    prompt_filename = f"game_{game_manager.game_count+1}_round{game_manager.round_count+1}_prompt.txt"
    prompt_metadata = {
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Game": game_manager.game_count+1,
        "Round": game_manager.round_count+1,
        "PRIMARY LLM Model": game_manager.args.model or "default",
        "PRIMARY LLM Provider": game_manager.args.provider
    }
    prompt_path = save_to_file(prompt, game_manager.prompts_dir, prompt_filename, metadata=prompt_metadata)
    print(Fore.GREEN + f"📝 Prompt saved to {prompt_path}")
    
    # Get next move from first LLM
    kwargs = {}
    if game_manager.args.model:
        kwargs['model'] = game_manager.args.model
        
    try:
        # Record request time
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get response from primary LLM
        response = game_manager.llm_client.generate_response(prompt, **kwargs)
        
        # Record response time
        response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log the response with updated filename format
        response_filename = f"game_{game_manager.game_count+1}_round{game_manager.round_count+1}_response.txt"
        response_metadata = {
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Request Time": request_time,
            "Response Time": response_time,
            "PRIMARY LLM Model": game_manager.args.model or "default",
            "PRIMARY LLM Provider": game_manager.args.provider,
            "SECONDARY LLM": "Not used" if not game_manager.args.parser_provider or game_manager.args.parser_provider.lower() == "none" else game_manager.args.parser_provider,
            "Response Format": "PRIMARY LLM RESPONSE (GAME STRATEGY)"
        }
        response_path = save_to_file(response, game_manager.responses_dir, response_filename, metadata=response_metadata)
        print(Fore.GREEN + f"📝 Response saved to {response_path}")
        
        # Parse the response
        parser_output = None
        if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
            # Track the previous parser usage count to detect if it gets used
            game_manager.previous_parser_usage = game_manager.game.game_state.parser_usage_count
            
            # Get parser input
            from utils.game_manager_utils import extract_state_for_parser
            parser_input = extract_state_for_parser(game_manager)
            
            # Create parser prompt
            parser_prompt = create_parser_prompt(response, *parser_input)
            
            # Save the secondary LLM prompt with updated filename format
            parser_prompt_filename = f"game_{game_manager.game_count+1}_round{game_manager.round_count+1}_parser_prompt.txt"
            parser_prompt_metadata = {
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Game": game_manager.game_count+1,
                "Round": game_manager.round_count+1,
                "SECONDARY LLM Model": game_manager.args.parser_model or "default",
                "SECONDARY LLM Provider": game_manager.args.parser_provider,
                "Head Position": parser_input[0],
                "Apple Position": parser_input[1],
                "Body Cells": parser_input[2]
            }
            parser_prompt_path = save_to_file(parser_prompt, game_manager.prompts_dir, parser_prompt_filename, metadata=parser_prompt_metadata)
            print(Fore.GREEN + f"📝 Parser prompt saved to {parser_prompt_path}")
            
            # Record parser request time
            parser_request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get next move using the parser
            parser_output = handle_llm_response(
                game_manager.llm_client,
                response,
                parser_input,
                game_manager.game.game_state
            )
            
            # Record parser response time
            parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if parser was used (parser usage count increased)
            if game_manager.game.game_state.parser_usage_count > game_manager.previous_parser_usage:
                game_manager.parser_usage_count += 1
                print(Fore.GREEN + f"🔍 Using parsed output (Parser usage: {game_manager.parser_usage_count})")
                
                # Format and save the parsed response
                parsed_response_filename = f"game_{game_manager.game_count+1}_round{game_manager.round_count+1}_parsed_response.txt"
                parsed_response_metadata = {
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Secondary LLM Request Time": parser_request_time,
                    "Secondary LLM Response Time": parser_response_time,
                    "SECONDARY LLM Model": game_manager.args.parser_model or "default",
                    "SECONDARY LLM Provider": game_manager.args.parser_provider,
                    "Response Format": "SECONDARY LLM RESPONSE (FORMATTED JSON)"
                }
                parsed_path = save_to_file(
                    format_parsed_response(parser_output),
                    game_manager.responses_dir,
                    parsed_response_filename,
                    metadata=parsed_response_metadata
                )
                print(Fore.GREEN + f"📝 Parsed response saved to {parsed_path}")
        else:
            # Direct extraction from primary LLM
            parser_output = parse_and_format(
                game_manager.llm_client,
                response,
                game_manager.game.game_state
            )
        
        # Extract the next move
        next_move = None
        if parser_output and "moves" in parser_output and parser_output["moves"]:
            # Record the move
            game_manager.current_game_moves.extend(parser_output["moves"])
            
            # Set the next move
            next_move = parser_output["moves"][0] if parser_output["moves"] else None
            game_manager.game.planned_moves = parser_output["moves"][1:] if len(parser_output["moves"]) > 1 else []
            
            # If we got a valid move, reset the consecutive empty steps counter
            if next_move:
                game_manager.consecutive_empty_steps = 0
                print(Fore.GREEN + f"🐍 Next move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count+1})")
            else:
                game_manager.consecutive_empty_steps += 1
                print(Fore.YELLOW + f"⚠️ No valid move extracted. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_empty_moves}")
        else:
            # No valid moves found
            game_manager.consecutive_empty_steps += 1
            print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_empty_moves}")
        
        # End tracking LLM communication time
        game_manager.game.game_state.record_llm_communication_end()
        
        # Check if we've reached the max consecutive empty moves
        if game_manager.consecutive_empty_steps >= game_manager.args.max_empty_moves:
            print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_empty_moves}). Game over.")
            game_manager.game.game_state.record_game_end("EMPTY_MOVES")
            return next_move, False
            
        return next_move, True
        
    except Exception as e:
        # End tracking LLM communication time even if there was an error
        game_manager.game.game_state.record_llm_communication_end()
        
        print(Fore.RED + f"❌ Error getting response from LLM: {e}")
        traceback.print_exc()
        return None, False

def format_parsed_response(parser_output):
    """Format the parsed response for saving to file.
    
    Args:
        parser_output: The output from the parser
        
    Returns:
        Formatted string representation of the parsed output
    """
    if not parser_output:
        return "No valid moves found."
        
    formatted = "Parsed Output:\n"
    
    if "moves" in parser_output:
        formatted += f"Moves: {parser_output['moves']}\n"
    
    if "reasoning" in parser_output:
        formatted += f"\nReasoning:\n{parser_output['reasoning']}\n"
        
    return formatted 