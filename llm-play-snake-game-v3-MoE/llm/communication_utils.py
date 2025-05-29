"""
LLM communication system.
Core functionality for interacting with language models in the Snake game,
supporting both single-LLM and dual-LLM configurations for gameplay strategy
and response parsing.
"""

import time
import traceback
from datetime import datetime
from colorama import Fore
from utils.file_utils import save_to_file
from llm.prompt_utils import create_parser_prompt
from llm.parsing_utils import parse_and_format
from llm.health_utils import check_llm_health

def extract_state_for_parser(game_manager):
    """Extract state information for the parser.
    
    Args:
        game_manager: The GameManager instance
        
    Returns:
        Tuple of (head_pos, apple_pos, body_cells) as strings
    """
    # Get the game state
    head_x, head_y = game_manager.game.head
    apple_x, apple_y = game_manager.game.apple
    body_cells = game_manager.game.body
    
    # Format for parser
    head_pos = f"({head_x}, {head_y})"
    apple_pos = f"({apple_x}, {apple_y})"
    body_cells_str = str(body_cells) if body_cells else "[]"
    
    return head_pos, apple_pos, body_cells_str

def get_llm_response(game_manager):
    """Get a response from the LLM system based on the current game state.
    
    This function handles both single-LLM and dual-LLM configurations:
    - In single-LLM mode: Uses only the primary LLM for generating and parsing moves
    - In dual-LLM mode: Uses the primary LLM for strategy and the secondary LLM for parsing
    
    Args:
        game_manager: The GameManager instance
        
    Returns:
        Tuple of (next_move, game_active)
    """
    # Start tracking LLM communication time
    game_manager.game.game_state.record_llm_communication_start()

    # Get game state
    game_state = game_manager.game.get_state_representation()

    # Format prompt for LLM
    prompt = game_state

    # Log the prompt
    prompt_filename = f"game_{game_manager.game_count+1}_round_{game_manager.round_count}_prompt.txt"

    # Get parser input for metadata
    parser_input = extract_state_for_parser(game_manager)

    prompt_metadata = {
        "PRIMARY LLM Provider": game_manager.args.provider,
        "PRIMARY LLM Model": game_manager.args.model or "default",
        "Head Position": parser_input[0],
        "Apple Position": parser_input[1],
        "Body Cells": parser_input[2],
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
        start_time = time.time()
        response = game_manager.llm_client.generate_response(prompt, **kwargs)
        primary_response_time = time.time() - start_time

        # Record response time in game state
        game_manager.game.game_state.record_primary_response_time(primary_response_time)

        # Record token usage if available
        if hasattr(game_manager.llm_client, 'last_token_count') and game_manager.llm_client.last_token_count:
            prompt_tokens = game_manager.llm_client.last_token_count.get('prompt_tokens', 0)
            completion_tokens = game_manager.llm_client.last_token_count.get('completion_tokens', 0)
            game_manager.game.game_state.record_primary_token_stats(prompt_tokens, completion_tokens)

        # Record response time
        response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log the response
        response_filename = f"game_{game_manager.game_count+1}_round_{game_manager.round_count}_raw_response.txt"

        # Get parser input to include in metadata
        parser_input = extract_state_for_parser(game_manager)

        response_metadata = {
            "PRIMARY LLM Request Time": request_time,
            "PRIMARY LLM Response Time": response_time,
            "PRIMARY LLM Provider": game_manager.args.provider,
            "PRIMARY LLM Model": game_manager.args.model or "default",
            "Head Position": parser_input[0],
            "Apple Position": parser_input[1],
            "Body Cells": parser_input[2],
        }
        response_path = save_to_file(response, game_manager.responses_dir, response_filename, metadata=response_metadata)
        print(Fore.GREEN + f"📝 Response saved to {response_path}")

        # Parse the response - dual or single LLM mode based on configuration
        parser_output = None
        if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
            # DUAL LLM MODE: Using a secondary LLM for parsing
            # Track the previous parser usage count to detect if it gets used
            game_manager.previous_parser_usage = game_manager.game.game_state.parser_usage_count

            # Get parser input
            parser_input = extract_state_for_parser(game_manager)

            # Create parser prompt
            parser_prompt = create_parser_prompt(response, *parser_input)

            # Save the secondary LLM prompt
            parser_prompt_filename = f"game_{game_manager.game_count+1}_round_{game_manager.round_count}_parser_prompt.txt"
            parser_prompt_metadata = {
                "SECONDARY LLM Provider": game_manager.args.parser_provider,
                "SECONDARY LLM Model": game_manager.args.parser_model or "default",
                "Head Position": parser_input[0],
                "Apple Position": parser_input[1],
                "Body Cells": parser_input[2],
            }
            parser_prompt_path = save_to_file(parser_prompt, game_manager.prompts_dir, parser_prompt_filename, metadata=parser_prompt_metadata)
            print(Fore.GREEN + f"📝 Parser prompt saved to {parser_prompt_path}")

            # Record parser request time
            parser_request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Get response from secondary LLM
            start_time = time.time()
            secondary_response = game_manager.llm_client.generate_text_with_secondary_llm(parser_prompt)
            secondary_response_time = time.time() - start_time

            # Record secondary response time in game state
            game_manager.game.game_state.record_secondary_response_time(secondary_response_time)

            # Record secondary token usage if available
            if hasattr(game_manager.llm_client, 'last_token_count') and game_manager.llm_client.last_token_count:
                prompt_tokens = game_manager.llm_client.last_token_count.get('prompt_tokens', 0)
                completion_tokens = game_manager.llm_client.last_token_count.get('completion_tokens', 0)
                game_manager.game.game_state.record_secondary_token_stats(prompt_tokens, completion_tokens)

            # Check if we got a valid secondary response
            if secondary_response is None or "ERROR" in secondary_response:
                print(Fore.YELLOW + "⚠️ Secondary LLM returned an error or no response. Falling back to primary LLM.")
                # Fall back to using the primary LLM response
                parser_output = parse_and_format(
                    game_manager.llm_client,
                    response,
                    {'game_state': game_manager.game.game_state}
                )
                
                # Use the primary LLM response for display
                from utils.text_utils import process_response_for_display
                game_manager.game.processed_response = process_response_for_display(response)
            else:
                # Record parser response time
                parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save the secondary LLM response
                parsed_response_filename = f"game_{game_manager.game_count+1}_round_{game_manager.round_count}_parsed_response.txt"
                parsed_response_metadata = {
                    "SECONDARY LLM Request Time": parser_request_time,
                    "SECONDARY LLM Response Time": parser_response_time,
                    "SECONDARY LLM Provider": game_manager.args.parser_provider,
                    "SECONDARY LLM Model": game_manager.args.parser_model or "default",
                    "Head Position": parser_input[0],
                    "Apple Position": parser_input[1],
                    "Body Cells": parser_input[2]
                }
                parsed_path = save_to_file(
                    secondary_response,
                    game_manager.responses_dir,
                    parsed_response_filename,
                    metadata=parsed_response_metadata
                )
                print(Fore.GREEN + f"📝 Parsed response saved to {parsed_path}")

                # Process the secondary LLM response
                parser_output = parse_and_format(
                    game_manager.llm_client,
                    response,
                    {'game_state': game_manager.game.game_state, 'head_pos': parser_input[0], 'apple_pos': parser_input[1], 'body_cells': parser_input[2]}
                )
                
                # Store the secondary LLM response for display in the UI
                from utils.text_utils import process_response_for_display
                game_manager.game.processed_response = process_response_for_display(secondary_response)

                # Track parser usage statistics
                if game_manager.game.game_state.parser_usage_count > game_manager.previous_parser_usage:
                    game_manager.parser_usage_count += 1
                    print(Fore.GREEN + f"🔍 Using parsed output (Parser usage: {game_manager.parser_usage_count})")
        else:
            # SINGLE LLM MODE: Direct extraction from primary LLM
            parser_output = parse_and_format(
                game_manager.llm_client,
                response,
                {'game_state': game_manager.game.game_state}
            )
            
            # Store the primary LLM response for display in the UI
            from utils.text_utils import process_response_for_display
            game_manager.game.processed_response = process_response_for_display(response)

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
                print(Fore.GREEN + f"🐍 Next move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count})")
                
                # For UI display, also log the planned moves
                if len(parser_output["moves"]) > 1:
                    print(Fore.CYAN + f"📝 Planned moves: {', '.join(parser_output['moves'][1:])}")
                
                # Increment round_count after getting a plan from the LLM
                game_manager.round_count += 1
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