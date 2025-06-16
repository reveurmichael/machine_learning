"""
LLM communication system.
Core functionality for interacting with language models in the Snake game,
supporting both single-LLM and dual-LLM configurations for gameplay strategy
and response parsing.
"""

import time
import traceback
import re
import json
from datetime import datetime
from colorama import Fore
from utils.file_utils import save_to_file, get_prompt_filename
from llm.prompt_utils import create_parser_prompt
from llm.parsing_utils import parse_and_format


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
            print(
                f"Health check attempt {attempt+1}/{max_retries} for {llm_client.provider} LLM..."
            )
            response = llm_client.generate_response(test_prompt)

            # Check if we got a response that contains the expected text or something reasonable
            if (
                response
                and isinstance(response, str)
                and len(response) > 5
                and "ERROR" not in response
            ):
                print(
                    Fore.GREEN
                    + f"✅ {llm_client.provider}/{llm_client.model} LLM health check passed!"
                )
                return True, response

            print(
                Fore.YELLOW
                + f"⚠️ {llm_client.provider}/{llm_client.model} LLM returned an unusual response: {response}"
            )

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

    return (
        False,
        f"Failed to get a valid response from {llm_client.provider}/{llm_client.model} LLM after {max_retries} attempts",
    )


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

    # Get parser input for metadata
    parser_input = extract_state_for_parser(game_manager)

    # Log the prompt using centralized naming
    prompt_filename = get_prompt_filename(
        game_number=game_manager.game_count+1,
        round_number=game_manager.round_count,
        file_type="prompt"
    )

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
            stats = game_manager.llm_client.last_token_count
            prompt_tokens = stats.get("prompt_tokens")
            completion_tokens = stats.get("completion_tokens")
            game_manager.game.game_state.record_primary_token_stats(prompt_tokens, completion_tokens)

        # Record response time
        response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log the response using centralized naming
        response_filename = get_prompt_filename(
            game_number=game_manager.game_count+1,
            round_number=game_manager.round_count,
            file_type="raw_response"
        )

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
            # Get parser input
            parser_input = extract_state_for_parser(game_manager)

            # Create parser prompt
            parser_prompt = create_parser_prompt(response, *parser_input)

            # Save the secondary LLM prompt using centralized naming
            parser_prompt_filename = get_prompt_filename(
                game_number=game_manager.game_count+1,
                round_number=game_manager.round_count,
                file_type="parser_prompt"
            )

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
                stats = game_manager.llm_client.last_token_count
                prompt_tokens = stats.get("prompt_tokens")
                completion_tokens = stats.get("completion_tokens")
                game_manager.game.game_state.record_secondary_token_stats(prompt_tokens, completion_tokens)

            # Check if we got a valid secondary response
            if secondary_response is None or "ERROR" in secondary_response:
                print(Fore.YELLOW + "⚠️ Secondary LLM returned an error or no response. Falling back to primary LLM.")
                # Fall back to using the primary LLM response
                parser_output = parse_and_format(
                    response,
                    {'game_state': game_manager.game.game_state}
                )
                
                # Use the primary LLM response for display
                from utils.text_utils import process_response_for_display
                game_manager.game.processed_response = process_response_for_display(response)
            else:
                # Record parser response time
                parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save the secondary LLM response using centralized naming
                parsed_response_filename = get_prompt_filename(
                    game_number=game_manager.game_count+1,
                    round_number=game_manager.round_count,
                    file_type="parsed_response"
                )

                parsed_response_metadata = {
                    "SECONDARY LLM Request Time": parser_request_time,
                    "SECONDARY LLM Response Time": parser_response_time,
                    "SECONDARY LLM Provider": game_manager.args.parser_provider,
                    "SECONDARY LLM Model": game_manager.args.parser_model if game_manager.args.parser_provider else None,
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
                    secondary_response,
                    {'game_state': game_manager.game.game_state, 'head_pos': parser_input[0], 'apple_pos': parser_input[1], 'body_cells': parser_input[2]}
                )
                
                # If we couldn't extract a valid parser output, try another approach to find code blocks
                if not parser_output:
                    print("Attempting direct code block extraction from secondary response...")
                    # Try to find any code blocks directly in the response
                    code_block_matches = re.findall(r'```(?:json|javascript|js)?\s*([\s\S]*?)```', secondary_response, re.DOTALL)
                    if code_block_matches:
                        for block in code_block_matches:
                            try:
                                json_data = json.loads(block)
                                if 'moves' in json_data:
                                    print(f"✅ Successfully extracted moves directly: {json_data['moves']}")
                                    parser_output = json_data
                                    break
                            except Exception as e:
                                print(f"Error parsing direct code block: {e}")
                
                # Store the secondary LLM response for display in the UI
                from utils.text_utils import process_response_for_display
                game_manager.game.processed_response = process_response_for_display(secondary_response)
        else:
            # SINGLE LLM MODE: Direct extraction from primary LLM
            parser_output = parse_and_format(
                response,
                {'game_state': game_manager.game.game_state}
            )
            
            # Store the primary LLM response for display in the UI
            from utils.text_utils import process_response_for_display
            game_manager.game.processed_response = process_response_for_display(response)

        # Extract the next move
        next_move = None
        if parser_output and "moves" in parser_output and parser_output["moves"]:
            print(f"✅ Parser output contains moves: {parser_output['moves']}")
            
            # We have valid moves, so we're no longer waiting for a plan
            game_manager.awaiting_plan = False
            
            # Record the plan under the *current* round
            game_manager.current_game_moves.extend(parser_output["moves"])
            
            # Store the full array of moves for the current round via RoundManager
            game_manager.game.game_state.round_manager.record_planned_moves(parser_output["moves"])

            # Set next move but keep the full plan for UI preview – we'll
            # pop it only after executing the move so the first draw shows
            # the complete plan.
            next_move = parser_output["moves"][0] if parser_output["moves"] else None
            game_manager.game.planned_moves = list(parser_output["moves"])  # full copy

            # If we got a valid move, reset the consecutive empty steps counter
            if next_move:
                game_manager.consecutive_empty_steps = 0
                print(Fore.GREEN + f"🐍 Next move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count})")
                
                # For UI display, also log the planned moves
                if len(parser_output["moves"]) > 1:
                    print(Fore.CYAN + f"📝 Planned moves: {', '.join(parser_output['moves'])}")
        else:
            # Log detailed information about what's missing
            if not parser_output:
                print("❌ No parser output received")
            elif "moves" not in parser_output:
                print(f"❌ 'moves' key missing from parser output. Keys: {parser_output.keys()}")
            elif not parser_output["moves"]:
                print("❌ Empty moves list in parser output")
            
            # No valid moves found
            game_manager.consecutive_empty_steps += 1
            print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_consecutive_empty_moves_allowed}")

        # End tracking LLM communication time
        game_manager.game.game_state.record_llm_communication_end()
        
        # Ensure round data is synchronized before returning
        game_manager.game.game_state.round_manager.sync_round_data()

        # Check if we've reached the max consecutive empty moves
        if game_manager.consecutive_empty_steps >= game_manager.args.max_consecutive_empty_moves_allowed:
            print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_consecutive_empty_moves_allowed}). Game over.")
            game_manager.game.game_state.record_game_end("MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")
            return next_move, False

        return next_move, True

    except Exception as e:
        # Record that there was an error in the LLM response BEFORE ending communication time
        game_manager.game.game_state.record_something_is_wrong_move()
        game_manager.something_is_wrong_steps += 1
        
        # Ensure round data is synchronized BEFORE ending communication time
        game_manager.game.game_state.round_manager.sync_round_data()
        
        # End tracking LLM communication time even if there was an error
        game_manager.game.game_state.record_llm_communication_end()
        
        # Increment consecutive errors but DO NOT increment consecutive empty steps
        game_manager.consecutive_something_is_wrong += 1
        print(Fore.RED + f"❌ Error getting response from LLM: {e}")
        print(Fore.YELLOW + f"⚠️ Consecutive LLM errors: {game_manager.consecutive_something_is_wrong}/{game_manager.args.max_consecutive_something_is_wrong_allowed}")
        
        # End game if maximum consecutive errors reached
        if game_manager.consecutive_something_is_wrong >= game_manager.args.max_consecutive_something_is_wrong_allowed:
            print(Fore.RED + f"❌ Maximum consecutive LLM errors reached ({game_manager.args.max_consecutive_something_is_wrong_allowed}). Game over.")
            game_manager.game.game_state.record_game_end("MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED")
            return None, False
        
        traceback.print_exc()
        return None, True 
