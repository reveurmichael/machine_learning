"""
LLM communication sub-system (Task-0).

This module is deliberately **narrow in scope**: given a *prompt* and some
lightweight contextual *metadata*, talk to the primary and (optionally)
secondary LLM, parse/format the answer, and hand back the next move plus a
continuation flag.

Historically the helper was *implicitly* tied to the game-flow concept of a
"round" – it reached into :class:`core.game_manager.GameManager` and
dereferenced ``round_count`` for log-file names and statistics.  That created
a hidden dependency: any future task that calls this function had to own a
``round_count`` attribute, even if its pacing scheme was *not* organised in
rounds (e.g. heuristic agents, RL curriculum epochs).

As part of the SOLID refactor the coupling was removed:

*   **The caller now passes the round explicitely** via the ``round_id`` kwarg
    of :func:`get_llm_response`.  If that argument is ``None`` we fall back to
    ``manager.round_count`` to preserve 100-% backward compatibility for
    Task-0.  All file names and metadata strings derive *solely* from the
    value that enters the function – no more global reach-through.

*   All other responsibilities (prompt construction, round management, timing
    of when to call the LLM, etc.) stay in higher-level orchestration classes
    such as :class:`core.game_loop.GameLoop` or the concrete
    :class:`llm.agent_llm.LLMSnakeAgent`.

Consequences
+------------
• Task-0 (round-based planning) behaves exactly as before.
• Future tasks can call the helper with any integer (``round_id=0`` for a
  single-shot game, ``epoch`` for RL, etc.) or omit the argument entirely if
  they *do* implement the legacy attribute.
• Unit tests can inject a synthetic round number without mocking
  ``GameManager`` internals.
"""

from __future__ import annotations

import time
import traceback
import re
import json
from datetime import datetime
from colorama import Fore
from typing import TYPE_CHECKING, Any, Dict

from llm.log_utils import get_prompt_filename, save_llm_artefact
from llm.parsing_utils import parse_llm_response
from llm.prompt_utils import create_parser_prompt, prepare_snake_prompt

if TYPE_CHECKING:
    from core.game_manager import GameManager

__all__ = [
    "check_llm_health",
    "extract_state_for_parser",
    "get_llm_response",
]

def check_llm_health(llm_client: LLMClient, max_retries: int = 2, retry_delay: int = 2) -> Tuple[bool, str]:
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


def extract_state_for_parser(game_manager: "GameManager") -> Tuple[str, str, str]:
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

def get_llm_response(game_manager: "GameManager", *, round_id: int | None = None):
    """Get a response from the LLM system based on the current game state.
    
    This function handles both single-LLM and dual-LLM configurations:
    - In single-LLM mode: Uses only the primary LLM for generating and parsing moves
    - In dual-LLM mode: Uses the primary LLM for strategy and the secondary LLM for parsing
    
    Args:
        game_manager: The active :class:`core.game_manager.GameManager`.
        round_id: Explicit round number used for filenames & metadata.  When
            ``None`` (default) we fall back to ``game_manager.round_count`` to
            preserve Task-0 behaviour.

    Returns:
        Tuple ``(next_move, game_active)`` where ``next_move`` is the first
        direction from the newly generated plan (or ``None`` on error) and
        ``game_active`` indicates whether the game should continue.

    Notes
    -----
    • Passing the round number from the *caller* side restores the correct
      dependency direction: the *game-flow* layer (loop or agent) decides
      *when* the LLM should be invoked and what contextual identifiers are
      relevant.  The communication helper only transmits bytes.

    • Example call for Task-0 (round-based planning)::

          next_move, cont = get_llm_response(manager, round_id=manager.round_count)

      Example call for a hypothetical Task-2 that queries once per *game*::

          next_move, cont = get_llm_response(manager, round_id=0)

    • When the argument is omitted entirely older Task-0 code keeps working
      because we internally fall back to the managerʼs attribute.
    """
    # Start tracking LLM communication time
    game_manager.game.game_state.record_llm_communication_start()

    # Get game state
    game_state = game_manager.game.get_state_representation()

    # Format prompt for LLM
    prompt = game_state

    # Get parser input for metadata
    parser_input = extract_state_for_parser(game_manager)

    # Determine which round number to embed in filenames / metadata.
    # Fallback to the managerʼs attribute so old call-sites remain unchanged.
    _round = round_id if round_id is not None else getattr(game_manager, "round_count", 0)

    # Log the prompt using centralized naming
    prompt_filename = get_prompt_filename(
        game_number=game_manager.game_count + 1,
        round_number=_round,
        file_type="prompt",
    )

    prompt_metadata = {
        "PRIMARY LLM Provider": game_manager.args.provider,
        "PRIMARY LLM Model": game_manager.args.model or get_default_model(game_manager.args.provider),
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
            game_number=game_manager.game_count + 1,
            round_number=_round,
            file_type="raw_response",
        )

        # Get parser input to include in metadata
        parser_input = extract_state_for_parser(game_manager)

        response_metadata = {
            "PRIMARY LLM Request Time": request_time,
            "PRIMARY LLM Response Time": response_time,
            "PRIMARY LLM Provider": game_manager.args.provider,
            "PRIMARY LLM Model": game_manager.args.model or get_default_model(game_manager.args.provider),
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
                game_number=game_manager.game_count + 1,
                round_number=_round,
                file_type="parser_prompt",
            )

            parser_prompt_metadata = {
                "SECONDARY LLM Provider": game_manager.args.parser_provider,
                "SECONDARY LLM Model": game_manager.args.parser_model or get_default_model(game_manager.args.parser_provider),
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
                parser_output = parse_and_format(response)
                
                # Use the primary LLM response for display
                from utils.text_utils import process_response_for_display
                game_manager.game.processed_response = process_response_for_display(response)
            else:
                # Record parser response time
                parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save the secondary LLM response using centralized naming
                parsed_response_filename = get_prompt_filename(
                    game_number=game_manager.game_count + 1,
                    round_number=_round,
                    file_type="parsed_response",
                )

                parsed_response_metadata = {
                    "SECONDARY LLM Request Time": parser_request_time,
                    "SECONDARY LLM Response Time": parser_response_time,
                    "SECONDARY LLM Provider": game_manager.args.parser_provider,
                    "SECONDARY LLM Model": (
                        game_manager.args.parser_model
                        or (get_default_model(game_manager.args.parser_provider) if game_manager.args.parser_provider else None)
                    ),
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
                parser_output = parse_and_format(secondary_response)
                
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
            parser_output = parse_and_format(response)
            
            # Store the primary LLM response for display in the UI
            from utils.text_utils import process_response_for_display
            game_manager.game.processed_response = process_response_for_display(response)

        # ----------------
        # Detect explicit "NO_PATH_FOUND" reasoning when *no* moves were
        # produced.  Case-insensitive match, tolerant of extra whitespace.
        # ----------------

        if parser_output and not parser_output.get("moves"):
            reason_text = str(parser_output.get("reasoning", "")).upper()
            if "NO_PATH_FOUND" in reason_text:
                game_manager.consecutive_no_path_found += 1
                streak = game_manager.consecutive_no_path_found
                limit_np = game_manager.args.max_consecutive_no_path_found_allowed
                print(
                    Fore.YELLOW +
                    f"⚠️ NO_PATH_FOUND from LLM. Consecutive: {streak}/{limit_np}"
                )

                # Mark flag so the game loop logs the sentinel move & handles any
                # special sleep logic.
                game_manager.last_no_path_found = True

                # If threshold reached – immediate game over
                if streak >= limit_np:
                    print(
                        Fore.RED +
                        f"❌ Maximum consecutive NO_PATH_FOUND reached ({limit_np}). Game over."
                    )
                    game_manager.game.game_state.record_game_end("MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED")
                    return None, False
            else:
                # Any response *other* than NO_PATH_FOUND resets the streak & flag
                game_manager.consecutive_no_path_found = 0
                game_manager.last_no_path_found = False

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

            # Keep the *full* plan in ``planned_moves`` so the game loop can
            # pop the first element explicitly.  This mirrors the behaviour
            # of the pre-refactor code and avoids edge-cases where an empty
            # list would prematurely skip ``finish_round()``.

            next_move = parser_output["moves"][0] if parser_output["moves"] else None
            game_manager.game.planned_moves = list(parser_output["moves"])

            # If we got a valid move, reset the consecutive empty steps counter
            if next_move:
                game_manager.consecutive_empty_steps = 0
                print(
                    Fore.GREEN
                    + f"🐍 Next move: {next_move} (Game {game_manager.game_count + 1}, Round {_round})"
                )
                
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
                print(Fore.YELLOW + "⚠️ No valid moves found. Delegating EMPTY-step handling to game loop.")

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
        # ---------------------
        # SOMETHING_IS_WRONG sentinel handling (exception path)
        # ---------------------
        # Any exception that bubbles up to this point means the LLM replied but
        # we could not parse *any* usable move set – usually malformed / empty
        # JSON or a truncated answer.  We treat this differently from the EMPTY
        # sentinel: EMPTY means *successfully parsed* but zero moves; this block
        # means *parsing failed outright*.  Maintaining a dedicated
        # consecutive-error counter lets the user decide how tolerant the system
        # should be towards repeated parsing failures without conflating them
        # with EMPTY back-off logic.
        #
        # Workflow:
        # 1. Mark skip_empty_this_tick so the game loop doesn't add another
        #    EMPTY sentinel.
        # 2. Append explicit SOMETHING_IS_WRONG move to the current move list
        #    for full traceability in logs & replays.
        # 3. Increment consecutive_something_is_wrong and optionally end the
        #    game when it exceeds the CLI limit.
        # Record that there was an error in the LLM response BEFORE ending communication time
        # Prevent duplicate EMPTY record later in game loop
        game_manager.skip_empty_this_tick = True
        # Append sentinel move so replay shows the issue explicitly
        game_manager.current_game_moves.append("SOMETHING_IS_WRONG")
        
        # Ensure round data is synchronized BEFORE ending communication time
        game_manager.game.game_state.round_manager.sync_round_data()
        
        # End tracking LLM communication time even if there was an error
        game_manager.game.game_state.record_llm_communication_end()
        
        # Increment consecutive errors – and because this tick is *not* an
        # EMPTY move, reset the empty-move streak so the threshold represents
        # truly consecutive EMPTY ticks only.
        game_manager.consecutive_empty_steps = 0
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
