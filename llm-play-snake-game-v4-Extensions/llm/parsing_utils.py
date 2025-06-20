"""Utilities for turning raw LLM text into structured *move lists*.

This module is intentionally self-contained so it can be unit-tested without
booting the entire game stack.  All heavy JSON heuristics live here; higher
layers only inspect its return value.

As LLM is Task0 specific, this whole module is Task0 specific.
"""

from __future__ import annotations

from utils.json_utils import (
    extract_valid_json,
    extract_json_from_code_block,
    extract_json_from_text,
    extract_moves_from_arrays,
)
from utils.moves_utils import normalize_directions
from typing import Any, Dict, Optional

__all__ = [
    "parse_and_format",
    "parse_llm_response",
]

def parse_and_format(llm_response: str) -> Optional[Dict[str, Any]]:
    """Parse an LLM response and format it for use by the game.
    
    Core parsing function that extracts structured move data from LLM responses.
    Works with both primary and secondary LLM responses.
    
    Args:
        llm_response: Raw response from the LLM
        
    Returns:
        Dictionary with parsed data or None if parsing failed
    """
    try:
        # Parse JSON directly from response
        parsed_data = extract_valid_json(llm_response, attempt_id=0)
        
        if parsed_data and "moves" in parsed_data:
            # Record success and return data if it includes a 'moves' field
            parsed_data["moves"] = normalize_directions(parsed_data["moves"])
            return parsed_data
        return None
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None

def parse_llm_response(
    response: str,
    processed_response_func,
    game_instance,
) -> Optional[str]:
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

        # If response is empty or contains an error marker
        if not response or "ERROR" in response.upper():
            print("Error in LLM response or empty response")
            return None

        # Parse JSON from the response
        print("Attempting to extract JSON from LLM response…")

        # ----------------
        # Helper – attempt several extraction strategies in order of
        # reliability, returning the first non-None result.
        # ----------------

        def _extract_json(text: str):
            """Try all extraction helpers in descending order of strictness."""

            for fn in (
                extract_valid_json,        # strict – full JSON
                extract_json_from_code_block,  # fenced ```json``` block
                extract_json_from_text,    # loose {...} in free text
                extract_moves_from_arrays, # final fallback – just ["UP", …]
            ):
                data = fn(text, attempt_id=0) if fn is extract_valid_json else fn(text)
                if data:
                    return data
            return None

        json_data = _extract_json(response)

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

                # Filter out any invalid reversals in the planned moves
                filtered_moves = game_instance.filter_invalid_reversals(moves[1:], current_direction)
                if len(filtered_moves) < len(moves[1:]):
                    print(f"Filtered out {len(moves[1:]) - len(filtered_moves)} invalid reversal moves")

                game_instance.planned_moves = filtered_moves

                # Return the first move
                return moves[0]

            # If we reach here, moves list was empty
            print("Moves list is empty")
        elif json_data:
            print(f"JSON data found but no 'moves' key. Keys: {json_data.keys()}")
        else:
            print("No valid JSON data found")

        # If we didn't get moves, check if reasoning contains "SOMETHING_IS_WRONG"
        reasoning_text = str(json_data.get("reasoning", "")).upper() if json_data else ""
        if "SOMETHING_IS_WRONG" in reasoning_text:
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
