"""
JSON processing system for the Snake game.
Comprehensive utilities for JSON parsing, validation, and extraction from LLM responses,
with special handling for common formatting variations and error conditions.
"""

import json
import re
import os
import numpy as np
from datetime import datetime

class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        """Handle NumPy types for JSON serialization.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_experiment_info_json(args, directory):
    """Save experiment configuration information to a JSON file.
    
    Args:
        args: Command line arguments
        directory: Directory to save the file in
        
    Returns:
        Experiment info dictionary
    """
    # Convert args to dict
    args_dict = vars(args)
    
    # Clean up configuration - set parser info to null if it's 'none'
    config_dict = args_dict.copy()
    if 'parser_provider' in config_dict and (not config_dict['parser_provider'] or config_dict['parser_provider'].lower() == 'none'):
        # In single LLM mode, set parser fields to null instead of removing them
        config_dict['parser_provider'] = None
        config_dict['parser_model'] = None
    
    # Create experiment info
    experiment_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": config_dict,
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_primary_llm_communication_time": 0,
            "total_secondary_llm_communication_time": 0,
        },
        "token_usage_stats": {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        },
        "step_stats": {
            "empty_steps": 0,
            "something_is_wrong_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0  # Aggregated count across all games
        },
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to file
    file_path = os.path.join(directory, "summary.json")
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, cls=NumPyJSONEncoder)
    
    return experiment_info

def save_session_stats(log_dir, **kwargs):
    """Save session statistics to the summary JSON file.
    
    Args:
        log_dir: Directory containing the summary.json file
        **kwargs: Statistics fields to save
    """
    # Read existing summary file
    summary_path = os.path.join(log_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return
    
    try:
        with open(summary_path, "r", encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Error reading summary.json: {e}")
        return
    
    def _safe_set(target: dict, key: str, val):
        """Overwrite only with a real, non-zero value."""
        if val:
            target[key] = val
    
    # Ensure all required sections exist
    if "game_statistics" not in summary:
        summary["game_statistics"] = {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        }
    
    if "time_statistics" not in summary:
        summary["time_statistics"] = {
            "total_llm_communication_time": 0,
            "total_primary_llm_communication_time": 0,
            "total_secondary_llm_communication_time": 0,
        }
    
    if "token_usage_stats" not in summary:
        summary["token_usage_stats"] = {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        }
    
    if "step_stats" not in summary:
        summary["step_stats"] = {
            "empty_steps": 0,
            "something_is_wrong_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0  # Aggregated count across all games
        }
    
    # Apply new statistics values to the appropriate sections
    for key, value in kwargs.items():
        if key == "game_count":
            summary["game_statistics"]["total_games"] = value
        elif key == "total_score":
            summary["game_statistics"]["total_score"] = value
        elif key == "total_steps":
            summary["game_statistics"]["total_steps"] = value
        elif key == "game_scores":
            summary["game_statistics"]["scores"] = value
        elif key == "empty_steps":
            summary["step_stats"]["empty_steps"] = value  # Already accumulated in process_game_over
        elif key == "something_is_wrong_steps":
            summary["step_stats"]["something_is_wrong_steps"] = value  # Already accumulated in process_game_over
        elif key == "valid_steps":
            summary["step_stats"]["valid_steps"] = value  # Already accumulated in process_game_over
        elif key == "invalid_reversals":
            summary["step_stats"]["invalid_reversals"] = value
        elif key == "time_stats":
            # Handle time statistics if provided
            if value and isinstance(value, dict):
                ts = value
                _safe_set(summary["time_statistics"], "total_llm_communication_time",
                          ts.get("llm_communication_time"))
                _safe_set(summary["time_statistics"], "total_primary_llm_communication_time",
                          ts.get("primary_llm_communication_time"))
                _safe_set(summary["time_statistics"], "total_secondary_llm_communication_time",
                          ts.get("secondary_llm_communication_time"))
        elif key == "token_stats":
            # Handle token statistics if provided
            if value and isinstance(value, dict):
                if "primary" in value and isinstance(value["primary"], dict):
                    primary = value["primary"]
                    
                    # Only add token stats if they're not None
                    if "total_tokens" in primary and primary["total_tokens"] is not None:
                        summary["token_usage_stats"]["primary_llm"]["total_tokens"] = primary["total_tokens"]
                    if "total_prompt_tokens" in primary and primary["total_prompt_tokens"] is not None:
                        summary["token_usage_stats"]["primary_llm"]["total_prompt_tokens"] = primary["total_prompt_tokens"]
                    if "total_completion_tokens" in primary and primary["total_completion_tokens"] is not None:
                        summary["token_usage_stats"]["primary_llm"]["total_completion_tokens"] = primary["total_completion_tokens"]
                
                if "secondary" in value and isinstance(value["secondary"], dict):
                    secondary = value["secondary"]
                    
                    # Only add token stats if they're not None
                    if "total_tokens" in secondary and secondary["total_tokens"] is not None:
                        summary["token_usage_stats"]["secondary_llm"]["total_tokens"] = secondary["total_tokens"]
                    if "total_prompt_tokens" in secondary and secondary["total_prompt_tokens"] is not None:
                        summary["token_usage_stats"]["secondary_llm"]["total_prompt_tokens"] = secondary["total_prompt_tokens"]
                    if "total_completion_tokens" in secondary and secondary["total_completion_tokens"] is not None:
                        summary["token_usage_stats"]["secondary_llm"]["total_completion_tokens"] = secondary["total_completion_tokens"]
        elif key == "step_stats":
            # Handle step statistics if provided as a complete dictionary
            if value and isinstance(value, dict):
                if "empty_steps" in value:
                    summary["step_stats"]["empty_steps"] = value["empty_steps"]  # Already accumulated in process_game_over
                if "something_is_wrong_steps" in value:
                    summary["step_stats"]["something_is_wrong_steps"] = value["something_is_wrong_steps"]  # Already accumulated in process_game_over
                if "valid_steps" in value:
                    summary["step_stats"]["valid_steps"] = value["valid_steps"]  # Already accumulated in process_game_over
                if "invalid_reversals" in value:
                    summary["step_stats"]["invalid_reversals"] = value["invalid_reversals"]  # Already accumulated in process_game_over
        else:
            # For any other fields, add them at the top level
            summary[key] = value
    
    # After merging totals, compute average token usage per game if total_games > 0
    total_games = summary["game_statistics"].get("total_games", 0)
    if total_games:
        # Primary averages
        prim = summary["token_usage_stats"].get("primary_llm", {})
        if prim:
            prim["avg_tokens"] = prim.get("total_tokens", 0) / total_games
            prim["avg_prompt_tokens"] = prim.get("total_prompt_tokens", 0) / total_games
            prim["avg_completion_tokens"] = prim.get("total_completion_tokens", 0) / total_games

        # Secondary averages
        sec = summary["token_usage_stats"].get("secondary_llm", {})
        if sec:
            sec["avg_tokens"] = sec.get("total_tokens", 0) / total_games if sec.get("total_tokens") is not None else None
            sec["avg_prompt_tokens"] = sec.get("total_prompt_tokens", 0) / total_games if sec.get("total_prompt_tokens") is not None else None
            sec["avg_completion_tokens"] = sec.get("total_completion_tokens", 0) / total_games if sec.get("total_completion_tokens") is not None else None

        # Average response times (seconds)
        ts = summary.get("time_statistics", {})
        if ts and total_games:
            if ts.get("total_primary_llm_communication_time") is not None:
                ts["avg_primary_response_time"] = ts.get("total_primary_llm_communication_time", 0) / total_games
            if ts.get("total_secondary_llm_communication_time") is not None:
                ts["avg_secondary_response_time"] = ts.get("total_secondary_llm_communication_time", 0) / total_games
    
    # Save the summary file
    try:
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
    except Exception as e:
        print(f"Error writing summary.json: {e}")


def preprocess_json_string(json_str):
    """Preprocess a JSON string to fix common formatting issues.
    
    Args:
        json_str: JSON string to preprocess
        
    Returns:
        Preprocessed JSON string
    """
    # Remove comments
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix unquoted keys
    json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)
    
    # Fix trailing commas in arrays
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Fix trailing commas in objects
    json_str = re.sub(r',\s*\}', '}', json_str)
    
    # Fix single quotes to double quotes (carefully, to avoid breaking JSON)
    in_string = False
    in_escape = False
    result = []
    
    for char in json_str:
        if char == '\\' and not in_escape:
            in_escape = True
            result.append(char)
        elif in_escape:
            in_escape = False
            result.append(char)
        elif char == '"' and not in_escape:
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string and not in_escape:
            result.append('"')
        else:
            result.append(char)
    
    return ''.join(result)

def validate_json_format(data):
    """Validate that the JSON data has the expected format.
    
    Args:
        data: JSON data to validate
        data: JSON data to validate
        
    Returns:
        (is_valid, error_message) tuple. If is_valid is False, error_message
        contains a description of the error.
    """
    if not isinstance(data, dict):
        return False, "JSON data is not a dictionary"
        
    if "moves" not in data:
        return False, "JSON data does not contain a 'moves' key"
        
    if not isinstance(data["moves"], list):
        return False, "The 'moves' field is not a list"
        
    # Validate all moves are valid directions
    valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    for i, move in enumerate(data["moves"]):
        if not isinstance(move, str):
            return False, f"Move is not a string: {move}"
        
        # Convert move to uppercase for case-insensitive validation
        move_upper = move.upper()
        if move_upper not in valid_moves:
            print(f"JSON validation error: Invalid move: '{move}' (upper: '{move_upper}'), valid moves are {valid_moves}")
            return False, f"Invalid move: {move}"
        
        # Ensure all moves are in uppercase format
        data["moves"][i] = move_upper
    
    return True, None

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    # Match JSON code blocks with comprehensive language identifier patterns
    # Covers: ```json, ```javascript, ```js, and plain ```
    code_block_matches = re.findall(r'```(?:json|javascript|js)?\s*([\s\S]*?)```', response, re.DOTALL)
    
    for i, code_block in enumerate(code_block_matches):
        try:
            # Preprocess the code block
            processed_block = preprocess_json_string(code_block)
            
            # Try to parse the JSON
            data = json.loads(processed_block)
            
            # Validate it has the expected format
            if isinstance(data, dict) and "moves" in data:
                print(f"✅ Valid JSON found in code block {i+1}")
                return data
            else:
                print(f"❌ Code block {i+1} does not contain a 'moves' key")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in code block {i+1}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error during code block extraction for block {i+1}: {e}")
    
    # If we get here, we couldn't extract valid JSON from any code block
    if len(code_block_matches) > 0:
        pass

    # If the normal extraction fails, try a more aggressive pattern matching approach
    # This handles cases where the LLM includes formatting characters around JSON
    try:
        # Look for JSON-like patterns in the text
        json_pattern = r'\{\s*"moves"\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            moves_content = match.group(1)
            # Extract the move strings from the array
            move_matches = re.findall(r'"([^"]+)"', moves_content)
            
            # Validate and convert to uppercase
            valid_moves = []
            for move in move_matches:
                move_upper = move.upper()
                if move_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    valid_moves.append(move_upper)
            
            if valid_moves:
                print(f"✅ Successfully extracted moves using pattern matching: {valid_moves}")
                return {"moves": valid_moves}
    except Exception as e:
        print(f"❌ Error during pattern extraction: {e}")
    
    return None

def extract_valid_json(text, game_state=None, attempt_id=0):
    """Extract valid JSON data from text.
    
    Attempts multiple extraction strategies:
    1. Direct JSON parsing
    2. Code block extraction
    3. Text-based extraction
    4. Move array extraction
    
    Args:
        text: Text that may contain JSON
        game_state: Optional GameData instance to record parsing stats
        attempt_id: Attempt number (0 for first attempt, incremented for retries)
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        if isinstance(data, dict) and "moves" in data:
            # Validate the moves format
            is_valid, error_msg = validate_json_format(data)
            if not is_valid:
                # Only print error message on first attempt
                if attempt_id == 0:
                    print(f"JSON format validation error: {error_msg}")
                return None
                
            # Print detailed info about the moves
            print(f"✅ Successfully extracted moves: {data['moves']}")
            
            return data
        else:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON missing 'moves' key or not a dict. Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
    except json.JSONDecodeError as e:
        # Only print error message on first attempt
        if attempt_id == 0:
            print(f"JSON decode error: {e}")
    
    # Try extracting from code block
    json_data = extract_json_from_code_block(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (code block): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from code block: {json_data['moves']}")
        
        return json_data

    # Try extracting from regular text
    json_data = extract_json_from_text(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (text): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from text: {json_data['moves']}")
        
        return json_data
            
    # As a last resort, try to extract move arrays directly
    json_data = extract_moves_from_arrays(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (arrays): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from arrays: {json_data['moves']}")
        return json_data
            
    # No valid JSON found
    return None

def extract_json_from_text(response):
    """Extract JSON data from text response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    # Special case pattern matching for moves array
    moves_pattern = r'"moves"\s*:\s*\[(.*?)\]'
    match = re.search(moves_pattern, response, re.DOTALL)
    
    if match:
        moves_content = match.group(1)
        
        # Extract quoted strings
        moves = re.findall(r'["\']([^"\']+)["\']', moves_content)
        
        if moves:
            # Convert all moves to uppercase
            standardized_moves = [move.upper() for move in moves]
            
            # Filter to only valid moves
            valid_moves = [move for move in standardized_moves if move in ["UP", "DOWN", "LEFT", "RIGHT"]]
            
            if valid_moves:
                print(f"Extracted moves from text: {valid_moves}")
                return {"moves": valid_moves}
    
    # Try extracting moves pattern
    moves = extract_moves_pattern(response)
    if moves:
        # Convert all moves to uppercase
        standardized_moves = [move.upper() for move in moves]
        
        # Filter to only valid moves
        valid_moves = [move for move in standardized_moves if move in ["UP", "DOWN", "LEFT", "RIGHT"]]
        
        if valid_moves:
            print(f"Extracted moves using pattern matching: {valid_moves}")
            return {"moves": valid_moves}
    
    return None

def extract_moves_pattern(json_str):
    """Extract moves from a JSON string using pattern matching.
    
    Args:
        json_str: JSON string to extract moves from
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    try:
        # Extract moves array
        moves_array_match = re.search(r'["\']?moves["\']?\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
        if not moves_array_match:
            return None
            
        moves_array = moves_array_match.group(1)
        # Extract valid move strings
        move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
        
        # Convert all moves to uppercase for case-insensitive validation
        valid_moves = []
        for move in move_matches:
            move_upper = move.upper()
            if move_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                valid_moves.append(move_upper)
    
        if valid_moves:
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Move extraction error: {e}")
        return None

def extract_moves_from_arrays(response):
    """Extract moves from arrays in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    # Look for arrays in the response
    array_matches = re.findall(r'\[(.*?)\]', response, re.DOTALL)
    
    for array_str in array_matches:
        # Extract quoted strings
        quoted_items = re.findall(r'["\']([^"\']+)["\']', array_str)
        
        # Convert all items to uppercase for case-insensitive validation
        valid_moves = []
        for item in quoted_items:
            item_upper = item.upper()
            if item_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                valid_moves.append(item_upper)  # Store in uppercase format
        
        if valid_moves and len(valid_moves) > 0:
            return {"moves": valid_moves}
    
    return None

def validate_game_summary(summary):
    """Validate a game summary JSON to ensure data consistency.
    
    Performs sanity checks on the game summary to ensure:
    - moves length matches steps
    - each round's moves are a subset of the global moves
    
    Args:
        summary: The game summary dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check steps against detailed_history.moves
        if "steps" in summary and "detailed_history" in summary and "moves" in summary["detailed_history"]:
            steps = summary["steps"]
            moves_len = len(summary["detailed_history"]["moves"])
            if steps != moves_len:
                return False, f"Moves length ({moves_len}) doesn't match steps ({steps})"
        
        # Check each round's moves against total steps
        if "detailed_history" in summary and "rounds_data" in summary["detailed_history"]:
            for rk, rd in summary["detailed_history"]["rounds_data"].items():
                if "moves" in rd and len(rd["moves"]) > summary["steps"]:
                    return False, f"Round {rk} moves ({len(rd['moves'])}) exceed total steps ({summary['steps']})"
        
        return True, "Validation passed"
    except Exception as e:
        return False, f"Validation error: {str(e)}" 