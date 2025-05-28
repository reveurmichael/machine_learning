"""
JSON utilities for the Snake game.
Handles JSON parsing, validation, and extraction from LLM responses.
"""

import json
import re
import os
import numpy as np
from datetime import datetime

# Global tracking of JSON parsing statistics
json_error_stats = {
    "total_extraction_attempts": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "json_decode_errors": 0,
    "text_extraction_errors": 0,
    "pattern_extraction_success": 0
}

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
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_json_error_stats():
    """Get the current JSON error statistics.
    
    Returns:
        Dictionary with error statistics
    """
    return json_error_stats

def reset_json_error_stats():
    """Reset all JSON error statistics to zero."""
    for key in json_error_stats:
        json_error_stats[key] = 0

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
    
    # Create experiment info
    experiment_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": args_dict,
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_game_movement_time": 0,
            "total_waiting_time": 0
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
            "error_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0
        },
        "json_parsing_stats": json_error_stats.copy()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to file
    file_path = os.path.join(directory, "summary.json")
    with open(file_path, "w") as f:
        json.dump(experiment_info, f, indent=2, cls=NumPyJSONEncoder)
    
    return experiment_info

def update_experiment_info_json(log_dir, **kwargs):
    """Update the experiment information JSON with new data.
    
    Args:
        log_dir: Directory containing the summary.json file
        **kwargs: Additional fields to update
    """
    # Read existing summary
    summary_path = os.path.join(log_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return
    
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Error reading summary.json: {e}")
        return
    
    # Update summary with new values
    for key, value in kwargs.items():
        if key == "json_error_stats":
            summary["json_parsing_stats"] = value
        elif key == "is_continuation" and value:
            # If this is a continuation run, merge stats from all game files
            try:
                merged_stats = merge_game_stats_for_continuation(log_dir)
                deep_update_dict(summary, merged_stats)
            except Exception as e:
                print(f"Error merging game stats: {e}")
        else:
            # For other keys, just update directly
            if key in summary:
                summary[key] = value
    
    # Save updated summary
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
    except Exception as e:
        print(f"Error writing summary.json: {e}")

def deep_update_dict(original, update):
    """Recursively update a dictionary with values from another dictionary.
    
    Args:
        original: Dictionary to update
        update: Dictionary with new values
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update_dict(original[key], value)
        else:
            original[key] = value

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
    """Validate that data contains the expected format for the game.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Tuple of (is_valid, message) where:
            is_valid: Boolean indicating if the data is valid
            message: Error message if not valid, None otherwise
    """
    if not isinstance(data, dict):
        return False, "Data is not a dictionary"
    
    if "moves" not in data:
        return False, "Missing 'moves' key"
    
    if not isinstance(data["moves"], list):
        return False, "Moves is not a list"
    
    # Validate moves
    valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
    for i, move in enumerate(data["moves"]):
        if not isinstance(move, str):
            return False, f"Move {i} is not a string"
        
        move_upper = move.upper()
        if move_upper not in valid_moves:
            return False, f"Invalid move: {move}"
    
    return True, None

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    global json_error_stats
    
    # Match JSON code blocks
    code_block_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response, re.DOTALL)
    
    for code_block in code_block_matches:
        try:
            # Preprocess the code block
            processed_block = preprocess_json_string(code_block)
            data = json.loads(processed_block)
            
            # Validate it has the expected format
            if isinstance(data, dict) and "moves" in data:
                return data
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Error during code block extraction: {e}")
    
    return None

def extract_valid_json(text, game_state=None):
    """Extract valid JSON data from text.
    
    Args:
        text: Text that may contain JSON
        game_state: Optional GameData instance to record parsing stats
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    global json_error_stats
    json_error_stats["total_extraction_attempts"] += 1
    
    # Also record in game_state if provided
    if game_state is not None:
        game_state.record_json_extraction_attempt(success=False)  # Will update if successful
    
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        if isinstance(data, dict) and "moves" in data:
            json_error_stats["successful_extractions"] += 1
            
            # Record success in game_state if provided
            if game_state is not None:
                game_state.record_json_extraction_attempt(success=True)
                
            return data
    except json.JSONDecodeError:
        json_error_stats["json_decode_errors"] += 1
        
        # Record in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=False, error_type="decode")
    
    # Try extracting from code block
    json_data = extract_json_from_code_block(text)
    if json_data:
        json_error_stats["successful_extractions"] += 1
        
        # Record success in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=True)
            
        return json_data
        
    # Try extracting from regular text
    json_data = extract_json_from_text(text)
    if json_data:
        json_error_stats["successful_extractions"] += 1
        
        # Record success in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=True)
            
        return json_data
            
    json_error_stats["failed_extractions"] += 1
    
    # Final failure record in game_state if provided
    if game_state is not None:
        game_state.record_json_extraction_attempt(success=False)
        
    return None

def extract_json_from_text(response):
    """Extract JSON data from text response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    global json_error_stats
    
    try:
        # Direct JSON parsing for clean responses
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "moves" in data:
                return data
        except json.JSONDecodeError:
            # Continue to alternative extraction methods
            pass
        except Exception as e:
            print(f"Error during direct JSON parsing: {e}")
            
        # Match JSON patterns with quoted or unquoted keys
        json_match = re.search(r'\{[\s\S]*?["\']moves["\']?\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        
        # More permissive pattern for unquoted keys
        if not json_match:
            json_match = re.search(r'\{\s*moves\s*:[\s\S]*?\}', response, re.DOTALL)
            
        if not json_match:
            # No JSON structure found
            json_error_stats["text_extraction_errors"] += 1
            return None
            
        json_str = json_match.group(0)
        # Preprocess to standardize format
        json_str = preprocess_json_string(json_str)
        
        # Parse the preprocessed JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_error_stats["text_extraction_errors"] += 1
            print(f"JSON parsing error: {e}")
            
            # Extract moves using pattern matching
            return extract_moves_pattern(json_str)
            
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return None

def extract_moves_pattern(json_str):
    """Extract moves from a JSON string using pattern matching.
    
    Args:
        json_str: JSON string to extract moves from
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    global json_error_stats
    
    try:
        # Extract moves array
        moves_array_match = re.search(r'["\']?moves["\']?\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
        if not moves_array_match:
            return None
            
        moves_array = moves_array_match.group(1)
        # Extract valid move strings
        move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
        valid_moves = [move.upper() for move in move_matches 
                      if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
    
        if valid_moves:
            json_error_stats["pattern_extraction_success"] += 1
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
        
        # Check if these are valid moves
        valid_moves = [item.upper() for item in quoted_items 
                      if item.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
        
        if valid_moves and len(valid_moves) > 0:
            return {"moves": valid_moves}
    
    return None 

def merge_game_stats_for_continuation(log_dir):
    """Merge statistics from all game files for a more accurate summary in continuation mode.
    
    This function reads all game_N.json files in the log directory and combines them
    to create accurate aggregated statistics.
    
    Args:
        log_dir: Directory containing game log files
        
    Returns:
        Dictionary containing merged statistics from all games
    """
    import glob
    
    # Get all game files
    game_files = glob.glob(os.path.join(log_dir, "game_*.json")) + glob.glob(os.path.join(log_dir, "game*.json"))
    
    # Initialize aggregated stats
    aggregated_stats = {
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_game_movement_time": 0,
            "total_waiting_time": 0
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
            "error_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0
        },
        "json_parsing_stats": {
            "total_extraction_attempts": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "json_decode_errors": 0,
            "text_extraction_errors": 0,
            "pattern_extraction_success": 0
        }
    }
    
    # Process each game file
    for game_file in game_files:
        try:
            with open(game_file, "r") as f:
                game_data = json.load(f)
                
            # Update game statistics
            aggregated_stats["game_statistics"]["total_games"] += 1
            aggregated_stats["game_statistics"]["total_score"] += game_data.get("score", 0)
            aggregated_stats["game_statistics"]["total_steps"] += game_data.get("steps", 0)
            aggregated_stats["game_statistics"]["scores"].append(game_data.get("score", 0))
            
            # Update time statistics
            time_stats = game_data.get("time_stats", {})
            aggregated_stats["time_statistics"]["total_llm_communication_time"] += time_stats.get("total_llm_communication_time", 0)
            aggregated_stats["time_statistics"]["total_game_movement_time"] += time_stats.get("total_game_movement_time", 0)
            aggregated_stats["time_statistics"]["total_waiting_time"] += time_stats.get("total_waiting_time", 0)
            
            # Update token usage stats
            token_stats = game_data.get("token_stats", {})
            primary_token_stats = token_stats.get("primary_llm", {})
            secondary_token_stats = token_stats.get("secondary_llm", {})
            
            aggregated_stats["token_usage_stats"]["primary_llm"]["total_tokens"] += primary_token_stats.get("total_tokens", 0)
            aggregated_stats["token_usage_stats"]["primary_llm"]["total_prompt_tokens"] += primary_token_stats.get("prompt_tokens", 0)
            aggregated_stats["token_usage_stats"]["primary_llm"]["total_completion_tokens"] += primary_token_stats.get("completion_tokens", 0)
            
            aggregated_stats["token_usage_stats"]["secondary_llm"]["total_tokens"] += secondary_token_stats.get("total_tokens", 0)
            aggregated_stats["token_usage_stats"]["secondary_llm"]["total_prompt_tokens"] += secondary_token_stats.get("prompt_tokens", 0)
            aggregated_stats["token_usage_stats"]["secondary_llm"]["total_completion_tokens"] += secondary_token_stats.get("completion_tokens", 0)
            
            # Update json parsing stats
            json_parsing_stats = game_data.get("json_parsing_stats", {})
            for key in aggregated_stats["json_parsing_stats"]:
                aggregated_stats["json_parsing_stats"][key] += json_parsing_stats.get(key, 0)
                
        except Exception as e:
            print(f"Error processing game file {game_file}: {e}")
            
    return aggregated_stats 