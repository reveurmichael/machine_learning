"""
Utility module for JSON processing in the Snake game.
Consolidates JSON extraction and validation functions used by multiple components.
"""

import os
import json
import re
from datetime import datetime

# Global counter for JSON extraction errors
json_error_stats = {
    "total_extraction_attempts": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "json_decode_errors": 0,
    "format_validation_errors": 0,
    "code_block_extraction_errors": 0,
    "text_extraction_errors": 0,
    "fallback_extraction_success": 0
}

def get_json_error_stats():
    """Get the current JSON error statistics.
    
    Returns:
        Dictionary with error statistics
    """
    return json_error_stats

def reset_json_error_stats():
    """Reset the JSON error statistics."""
    global json_error_stats
    json_error_stats = {
        "total_extraction_attempts": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "json_decode_errors": 0,
        "format_validation_errors": 0,
        "code_block_extraction_errors": 0,
        "text_extraction_errors": 0,
        "fallback_extraction_success": 0
    }

def save_experiment_info_json(args, directory):
    """Save experiment information to a JSON file.
    
    Args:
        args: Command line arguments
        directory: Directory to save to
        
    Returns:
        Path to the saved file
    """
    # Create experiment information in structured JSON format
    info_data = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "primary_llm": {
            "provider": args.provider,
            "model": args.model if args.model else 'Default model for provider'
        },
        "secondary_llm": {
            "provider": args.parser_provider if args.parser_provider else args.provider,
            "model": args.parser_model if args.parser_model else 'Default model for parser provider'
        },
        "game_configuration": {
            "max_steps_per_game": args.max_steps,
            "max_consecutive_empty_moves": args.max_empty_moves,
            "max_games": args.max_games
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to JSON file
    file_path = os.path.join(directory, "info.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2)
    
    return os.path.abspath(file_path)

def update_experiment_info_json(directory, game_count, total_score, total_steps, parser_usage_count=0, game_scores=None, empty_steps=0, error_steps=0, json_error_stats=None, max_empty_moves=3):
    """Update the experiment information JSON file with game statistics.
    
    Args:
        directory: Directory containing the info.json file
        game_count: Total number of games played
        total_score: Total score across all games
        total_steps: Total steps taken across all games
        parser_usage_count: Number of times the secondary LLM was used
        game_scores: List of individual game scores
        empty_steps: Number of empty steps (moves with empty JSON)
        error_steps: Number of steps with ERROR in reasoning
        json_error_stats: Dictionary containing JSON extraction error statistics
        max_empty_moves: Maximum number of consecutive empty moves before termination
    """
    file_path = os.path.join(directory, "info.json")
    
    # Read existing content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create a new file if it doesn't exist or is invalid
        info_data = {}
    
    # Calculate score statistics
    mean_score = total_score / game_count if game_count > 0 else 0
    max_score = 0
    min_score = 0
    
    if game_scores and len(game_scores) > 0:
        max_score = max(game_scores)
        min_score = min(game_scores)
    
    # Calculate step statistics
    empty_step_percentage = (empty_steps / total_steps) * 100 if total_steps > 0 else 0
    error_step_percentage = (error_steps / total_steps) * 100 if total_steps > 0 else 0
    valid_steps = total_steps - empty_steps - error_steps
    valid_step_percentage = (valid_steps / total_steps) * 100 if total_steps > 0 else 0
    
    # Add game statistics to the info data
    info_data["game_statistics"] = {
        "total_games": game_count,
        "total_score": total_score,
        "total_steps": total_steps,
        "mean_score": mean_score,
        "max_score": max_score,
        "min_score": min_score,
        "steps_per_game": total_steps / game_count if game_count > 0 else 0
    }
    
    # Add LLM usage statistics
    info_data["llm_usage_stats"] = {
        "parser_usage_count": parser_usage_count,
        "parser_usage_per_game": parser_usage_count / game_count if game_count > 0 else 0
    }
    
    # Add step statistics
    info_data["step_stats"] = {
        "empty_steps": empty_steps,
        "empty_step_percentage": empty_step_percentage,
        "error_steps": error_steps,
        "error_step_percentage": error_step_percentage,
        "valid_steps": valid_steps,
        "valid_step_percentage": valid_step_percentage,
        "max_consecutive_empty_moves": max_empty_moves
    }
    
    # Add JSON error statistics if available
    if json_error_stats:
        # Calculate success rate
        success_rate = (json_error_stats["successful_extractions"] / json_error_stats["total_extraction_attempts"]) * 100 if json_error_stats["total_extraction_attempts"] > 0 else 0
        failure_rate = (json_error_stats["failed_extractions"] / json_error_stats["total_extraction_attempts"]) * 100 if json_error_stats["total_extraction_attempts"] > 0 else 0
        
        info_data["json_parsing_stats"] = {
            "total_extraction_attempts": json_error_stats["total_extraction_attempts"],
            "successful_extractions": json_error_stats["successful_extractions"],
            "success_rate": success_rate,
            "failed_extractions": json_error_stats["failed_extractions"],
            "failure_rate": failure_rate,
            "json_decode_errors": json_error_stats["json_decode_errors"],
            "format_validation_errors": json_error_stats["format_validation_errors"],
            "code_block_extraction_errors": json_error_stats["code_block_extraction_errors"],
            "text_extraction_errors": json_error_stats["text_extraction_errors"],
            "fallback_extraction_success": json_error_stats["fallback_extraction_success"]
        }
    
    # Add efficiency metrics
    info_data["efficiency_metrics"] = {
        "apples_per_step": total_score/(total_steps if total_steps > 0 else 1),
        "steps_per_game": total_steps/game_count if game_count > 0 else 0,
        "valid_move_ratio": valid_steps/(total_steps if total_steps > 0 else 1)
    }
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2)

def preprocess_json_string(json_str):
    """Preprocess a JSON string to handle common issues.
    
    Args:
        json_str: JSON string to preprocess
        
    Returns:
        Preprocessed JSON string
    """
    # Handle single quotes as double quotes
    # Only do this conversion for keys and values, not within string content
    in_string = False
    processed = ""
    for i, char in enumerate(json_str):
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
            processed += char
        elif char == "'" and not in_string and (i == 0 or json_str[i-1] != '\\'):
            processed += '"'  # Replace ' with " for JSON keys
        else:
            processed += char
    
    # Handle unquoted keys
    processed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', processed)
    
    # Handle trailing commas in arrays and objects
    processed = re.sub(r',\s*([}\]])', r'\1', processed)
    
    # Handle multiline strings
    multiline_strings = {}
    counter = 0
    
    # First, extract all string values and replace them with placeholders
    for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', processed):
        value = match.group(1)
        if '\n' in value:
            key = f"__MULTILINE_STRING_{counter}__"
            multiline_strings[key] = value
            counter += 1
    
    # Replace placeholders with properly escaped strings
    for placeholder, value in multiline_strings.items():
        escaped_value = value.replace('\n', '\\n').replace('\r', '\\r')
        processed = processed.replace(placeholder, escaped_value)
    
    return processed

def validate_json_format(data):
    """Validate the format of parsed JSON data.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Boolean indicating if the format is valid
    """
    global json_error_stats
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if it has a 'moves' key
    if 'moves' not in data:
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if 'moves' is a list
    if not isinstance(data['moves'], list):
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if all moves are valid directions
    valid_directions = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
    for move in data['moves']:
        if not isinstance(move, str) or move.upper() not in valid_directions:
            json_error_stats["format_validation_errors"] += 1
            return False
    
    return True

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in text.
    
    Args:
        response: Text that may contain a code block with JSON
        
    Returns:
        Parsed JSON data or None if extraction failed
    """
    global json_error_stats
    
    try:
        # Match code blocks with optional language specifier
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response, re.MULTILINE)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                # Try direct parsing first
                data = json.loads(json_str)
                if validate_json_format(data):
                    return data
            except json.JSONDecodeError:
                # Try preprocessing
                try:
                    preprocessed = preprocess_json_string(json_str)
                    data = json.loads(preprocessed)
                    if validate_json_format(data):
                        return data
                except json.JSONDecodeError:
                    # Failed after preprocessing
                    json_error_stats["code_block_extraction_errors"] += 1
        return None
    except Exception as e:
        json_error_stats["code_block_extraction_errors"] += 1
        print(f"Error extracting JSON from code block: {e}")
        return None

def extract_valid_json(text):
    """Extract valid JSON data from text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    global json_error_stats
    json_error_stats["total_extraction_attempts"] += 1
    
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        json_error_stats["successful_extractions"] += 1
        return data
    except json.JSONDecodeError:
        json_error_stats["json_decode_errors"] += 1
        # Try with our preprocessing for single quotes and unquoted keys
        try:
            preprocessed_text = preprocess_json_string(text)
            data = json.loads(preprocessed_text)
            json_error_stats["successful_extractions"] += 1
            return data
        except json.JSONDecodeError:
            json_error_stats["json_decode_errors"] += 1
        
        # Try extracting from code block
        json_data = extract_json_from_code_block(text)
        if json_data:
            json_error_stats["successful_extractions"] += 1
            return json_data
            
        # Try extracting from regular text
        json_data = extract_json_from_text(text)
        if json_data:
            json_error_stats["successful_extractions"] += 1
            return json_data
                
    json_error_stats["failed_extractions"] += 1
    return None

def extract_json_from_text(response):
    """Extract JSON data directly from text without code block.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    global json_error_stats
    
    try:
        # First try direct JSON parsing (for clean JSON responses)
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "moves" in data:
                return data
        except json.JSONDecodeError:
            # Expected error for malformed JSON, just continue to next method
            pass
        except Exception as e:
            # Unexpected error during direct parsing
            print(f"Unexpected error during direct JSON parsing: {e}")
            
        # Try to match both double-quoted and single-quoted JSON patterns
        json_match = re.search(r'\{[\s\S]*?["\']moves["\']?\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        
        # If that fails, try a more permissive pattern that might match unquoted keys
        if not json_match:
            json_match = re.search(r'\{\s*moves\s*:[\s\S]*?\}', response, re.DOTALL)
            
        if not json_match:
            # No JSON-like structure found
            return None
            
        json_str = json_match.group(0)
        # Use our comprehensive preprocessing function
        json_str = preprocess_json_string(json_str)
        
        # Now try to parse the cleaned JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_error_stats["text_extraction_errors"] += 1
            print(f"Failed to parse JSON after preprocessing: {e}, trying fallback extraction")
            
            # Try to extract just the moves array using fallback method
            return extract_moves_fallback(json_str)
    except json.JSONDecodeError as e:
        json_error_stats["text_extraction_errors"] += 1
        print(f"JSON decode error: {e}, trying fallback extraction")
        
        # Try to extract just the moves array
        return extract_moves_fallback(json_str if 'json_str' in locals() else response)
    except Exception as e:
        json_error_stats["text_extraction_errors"] += 1
        print(f"Unexpected error in JSON extraction: {e}")
        return None

def extract_moves_fallback(json_str):
    """Extract moves from a JSON string using pattern matching.
    
    Args:
        json_str: JSON string that couldn't be parsed normally
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    global json_error_stats
    
    try:
        # Try to extract just the moves array
        moves_array_match = re.search(r'["\']?moves["\']?\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
        if not moves_array_match:
            return None
            
        moves_array = moves_array_match.group(1)
        # Extract both single and double quoted strings
        move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
        valid_moves = [move.upper() for move in move_matches 
                      if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
    
        if valid_moves:
            json_error_stats["fallback_extraction_success"] += 1
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Failed in fallback extraction: {e}")
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