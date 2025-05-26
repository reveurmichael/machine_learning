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
    
    # Create or update game statistics section
    info_data["game_statistics"] = {
        "total_games": game_count,
        "total_score": total_score,
        "total_steps": total_steps,
        "max_score": max_score,
        "min_score": min_score,
        "mean_score": mean_score,
        "average_steps_per_game": total_steps/game_count if game_count > 0 else 0
    }
    
    # Create or update response statistics section
    info_data["response_statistics"] = {
        "empty_steps": empty_steps,
        "empty_step_percentage": empty_step_percentage,
        "error_steps": error_steps,
        "error_step_percentage": error_step_percentage,
        "valid_steps": valid_steps,
        "valid_step_percentage": valid_step_percentage,
        "parser_usage_count": parser_usage_count
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

def preprocess_json_string(json_str):
    """Preprocess a JSON string to handle common LLM-generated issues.
    
    Args:
        json_str: JSON string that may contain syntax issues
        
    Returns:
        Preprocessed JSON string that's more likely to be valid
    """
    # Handle single quotes
    # First, escape any existing double quotes to avoid conflicts
    processed = json_str.replace('\\"', '_ESCAPED_DOUBLE_QUOTE_')
    # Convert single quotes to double quotes (but not those within already double-quoted strings)
    processed = re.sub(r"(?<!\\)'", '"', processed)
    # Restore escaped double quotes
    processed = processed.replace('_ESCAPED_DOUBLE_QUOTE_', '\\"')
    
    # Handle unquoted keys (word characters followed by colon)
    processed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', processed)
    
    # Remove trailing commas
    processed = re.sub(r',\s*([}\]])', r'\1', processed)
    
    return processed

def validate_json_format(json_data):
    """Validate that JSON data follows the required format for snake moves.
    
    Args:
        json_data: Parsed JSON data
        
    Returns:
        Boolean indicating if the JSON follows the required format
    """
    global json_error_stats
    
    # Check that json_data is a dict with the required keys
    if not isinstance(json_data, dict):
        json_error_stats["format_validation_errors"] += 1
        return False
        
    # Check for required keys
    if "moves" not in json_data or "reasoning" not in json_data:
        json_error_stats["format_validation_errors"] += 1
        return False
        
    # Check that moves is a list
    if not isinstance(json_data["moves"], list):
        json_error_stats["format_validation_errors"] += 1
        return False
        
    # Check that each move is a valid direction
    valid_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for move in json_data["moves"]:
        if not isinstance(move, str) or move.upper() not in valid_directions:
            json_error_stats["format_validation_errors"] += 1
            return False
            
    # Check that reasoning is a string
    if not isinstance(json_data["reasoning"], str):
        json_error_stats["format_validation_errors"] += 1
        return False
        
    return True

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    global json_error_stats
    
    try:
        # Look for JSON code block which is common in LLM responses
        json_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response, re.DOTALL)
        if not json_block_match:
            return None
            
        json_str = json_block_match.group(1)
        # Use our comprehensive preprocessing function
        json_str = preprocess_json_string(json_str)
        
        data = json.loads(json_str)
        return data
    except Exception as e:
        json_error_stats["code_block_extraction_errors"] += 1
        print(f"Failed to parse JSON code block: {e}")
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
    """Extract moves from a JSON string as a fallback method.
    
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
    """Extract moves from array notation in text.
    
    Args:
        response: LLM response text
        
    Returns:
        List of extracted moves, or empty list if none found
    """
    moves = []
    
    # First try a more comprehensive approach to find arrays of direction strings
    # Support both single and double quotes
    array_match = re.search(r'\[\s*(["\'](?:UP|DOWN|LEFT|RIGHT)["\'](?:\s*,\s*["\'](?:UP|DOWN|LEFT|RIGHT)["\'])*)\s*\]', 
                          response, re.IGNORECASE | re.DOTALL)
    
    if array_match:
        # Extract all quoted strings (both single and double quotes)
        directions = re.findall(r'["\']([^"\']+)["\']', array_match.group(1))
        if directions:
            moves = [move.upper() for move in directions 
                   if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            return moves
    
    # Fallback: Look for arrays of directions in quotes (original method extended for single quotes)
    move_arrays = re.findall(r'\[\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*\s*\]', response)
    if move_arrays:
        for move_group in move_arrays:
            for move in move_group:
                if move and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    moves.append(move.upper())
                    
    return moves 