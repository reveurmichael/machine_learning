"""
Utility module for JSON processing in the Snake game.
Consolidates JSON extraction and validation functions used by multiple components.
"""

import os
import json
import re
import numpy as np
from datetime import datetime
import glob

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# NumPyJSONEncoder moved from core/game_data.py to fix circular imports
class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
            "model": None if (args.parser_provider and args.parser_provider.lower() == "none") else (args.parser_model if args.parser_model else 'Default model for parser provider')
        },
        "game_configuration": {
            "max_steps_per_game": args.max_steps,
            "max_consecutive_empty_moves": args.max_empty_moves,
            "max_consecutive_errors_allowed": args.max_consecutive_errors_allowed,
            "max_games": args.max_games,
            "use_gui": not args.no_gui
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to JSON file
    file_path = os.path.join(directory, "summary.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, cls=NumpyEncoder)
    
    return os.path.abspath(file_path)

def update_experiment_info_json(log_dir, **kwargs):
    """Update the experiment info JSON file with additional statistics.
    
    Args:
        log_dir: Directory containing the experiment logs
        **kwargs: Additional data to add to the JSON file
    """
    summary_path = os.path.join(log_dir, "summary.json")
    data = {}
    
    # Load existing data if file exists
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading existing summary.json: {e}")
    
    # Extract continuation information to place at the bottom
    continuation_info = {}
    is_continuation = kwargs.get('is_continuation', False)
    
    # If this is a continuation or we have continuation data
    if is_continuation or data.get('continuation_info'):
        # Initialize continuation info structure if it doesn't exist
        if not data.get('continuation_info'):
            data['continuation_info'] = {
                'is_continued': False,
                'continuation_count': 0,
                'continuation_timestamps': [],
                'session_metadata': []
            }
        
        continuation_info = data.get('continuation_info', {}).copy()
        
        # Update continuation info
        if is_continuation:
            continuation_info['is_continued'] = True
            continuation_info['continuation_count'] = continuation_info.get('continuation_count', 0) + 1
            
            # Add timestamp for this continuation
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'continuation_timestamps' not in continuation_info:
                continuation_info['continuation_timestamps'] = []
            continuation_info['continuation_timestamps'].append(current_timestamp)
            
            # Add session metadata
            if 'session_metadata' not in continuation_info:
                continuation_info['session_metadata'] = []
            
            # Add metadata about this continuation session
            session_meta = {
                'timestamp': current_timestamp,
                'games_before_continuation': kwargs.get('game_count', 0),
                'score_before_continuation': kwargs.get('total_score', 0),
                'steps_before_continuation': kwargs.get('total_steps', 0)
            }
            continuation_info['session_metadata'].append(session_meta)
        
        # Remove continuation info from kwargs to handle it separately
        for key in ['is_continuation', 'continuation_count', 'continuation_timestamps']:
            if key in kwargs:
                del kwargs[key]
    
    # Collect token usage and response time statistics from game files
    game_files = glob.glob(os.path.join(log_dir, "game_*.json"))
    token_stats = {
        "primary_llm": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "avg_tokens_per_request": 0,
            "avg_prompt_tokens": 0,
            "avg_completion_tokens": 0
        },
        "secondary_llm": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "avg_tokens_per_request": 0,
            "avg_prompt_tokens": 0,
            "avg_completion_tokens": 0
        }
    }
    
    response_time_stats = {
        "primary_llm": {
            "total_response_time": 0,
            "avg_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0
        },
        "secondary_llm": {
            "total_response_time": 0,
            "avg_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0
        }
    }
    
    primary_requests = 0
    secondary_requests = 0
    
    for game_file in game_files:
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Collect token stats
            if 'token_stats' in game_data:
                # Primary LLM
                if 'primary' in game_data['token_stats']:
                    primary = game_data['token_stats']['primary']
                    token_stats['primary_llm']['total_tokens'] += primary.get('total_tokens', 0)
                    token_stats['primary_llm']['total_prompt_tokens'] += primary.get('total_prompt_tokens', 0)
                    token_stats['primary_llm']['total_completion_tokens'] += primary.get('total_completion_tokens', 0)
                    
                    # Count requests based on primary response times
                    if 'primary_response_times' in game_data:
                        primary_requests += len(game_data['primary_response_times'])
                
                # Secondary LLM
                if 'secondary' in game_data['token_stats']:
                    secondary = game_data['token_stats']['secondary']
                    token_stats['secondary_llm']['total_tokens'] += secondary.get('total_tokens', 0)
                    token_stats['secondary_llm']['total_completion_tokens'] += secondary.get('total_completion_tokens', 0)
                    token_stats['secondary_llm']['total_prompt_tokens'] += secondary.get('total_prompt_tokens', 0)
                    
                    # Count requests based on secondary response times
                    if 'secondary_response_times' in game_data:
                        secondary_requests += len(game_data['secondary_response_times'])
            
            # Collect response time stats
            if 'prompt_response_stats' in game_data:
                stats = game_data['prompt_response_stats']
                
                # Primary LLM
                avg_primary = stats.get('avg_primary_response_time', 0)
                min_primary = stats.get('min_primary_response_time', 0)
                max_primary = stats.get('max_primary_response_time', 0)
                
                if avg_primary > 0:
                    primary_count = len(game_data.get('primary_response_times', []))
                    response_time_stats['primary_llm']['total_response_time'] += avg_primary * primary_count
                    response_time_stats['primary_llm']['min_response_time'] = min(
                        response_time_stats['primary_llm']['min_response_time'], 
                        min_primary
                    ) if min_primary > 0 else response_time_stats['primary_llm']['min_response_time']
                    response_time_stats['primary_llm']['max_response_time'] = max(
                        response_time_stats['primary_llm']['max_response_time'], 
                        max_primary
                    )
                
                # Secondary LLM
                avg_secondary = stats.get('avg_secondary_response_time', 0)
                min_secondary = stats.get('min_secondary_response_time', 0)
                max_secondary = stats.get('max_secondary_response_time', 0)
                
                if avg_secondary > 0:
                    secondary_count = len(game_data.get('secondary_response_times', []))
                    response_time_stats['secondary_llm']['total_response_time'] += avg_secondary * secondary_count
                    response_time_stats['secondary_llm']['min_response_time'] = min(
                        response_time_stats['secondary_llm']['min_response_time'], 
                        min_secondary
                    ) if min_secondary > 0 else response_time_stats['secondary_llm']['min_response_time']
                    response_time_stats['secondary_llm']['max_response_time'] = max(
                        response_time_stats['secondary_llm']['max_response_time'], 
                        max_secondary
                    )
                
        except Exception as e:
            print(f"Error processing game file {game_file} for stats: {e}")
    
    # Calculate averages
    if primary_requests > 0:
        token_stats['primary_llm']['avg_tokens_per_request'] = token_stats['primary_llm']['total_tokens'] / primary_requests
        token_stats['primary_llm']['avg_prompt_tokens'] = token_stats['primary_llm']['total_prompt_tokens'] / primary_requests
        token_stats['primary_llm']['avg_completion_tokens'] = token_stats['primary_llm']['total_completion_tokens'] / primary_requests
        response_time_stats['primary_llm']['avg_response_time'] = response_time_stats['primary_llm']['total_response_time'] / primary_requests
    
    if secondary_requests > 0:
        token_stats['secondary_llm']['avg_tokens_per_request'] = token_stats['secondary_llm']['total_tokens'] / secondary_requests
        token_stats['secondary_llm']['avg_prompt_tokens'] = token_stats['secondary_llm']['total_prompt_tokens'] / secondary_requests
        token_stats['secondary_llm']['avg_completion_tokens'] = token_stats['secondary_llm']['total_completion_tokens'] / secondary_requests
        response_time_stats['secondary_llm']['avg_response_time'] = response_time_stats['secondary_llm']['total_response_time'] / secondary_requests
    
    # Reset min response time if it wasn't set
    if response_time_stats['primary_llm']['min_response_time'] == float('inf'):
        response_time_stats['primary_llm']['min_response_time'] = 0
    if response_time_stats['secondary_llm']['min_response_time'] == float('inf'):
        response_time_stats['secondary_llm']['min_response_time'] = 0
    
    # Add token and response time stats to data
    data['token_usage_stats'] = token_stats
    data['response_time_stats'] = response_time_stats
    
    # Update nested dictionaries intelligently (don't completely overwrite them)
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in data and isinstance(data[key], dict):
            # Recursively update the nested dictionary
            deep_update_dict(data[key], value)
        else:
            # Direct update for non-dictionary values or new keys
            data[key] = value
    
    # Add continuation info at the end to ensure it appears at the bottom
    if continuation_info:
        data['continuation_info'] = continuation_info
    
    # Save updated data
    try:
        with open(summary_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumPyJSONEncoder)
        print(f"📝 Experiment information saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary.json: {e}")

def deep_update_dict(original, update):
    """Recursively update a dictionary without overwriting nested dictionaries.
    
    Args:
        original: Original dictionary to update
        update: Dictionary with new values to apply
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update_dict(original[key], value)
        else:
            original[key] = value

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
            json_error_stats["fallback_extraction_success"] += 1
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Move extraction error: {e}")
        return None

# Alias for backward compatibility
extract_moves_fallback = extract_moves_pattern

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
            "failed_extractions": 0
        }
    }
    
    # Extract data from each game file
    for game_file in sorted(game_files, key=lambda x: int(re.search(r'game_?(\d+)', x).group(1))):
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Basic stats
            aggregated_stats["game_statistics"]["total_games"] += 1
            aggregated_stats["game_statistics"]["total_score"] += game_data.get("score", 0)
            aggregated_stats["game_statistics"]["scores"].append(game_data.get("score", 0))
            aggregated_stats["game_statistics"]["total_steps"] += game_data.get("steps", 0)
            
            # Time stats
            if "time_stats" in game_data:
                time_stats = game_data["time_stats"]
                aggregated_stats["time_statistics"]["total_llm_communication_time"] += time_stats.get("llm_communication_time", 0)
                aggregated_stats["time_statistics"]["total_game_movement_time"] += time_stats.get("game_movement_time", 0)
                aggregated_stats["time_statistics"]["total_waiting_time"] += time_stats.get("waiting_time", 0)
            
            # Token stats
            if "token_stats" in game_data:
                token_stats = game_data["token_stats"]
                
                # Primary LLM
                if "primary" in token_stats:
                    primary = token_stats["primary"]
                    aggregated_stats["token_usage_stats"]["primary_llm"]["total_tokens"] += primary.get("total_tokens", 0)
                    aggregated_stats["token_usage_stats"]["primary_llm"]["total_prompt_tokens"] += primary.get("total_prompt_tokens", 0)
                    aggregated_stats["token_usage_stats"]["primary_llm"]["total_completion_tokens"] += primary.get("total_completion_tokens", 0)
                
                # Secondary LLM
                if "secondary" in token_stats:
                    secondary = token_stats["secondary"]
                    aggregated_stats["token_usage_stats"]["secondary_llm"]["total_tokens"] += secondary.get("total_tokens", 0)
                    aggregated_stats["token_usage_stats"]["secondary_llm"]["total_prompt_tokens"] += secondary.get("total_prompt_tokens", 0)
                    aggregated_stats["token_usage_stats"]["secondary_llm"]["total_completion_tokens"] += secondary.get("total_completion_tokens", 0)
            
            # Step stats
            if "step_stats" in game_data:
                step_stats = game_data["step_stats"]
                aggregated_stats["step_stats"]["empty_steps"] += step_stats.get("empty_steps", 0)
                aggregated_stats["step_stats"]["error_steps"] += step_stats.get("error_steps", 0)
                aggregated_stats["step_stats"]["valid_steps"] += step_stats.get("valid_steps", 0)
                aggregated_stats["step_stats"]["invalid_reversals"] += step_stats.get("invalid_reversals", 0)
            
            # JSON parsing stats
            if "json_parsing_stats" in game_data:
                json_stats = game_data["json_parsing_stats"]
                aggregated_stats["json_parsing_stats"]["total_extraction_attempts"] += json_stats.get("total_extraction_attempts", 0)
                aggregated_stats["json_parsing_stats"]["successful_extractions"] += json_stats.get("successful_extractions", 0)
                aggregated_stats["json_parsing_stats"]["failed_extractions"] += json_stats.get("failed_extractions", 0)
                
        except Exception as e:
            print(f"Error processing game file {game_file}: {e}")
    
    # Calculate derived statistics
    if aggregated_stats["game_statistics"]["total_games"] > 0:
        scores = aggregated_stats["game_statistics"]["scores"]
        aggregated_stats["game_statistics"]["mean_score"] = sum(scores) / len(scores)
        aggregated_stats["game_statistics"]["max_score"] = max(scores) if scores else 0
        aggregated_stats["game_statistics"]["min_score"] = min(scores) if scores else 0
    
    # Calculate steps per game
    if aggregated_stats["game_statistics"]["total_games"] > 0:
        aggregated_stats["game_statistics"]["steps_per_game"] = (
            aggregated_stats["game_statistics"]["total_steps"] / 
            aggregated_stats["game_statistics"]["total_games"]
        )
    
    # Calculate steps per apple
    if aggregated_stats["game_statistics"]["total_score"] > 0:
        aggregated_stats["game_statistics"]["steps_per_apple"] = (
            aggregated_stats["game_statistics"]["total_steps"] / 
            aggregated_stats["game_statistics"]["total_score"]
        )
        
        aggregated_stats["game_statistics"]["apples_per_step"] = (
            aggregated_stats["game_statistics"]["total_score"] / 
            aggregated_stats["game_statistics"]["total_steps"]
        )
    
    # Calculate valid move ratio
    total_steps = aggregated_stats["game_statistics"]["total_steps"]
    if total_steps > 0:
        aggregated_stats["game_statistics"]["valid_move_ratio"] = (
            aggregated_stats["step_stats"]["valid_steps"] / total_steps * 100
        )
    
    # JSON parsing success rate
    total_attempts = aggregated_stats["json_parsing_stats"]["total_extraction_attempts"]
    if total_attempts > 0:
        aggregated_stats["json_parsing_stats"]["success_rate"] = (
            aggregated_stats["json_parsing_stats"]["successful_extractions"] / total_attempts * 100
        )
        aggregated_stats["json_parsing_stats"]["failure_rate"] = (
            aggregated_stats["json_parsing_stats"]["failed_extractions"] / total_attempts * 100
        )
    
    return aggregated_stats 