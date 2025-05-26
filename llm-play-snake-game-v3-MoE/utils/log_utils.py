"""
Utility module for logging and data formatting.
Handles saving and formatting of game logs, LLM responses, and game summaries.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from colorama import Fore

def save_to_file(content, directory_or_file_path, filename=None):
    """Save content to a file, creating the directory if it doesn't exist.
    
    This function can be called in two ways:
    1. save_to_file(content, directory, filename)
    2. save_to_file(content, file_path)
    
    Args:
        content: Content to save
        directory_or_file_path: Directory to save to or complete file path
        filename: Name of the file (optional, required if directory is provided)
        
    Returns:
        Path to the saved file
    """
    # Check if we have a complete file path or a directory + filename
    if filename is None:
        # First argument is a complete file path
        file_path = directory_or_file_path
        directory = os.path.dirname(file_path)
    else:
        # First argument is a directory, second is the filename
        directory = directory_or_file_path
        file_path = os.path.join(directory, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        # Check if content is a string or a dict/object that needs to be serialized
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))
    
    return file_path

def save_experiment_info_json(args, directory):
    """Save experiment information to a JSON file.
    
    Args:
        args: Command line arguments (as Namespace or dictionary)
        directory: Directory to save to
        
    Returns:
        Path to the saved file
    """
    # Convert to dictionary if args is a Namespace
    config = vars(args) if not isinstance(args, dict) else args
    
    # Create experiment information in structured JSON format
    info_data = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "primary_llm": {
            "provider": config.get("provider"),
            "model": config.get("model")
        },
        "secondary_llm": {
            "provider": config.get("parser_provider", config.get("provider")),
            "model": config.get("parser_model")
        },
        "game_configuration": {
            "max_steps_per_game": config.get("max_steps"),
            "max_consecutive_empty_moves": config.get("max_empty_moves"),
            "max_games": config.get("max_games")
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to JSON file
    file_path = os.path.join(directory, "info.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2)
    
    return os.path.abspath(file_path)

def update_experiment_info_json(directory, game_count, total_score, total_steps, json_error_stats, parser_usage_count=0, game_scores=None, empty_steps=0, error_steps=0, max_empty_moves=3, token_stats=None):
    """Update the experiment information JSON file with game statistics.
    
    Args:
        directory: Directory containing the info.json file
        game_count: Total number of games played
        total_score: Total score across all games
        total_steps: Total steps taken across all games
        json_error_stats: Dictionary containing JSON extraction error statistics
        parser_usage_count: Number of times the secondary LLM was used
        game_scores: List of individual game scores
        empty_steps: Number of empty steps (moves with empty JSON)
        error_steps: Number of steps with ERROR in reasoning
        max_empty_moves: Maximum number of consecutive empty moves before termination
        token_stats: Dictionary containing token usage statistics
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
    
    # Create a new ordered info_data dictionary
    ordered_info = {}
    
    # 1. Preserve the date and session info from the original data
    ordered_info["date"] = info_data.get("date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 2. Key performance metrics and results (most important information)
    ordered_info["performance_summary"] = {
        "total_score": total_score,
        "mean_score": mean_score,
        "total_steps": total_steps,
        "steps_per_apple": total_score/(total_steps if total_steps > 0 else 1),
        "apples_per_step": total_score/(total_steps if total_steps > 0 else 1),
        "valid_move_ratio": valid_steps/(total_steps if total_steps > 0 else 1),
    }
    
    # 3. Game statistics
    ordered_info["game_statistics"] = {
        "total_games": game_count,
        "total_score": total_score,
        "total_steps": total_steps,
        "mean_score": mean_score,
        "max_score": max_score,
        "min_score": min_score,
        "steps_per_game": total_steps / game_count if game_count > 0 else 0
    }
    
    # 4. Response time and token statistics
    response_stats = {}
    if token_stats:
        # Organize time statistics
        response_time_stats = {}
        
        # Primary LLM response time stats
        if token_stats.get("primary") and "response_times" in token_stats.get("primary", {}):
            primary_times = token_stats["primary"]["response_times"]
            if primary_times:
                response_time_stats["primary_llm"] = {
                    "avg_response_time": sum(primary_times) / len(primary_times),
                    "min_response_time": min(primary_times),
                    "max_response_time": max(primary_times),
                    "total_response_time": sum(primary_times),
                    "response_count": len(primary_times)
                }
        
        # Secondary LLM response time stats
        if token_stats.get("secondary") and "response_times" in token_stats.get("secondary", {}):
            secondary_times = token_stats["secondary"]["response_times"]
            if secondary_times:
                response_time_stats["secondary_llm"] = {
                    "avg_response_time": sum(secondary_times) / len(secondary_times),
                    "min_response_time": min(secondary_times),
                    "max_response_time": max(secondary_times),
                    "total_response_time": sum(secondary_times),
                    "response_count": len(secondary_times)
                }
        
        if response_time_stats:
            response_stats["response_time_stats"] = response_time_stats
        
        # Organize token statistics
        token_usage_stats = {}
        
        # Primary LLM token stats
        if "primary" in token_stats:
            primary_tokens = token_stats["primary"]
            token_usage_stats["primary_llm"] = {
                "total_tokens": primary_tokens.get("total_tokens", 0),
                "total_prompt_tokens": primary_tokens.get("total_prompt_tokens", 0),
                "total_completion_tokens": primary_tokens.get("total_completion_tokens", 0),
                "avg_tokens_per_request": primary_tokens.get("avg_total_tokens", 0),
                "avg_prompt_tokens": primary_tokens.get("avg_prompt_tokens", 0),
                "avg_completion_tokens": primary_tokens.get("avg_completion_tokens", 0)
            }
        
        # Secondary LLM token stats
        if "secondary" in token_stats:
            secondary_tokens = token_stats["secondary"]
            token_usage_stats["secondary_llm"] = {
                "total_tokens": secondary_tokens.get("total_tokens", 0),
                "total_prompt_tokens": secondary_tokens.get("total_prompt_tokens", 0),
                "total_completion_tokens": secondary_tokens.get("total_completion_tokens", 0),
                "avg_tokens_per_request": secondary_tokens.get("avg_total_tokens", 0),
                "avg_prompt_tokens": secondary_tokens.get("avg_prompt_tokens", 0),
                "avg_completion_tokens": secondary_tokens.get("avg_completion_tokens", 0)
            }
        
        if token_usage_stats:
            response_stats["token_usage_stats"] = token_usage_stats
    
    # Add response stats to ordered info if available
    if response_stats:
        ordered_info.update(response_stats)
    
    # 5. JSON parsing statistics if available
    if json_error_stats:
        # Calculate success rate
        total_attempts = json_error_stats.get("total_extraction_attempts", 0)
        successful_extractions = json_error_stats.get("successful_extractions", 0)
        failed_extractions = json_error_stats.get("failed_extractions", 0)
        
        success_rate = (successful_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        failure_rate = (failed_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        
        ordered_info["json_parsing_stats"] = {
            "success_rate": success_rate,
            "total_extraction_attempts": total_attempts,
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "failure_rate": failure_rate
        }
        
        # Detailed JSON parsing stats (less important)
        ordered_info["detailed_json_parsing_stats"] = {
            "json_decode_errors": json_error_stats.get("json_decode_errors", 0),
            "format_validation_errors": json_error_stats.get("format_validation_errors", 0),
            "code_block_extraction_errors": json_error_stats.get("code_block_extraction_errors", 0),
            "text_extraction_errors": json_error_stats.get("text_extraction_errors", 0),
            "fallback_extraction_success": json_error_stats.get("fallback_extraction_success", 0)
        }
    
    # 6. Step statistics
    ordered_info["step_stats"] = {
        "valid_steps": valid_steps,
        "valid_step_percentage": valid_step_percentage,
        "empty_steps": empty_steps,
        "empty_step_percentage": empty_step_percentage,
        "error_steps": error_steps,
        "error_step_percentage": error_step_percentage,
        "max_consecutive_empty_moves": max_empty_moves
    }
    
    # 7. LLM usage statistics
    ordered_info["llm_usage_stats"] = {
        "parser_usage_count": parser_usage_count,
        "parser_usage_per_game": parser_usage_count / game_count if game_count > 0 else 0
    }
    
    # 8. Configuration information (moved lower as it's the same for all runs)
    if "primary_llm" in info_data:
        ordered_info["primary_llm"] = info_data["primary_llm"]
    
    if "secondary_llm" in info_data:
        ordered_info["secondary_llm"] = info_data["secondary_llm"]
    
    if "game_configuration" in info_data:
        ordered_info["game_configuration"] = info_data["game_configuration"]
    
    # 9. Copy any other fields from the original info_data not already included
    for key, value in info_data.items():
        if key not in ordered_info and key != "token_stats" and key != "efficiency_metrics":
            ordered_info[key] = value
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_info, f, indent=2)

def format_raw_llm_response(raw_response, request_time, response_time, model_name, provider, 
                           parser_model=None, parser_provider=None, response_duration=None):
    """Format a raw LLM response with metadata.
    
    Args:
        raw_response: Raw response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_name: Name of the model
        provider: Provider of the model
        parser_model: Name of the parser model (optional)
        parser_provider: Provider of the parser model (optional)
        response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_name,
        "provider": provider,
        "response_duration_seconds": response_duration
    }
    
    if parser_model:
        metadata["parser_model"] = parser_model
    
    if parser_provider:
        metadata["parser_provider"] = parser_provider
        
    # Format as metadata block followed by raw response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- RESPONSE ---\n{raw_response}"
    
    return formatted_response

def format_parsed_llm_response(parsed_response, request_time, response_time, model_name, provider, 
                              response_duration=None):
    """Format a parsed LLM response with metadata.
    
    Args:
        parsed_response: Parsed response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_name: Name of the model
        provider: Provider of the model
        response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_name,
        "provider": provider,
        "response_duration_seconds": response_duration
    }
    
    # Format as metadata block followed by parsed response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- PARSED RESPONSE ---\n{parsed_response}"
    
    return formatted_response

def generate_game_summary_json(game_number, timestamp, score, steps, next_move, parser_usage_count, 
                              snake_length, collision_type, round_count, primary_model=None, 
                              primary_provider=None, parser_model=None, parser_provider=None,
                              json_error_stats=None, max_empty_moves=3, apple_positions=None,
                              avg_response_time=None, avg_secondary_response_time=None,
                              steps_per_apple=None, moves=None, token_stats=None):
    """Generate a JSON summary of a game.
    
    Args:
        game_number: Number of the game
        timestamp: Timestamp of the game
        score: Final score
        steps: Total steps taken
        next_move: Last move made
        parser_usage_count: Number of times the parser was used
        snake_length: Final length of the snake
        collision_type: Type of collision that ended the game
        round_count: Total number of rounds
        primary_model: Primary LLM model name
        primary_provider: Primary LLM provider
        parser_model: Parser LLM model name
        parser_provider: Parser LLM provider
        json_error_stats: Statistics about JSON parsing errors
        max_empty_moves: Maximum allowed empty moves
        apple_positions: List of apple positions
        avg_response_time: Average response time
        avg_secondary_response_time: Average secondary response time
        steps_per_apple: Average steps per apple
        moves: Dictionary of moves by round
        token_stats: Dictionary with token usage statistics
        
    Returns:
        Dictionary containing game summary
    """
    # Create an ordered dictionary with important information at the top
    summary = {}
    
    # 1. Core game metrics (most important first)
    summary["score"] = score
    summary["steps"] = steps
    summary["game_end_reason"] = collision_type
    summary["snake_length"] = snake_length
    
    # 2. LLM information
    if primary_provider:
        summary["primary_provider"] = primary_provider
    if primary_model:
        summary["primary_model"] = primary_model
    if parser_provider:
        summary["parser_provider"] = parser_provider
    if parser_model:
        summary["parser_model"] = parser_model
    
    # 3. Performance metrics
    if steps_per_apple is not None:
        summary["performance_metrics"] = {
            "steps_per_apple": steps_per_apple
        }
    
    # 4. Response time statistics
    if avg_response_time is not None or avg_secondary_response_time is not None or token_stats is not None:
        prompt_response_stats = {}
        if avg_response_time is not None:
            prompt_response_stats["avg_primary_response_time"] = avg_response_time
        if avg_secondary_response_time is not None:
            prompt_response_stats["avg_secondary_response_time"] = avg_secondary_response_time
        
        # Add token statistics if available
        if token_stats:
            if "total_prompt_tokens" in token_stats:
                prompt_response_stats["total_prompt_tokens"] = token_stats["total_prompt_tokens"]
            if "total_completion_tokens" in token_stats:
                prompt_response_stats["total_completion_tokens"] = token_stats["total_completion_tokens"]
            if "total_tokens" in token_stats:
                prompt_response_stats["total_tokens"] = token_stats["total_tokens"]
            if "avg_prompt_tokens" in token_stats:
                prompt_response_stats["avg_prompt_tokens"] = token_stats["avg_prompt_tokens"]
            if "avg_completion_tokens" in token_stats:
                prompt_response_stats["avg_completion_tokens"] = token_stats["avg_completion_tokens"]
            if "avg_total_tokens" in token_stats:
                prompt_response_stats["avg_total_tokens"] = token_stats["avg_total_tokens"]
            
        summary["prompt_response_stats"] = prompt_response_stats
    
    # 5. JSON extraction success metrics
    if json_error_stats:
        # Only include the most important metrics in the top-level
        successful = json_error_stats.get("successful_extractions", 0)
        total = json_error_stats.get("total_extraction_attempts", 0)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        summary["json_parsing_stats"] = {
            "success_rate": success_rate,
            "successful_extractions": successful,
            "total_extraction_attempts": total
        }
    
    # 6. Game metadata (medium importance)
    summary["game_number"] = game_number
    summary["timestamp"] = timestamp
    summary["last_move"] = next_move
    summary["round_count"] = round_count
    summary["max_empty_moves"] = max_empty_moves
    
    if parser_usage_count > 0:
        summary["parser_usage_count"] = parser_usage_count
    
    # 7. Detailed JSON error statistics (lower importance)
    if json_error_stats:
        detailed_json_stats = {k: v for k, v in json_error_stats.items() 
                             if k not in ["successful_extractions", "total_extraction_attempts", "success_rate"]}
        if detailed_json_stats:
            summary["detailed_json_parsing_stats"] = detailed_json_stats
    
    # 8. Group apple positions and moves together by round (at the bottom of the JSON)
    if apple_positions or moves:
        rounds_data = {}
        
        # Determine the number of rounds
        num_rounds = max(
            len(apple_positions) if apple_positions else 0,
            len(moves) if moves else 0,
            round_count if round_count else 0
        )
        
        for i in range(num_rounds):
            round_key = f"round_{i+1}"
            round_info = {}
            
            # Add apple position for this round if available
            if apple_positions and i < len(apple_positions):
                round_info["apple_position"] = apple_positions[i]
            
            # Add moves for this round if available
            if moves and i < len(moves):
                round_info["moves"] = moves[i]
            
            # Only add round data if we have either apple position or moves
            if round_info:
                rounds_data[round_key] = round_info
        
        # Add rounds data to summary (at the bottom)
        if rounds_data:
            summary["rounds_data"] = rounds_data
    
    return summary

def load_game_from_file(game_file_path):
    """Load game data directly from a game summary file.
    
    Args:
        game_file_path: Path to the game summary file
        
    Returns:
        Tuple containing the moves list and game info
    """
    try:
        with open(game_file_path, "r") as f:
            data = json.load(f)
            
            # Extract moves data
            moves_data = []
            
            # Check if using old format with "moves" array
            if "moves" in data:
                # Check if moves is a list
                if isinstance(data["moves"], list):
                    moves_data = data["moves"]
                # Check if moves is a dictionary with round keys
                elif isinstance(data["moves"], dict):
                    # Convert dictionary of moves by round to flat list of moves
                    moves_list = []
                    # Sort round keys numerically
                    sorted_rounds = sorted(data["moves"].keys(), 
                                         key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                    for round_key in sorted_rounds:
                        move = data["moves"][round_key]
                        moves_list.append(move)
                    moves_data = moves_list
            
            # Check if using new format with "rounds_data"
            elif "rounds_data" in data:
                # Extract moves from rounds_data and flatten into a list
                moves_list = []
                # Sort round keys numerically
                sorted_rounds = sorted(data["rounds_data"].keys(),
                                     key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                for round_key in sorted_rounds:
                    round_data = data["rounds_data"][round_key]
                    if "moves" in round_data:
                        moves_list.append(round_data["moves"])
                moves_data = moves_list
            
            # Extract basic game info for display
            game_info = {
                "score": data.get("score", 0),
                "steps": data.get("steps", 0),
                "game_end_reason": data.get("game_end_reason", "Unknown"),
                "primary_model": data.get("primary_model", "Unknown"),
                "parser_model": data.get("parser_model", "Unknown")
            }
            
            return moves_data, game_info
    except Exception as e:
        print(f"{Fore.RED}Error loading game data from {game_file_path}: {e}{Fore.RESET}")
        return [], {} 