"""
Utility module for logging and data formatting.
Handles saving and formatting of game logs, LLM responses, and game summaries.
"""

import os
import json
from datetime import datetime
from pathlib import Path

def save_to_file(content, directory, filename):
    """Save content to a file, creating the directory if it doesn't exist.
    
    Args:
        content: Content to save
        directory: Directory to save to
        filename: Name of the file
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create file path
    file_path = os.path.join(directory, filename)
    
    # Save content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path

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

def update_experiment_info_json(directory, game_count, total_score, total_steps, json_error_stats, parser_usage_count=0, game_scores=None, empty_steps=0, error_steps=0, max_empty_moves=3):
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
        total_attempts = json_error_stats.get("total_extraction_attempts", 0)
        successful_extractions = json_error_stats.get("successful_extractions", 0)
        failed_extractions = json_error_stats.get("failed_extractions", 0)
        
        success_rate = (successful_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        failure_rate = (failed_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        
        info_data["json_parsing_stats"] = {
            "total_extraction_attempts": total_attempts,
            "successful_extractions": successful_extractions,
            "success_rate": success_rate,
            "failed_extractions": failed_extractions,
            "failure_rate": failure_rate,
            "json_decode_errors": json_error_stats.get("json_decode_errors",0),
            "format_validation_errors": json_error_stats.get("format_validation_errors",0),
            "code_block_extraction_errors": json_error_stats.get("code_block_extraction_errors",0),
            "text_extraction_errors": json_error_stats.get("text_extraction_errors",0),
            "fallback_extraction_success": json_error_stats.get("fallback_extraction_success",0)
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
                              steps_per_apple=None, moves=None):
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
        
    Returns:
        Dictionary containing game summary
    """
    summary = {
        "game_number": game_number,
        "timestamp": timestamp,
        "score": score,
        "steps": steps,
        "last_move": next_move,
        "game_end_reason": collision_type,
        "max_empty_moves": max_empty_moves,
        "snake_length": snake_length,
        "round_count": round_count
    }
    
    # Add optional fields
    if primary_model:
        summary["primary_model"] = primary_model
    
    if primary_provider:
        summary["primary_provider"] = primary_provider
    
    if parser_model:
        summary["parser_model"] = parser_model
    
    if parser_provider:
        summary["parser_provider"] = parser_provider
    
    if parser_usage_count > 0:
        summary["parser_usage_count"] = parser_usage_count
    
    # Add JSON error statistics if available
    if json_error_stats:
        summary["json_parsing_stats"] = json_error_stats
    
    # Add apple positions if available
    if apple_positions:
        summary["apple_positions"] = apple_positions
    
    # Add response time statistics if available
    if avg_response_time is not None or avg_secondary_response_time is not None:
        prompt_response_stats = {}
        
        if avg_response_time is not None:
            prompt_response_stats["avg_primary_response_time"] = avg_response_time
        
        if avg_secondary_response_time is not None:
            prompt_response_stats["avg_secondary_response_time"] = avg_secondary_response_time
        
        summary["prompt_response_stats"] = prompt_response_stats
    
    # Add performance metrics if available
    if steps_per_apple is not None:
        summary["performance_metrics"] = {
            "steps_per_apple": steps_per_apple
        }
    
    # Add moves if available, organized by round
    if moves:
        # Convert moves to a dictionary with round numbers as keys
        moves_by_round = {}
        for round_num, move in enumerate(moves, 1):
            moves_by_round[f"round_{round_num}"] = move
        summary["moves"] = moves_by_round
    
    return summary 