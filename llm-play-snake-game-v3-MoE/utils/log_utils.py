"""
Utility module for logging and data formatting.
Handles saving and formatting of game logs, LLM responses, and game summaries.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from utils.file_utils import save_to_file

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
        game_number: Game number
        timestamp: Timestamp of the game
        score: Final score
        steps: Number of steps
        next_move: Last move made
        parser_usage_count: Number of times the parser was used
        snake_length: Final length of the snake
        collision_type: Type of collision that ended the game
        round_count: Number of rounds
        primary_model: Name of the primary model (optional)
        primary_provider: Provider of the primary model (optional)
        parser_model: Name of the parser model (optional)
        parser_provider: Provider of the parser model (optional)
        json_error_stats: Statistics about JSON parsing errors (optional)
        max_empty_moves: Maximum number of empty moves allowed (optional)
        apple_positions: List of apple positions (optional)
        avg_response_time: Average response time in seconds (optional)
        avg_secondary_response_time: Average secondary response time in seconds (optional)
        steps_per_apple: Average steps per apple (optional)
        moves: List of moves made during the game (optional)
        
    Returns:
        JSON summary of the game
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
    
    # Add moves if available
    if moves:
        summary["moves"] = moves
    
    return summary 