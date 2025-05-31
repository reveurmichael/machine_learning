"""
Utility module for logging and data formatting.
Handles saving and formatting of game logs, LLM responses, and game summaries.
"""

import json

def format_raw_llm_response(raw_response, request_time, response_time, model_info):
    """Format a raw LLM response with metadata.
    
    Args:
        raw_response: Raw response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_info: Dictionary containing model details:
            - model_name: Name of the model
            - provider: Provider of the model
            - parser_model: Name of the parser model (optional)
            - parser_provider: Provider of the parser model (optional)
            - response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_info.get("model_name"),
        "provider": model_info.get("provider"),
        "response_duration_seconds": model_info.get("response_duration")
    }
    
    if "parser_model" in model_info:
        metadata["parser_model"] = model_info["parser_model"]
    
    if "parser_provider" in model_info:
        metadata["parser_provider"] = model_info["parser_provider"]
        
    # Format as metadata block followed by raw response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- RESPONSE ---\n{raw_response}"
    
    return formatted_response

def format_parsed_llm_response(parsed_response, request_time, response_time, model_info):
    """Format a parsed LLM response with metadata.
    
    Args:
        parsed_response: Parsed response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_info: Dictionary containing model details:
            - model_name: Name of the model
            - provider: Provider of the model
            - response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_info.get("model_name"),
        "provider": model_info.get("provider"),
        "response_duration_seconds": model_info.get("response_duration")
    }
    
    # Format as metadata block followed by parsed response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- PARSED RESPONSE ---\n{parsed_response}"
    
    return formatted_response

def generate_game_summary_json(game_data):
    """Generate a JSON summary of a game.
    
    Args:
        game_data: Dictionary containing game information:
            - game_number: Game number
            - timestamp: Timestamp of the game
            - score: Final score
            - steps: Number of steps
            - next_move: Last move made
            - parser_usage_count: Number of times the parser was used
            - snake_length: Final length of the snake
            - collision_type: Type of collision that ended the game
            - round_count: Number of rounds
            - primary_model: Name of the primary model (optional)
            - primary_provider: Provider of the primary model (optional)
            - parser_model: Name of the parser model (optional)
            - parser_provider: Provider of the parser model (optional)
            - json_error_stats: Statistics about JSON parsing errors (optional)
            - max_empty_moves_allowed: Maximum number of empty moves allowed (optional)
            - apple_positions: List of apple positions (optional)
            - avg_response_time: Average response time in seconds (optional)
            - avg_secondary_response_time: Average secondary response time in seconds (optional)
            - steps_per_apple: Average steps per apple (optional)
            - moves: List of moves made during the game (optional)
        
    Returns:
        JSON summary of the game
    """
    summary = {
        "game_number": game_data.get("game_number"),
        "timestamp": game_data.get("timestamp"),
        "score": game_data.get("score"),
        "steps": game_data.get("steps"),
        "last_move": game_data.get("next_move"),
        "game_end_reason": game_data.get("collision_type"),
        "max_empty_moves_allowed": game_data.get("max_empty_moves_allowed"),
        "max_consecutive_errors_allowed": game_data.get("max_consecutive_errors_allowed"),
        "snake_length": game_data.get("snake_length"),
        "round_count": game_data.get("round_count")
    }
    
    # Add optional fields
    if game_data.get("primary_model"):
        summary["primary_model"] = game_data["primary_model"]
    
    if game_data.get("primary_provider"):
        summary["primary_provider"] = game_data["primary_provider"]
    
    if game_data.get("parser_model"):
        summary["parser_model"] = game_data["parser_model"]
    
    if game_data.get("parser_provider"):
        summary["parser_provider"] = game_data["parser_provider"]
    
    if game_data.get("parser_usage_count", 0) > 0:
        summary["parser_usage_count"] = game_data["parser_usage_count"]
    
    # Add JSON error statistics if available
    if game_data.get("json_error_stats"):
        summary["json_parsing_stats"] = game_data["json_error_stats"]
    
    # Add apple positions if available
    if game_data.get("apple_positions"):
        summary["apple_positions"] = game_data["apple_positions"]
    
    # Add response time statistics if available
    if game_data.get("avg_response_time") is not None or game_data.get("avg_secondary_response_time") is not None:
        prompt_response_stats = {}
        
        if game_data.get("avg_response_time") is not None:
            prompt_response_stats["avg_primary_response_time"] = game_data["avg_response_time"]
        
        if game_data.get("avg_secondary_response_time") is not None:
            prompt_response_stats["avg_secondary_response_time"] = game_data["avg_secondary_response_time"]
        
        summary["prompt_response_stats"] = prompt_response_stats
    
    # Add performance metrics if available
    if game_data.get("steps_per_apple") is not None:
        summary["performance_metrics"] = {
            "steps_per_apple": game_data["steps_per_apple"]
        }
    
    # Add moves if available
    if game_data.get("moves"):
        summary["moves"] = game_data["moves"]
    
    return summary 