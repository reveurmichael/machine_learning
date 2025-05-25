"""
Utility module for text processing in the Snake game.
Handles processing and formatting of text responses, logging, and file operations.
"""

import os
from datetime import datetime

def process_response_for_display(response):
    """Process the LLM response for display purposes.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Processed response text ready for display
    """
    try:
        processed = response
            
        # Limit to a reasonable length for display
        if len(processed) > 1500:
            processed = processed[:1500] + "...\n(response truncated)"
            
        return processed
    except Exception as e:
        print(f"Error processing response for display: {e}")
        return "Error processing response"

def save_to_file(content, directory, filename):
    """Save content to a file.
    
    Args:
        content: Content to save
        directory: Directory to save to
        filename: Name of the file
        
    Returns:
        The full path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return os.path.abspath(file_path)

def save_experiment_info(args, directory):
    """Save experiment information to a file.
    
    Args:
        args: Command line arguments
        directory: Directory to save to
        
    Returns:
        Path to the saved file
    """
    # Create content with experiment information
    content = f"""Experiment Information
====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Primary LLM (Game Strategy Expert):
- Provider: {args.provider}
- Model: {args.model if args.model else 'Default model for provider'}

Secondary LLM (Formatting Expert):
- Provider: {args.parser_provider if args.parser_provider else args.provider}
- Model: {args.parser_model if args.parser_model else 'Default model for parser provider'}

Game Configuration:
- Max Steps per Game: {args.max_steps}
- Max Consecutive Empty Moves: {args.max_empty_moves}
- Max Games: {args.max_games}
"""
    
    # Save to file
    return save_to_file(content, directory, "info.txt")

def update_experiment_info(directory, game_count, total_score, total_steps, parser_usage_count=0, game_scores=None, empty_steps=0, error_steps=0, json_error_stats=None, max_empty_moves=3):
    """Update the experiment information file with game statistics.
    
    Args:
        directory: Directory containing the info.txt file
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
    file_path = os.path.join(directory, "info.txt")
    
    # Read existing content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Calculate score statistics
    mean_score = total_score / game_count if game_count > 0 else 0
    max_score = 0
    min_score = 0
    
    if game_scores and len(game_scores) > 0:
        max_score = max(game_scores)
        min_score = min(game_scores)
    
    # Calculate empty step statistics
    empty_step_percentage = (empty_steps / total_steps) * 100 if total_steps > 0 else 0
    error_step_percentage = (error_steps / total_steps) * 100 if total_steps > 0 else 0
    valid_steps = total_steps - empty_steps - error_steps
    valid_step_percentage = (valid_steps / total_steps) * 100 if total_steps > 0 else 0
    
    # Add statistics section
    stats = f"""

Game Statistics
==============
Total Games Played: {game_count}
Total Score: {total_score}
Total Steps: {total_steps}
Maximum Score: {max_score}
Minimum Score: {min_score}
Mean Score: {mean_score:.2f}
Average Steps per Game: {total_steps/game_count:.2f}

LLM Response Statistics
======================
Empty Steps (empty moves): {empty_steps} ({empty_step_percentage:.2f}%)
Error Steps (with ERROR in reasoning): {error_steps} ({error_step_percentage:.2f}%)
Valid Steps: {valid_steps} ({valid_step_percentage:.2f}%)
Parser Usage Count: {parser_usage_count}
"""
    
    # Add JSON error statistics if available
    if json_error_stats:
        # Calculate success rate
        success_rate = (json_error_stats["successful_extractions"] / json_error_stats["total_extraction_attempts"]) * 100 if json_error_stats["total_extraction_attempts"] > 0 else 0
        failure_rate = (json_error_stats["failed_extractions"] / json_error_stats["total_extraction_attempts"]) * 100 if json_error_stats["total_extraction_attempts"] > 0 else 0
        
        json_stats = f"""
JSON Parsing Statistics
======================
Total JSON Extraction Attempts: {json_error_stats["total_extraction_attempts"]}
Successful Extractions: {json_error_stats["successful_extractions"]} ({success_rate:.2f}%)
Failed Extractions: {json_error_stats["failed_extractions"]} ({failure_rate:.2f}%)

JSON Error Breakdown:
- JSON Decode Errors: {json_error_stats["json_decode_errors"]}
- Format Validation Errors: {json_error_stats["format_validation_errors"]}
- Code Block Extraction Errors: {json_error_stats["code_block_extraction_errors"]}
- Text Extraction Errors: {json_error_stats["text_extraction_errors"]}
- Fallback Extraction Successes: {json_error_stats["fallback_extraction_success"]}
"""
        stats += json_stats

    stats += f"""
Efficiency Metrics
=================
Apples per Step: {total_score/(total_steps if total_steps > 0 else 1):.4f}
Steps per Game: {total_steps/game_count:.2f}
Valid Move Ratio: {valid_steps/(total_steps if total_steps > 0 else 1):.4f}
"""
    
    # Append statistics to content
    content += stats
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def format_raw_llm_response(raw_response, request_time, response_time, model, provider, 
                          parser_model=None, parser_provider=None, response_duration=None):
    """Format a raw LLM response with metadata.
    
    Args:
        raw_response: The raw response from the primary LLM
        request_time: Time when the request was sent
        response_time: Time when the response was received
        model: The model used
        provider: The provider used
        parser_model: The secondary LLM model (if any)
        parser_provider: The secondary LLM provider (if any)
        response_duration: The duration of the response in seconds (if available)
        
    Returns:
        Formatted response with metadata
    """
    # Format secondary LLM info
    secondary_llm_info = ""
    if parser_provider and parser_provider.lower() != "none":
        secondary_llm_info = f"""SECONDARY LLM Model: {parser_model if parser_model else 'Default model'}
SECONDARY LLM Provider: {parser_provider}"""
    else:
        secondary_llm_info = "SECONDARY LLM: Not used"
    
    # Add response duration if available
    duration_info = ""
    if response_duration is not None:
        duration_info = f"Response Duration: {response_duration:.2f} seconds"
        
    return f"""Timestamp: {response_time}
Request Time: {request_time}
Response Time: {response_time}
{duration_info}
PRIMARY LLM Model: {model}
PRIMARY LLM Provider: {provider}
{secondary_llm_info}

========== PRIMARY LLM RESPONSE (GAME STRATEGY) ==========

{raw_response}
"""

def format_parsed_llm_response(parsed_response, parser_request_time, parser_response_time, parser_model, parser_provider, response_duration=None):
    """Format a parsed LLM response with metadata.
    
    Args:
        parsed_response: The parsed response from the secondary LLM
        parser_request_time: Time when the secondary LLM request was sent
        parser_response_time: Time when the secondary LLM response was received
        parser_model: The secondary LLM model used
        parser_provider: The secondary LLM provider used
        response_duration: The duration of the response in seconds (if available)
        
    Returns:
        Formatted response with metadata
    """
    # Add response duration if available
    duration_info = ""
    if response_duration is not None:
        duration_info = f"Secondary LLM Response Duration: {response_duration:.2f} seconds"
        
    return f"""Timestamp: {parser_response_time}
Secondary LLM Request Time: {parser_request_time}
Secondary LLM Response Time: {parser_response_time}
{duration_info}
SECONDARY LLM Model: {parser_model}
SECONDARY LLM Provider: {parser_provider}

========== SECONDARY LLM RESPONSE (FORMATTED JSON) ==========

{parsed_response}
"""

def generate_game_summary_json(game_count, timestamp, score, steps, next_move, game_parser_usage, 
                         snake_positions_length, last_collision_type, round_count, 
                         primary_model=None, primary_provider=None, parser_model=None, parser_provider=None,
                         json_error_stats=None, max_empty_moves=3, apple_positions=None,
                         avg_response_time=0, avg_secondary_response_time=0, steps_per_apple=0):
    """Generate a game summary in JSON format.
    
    Args:
        game_count: Number of the current game
        timestamp: Current timestamp
        score: Game score
        steps: Number of steps taken
        next_move: Last direction moved
        game_parser_usage: Number of times the secondary LLM was used in this game
        snake_positions_length: Length of the snake
        last_collision_type: Type of collision that ended the game (wall, self, max_steps, empty_moves, error)
        round_count: Number of rounds played
        primary_model: Primary LLM model name
        primary_provider: Primary LLM provider
        parser_model: Secondary LLM model name
        parser_provider: Secondary LLM provider
        json_error_stats: Dictionary containing JSON extraction error statistics
        max_empty_moves: Maximum number of consecutive empty moves before termination
        apple_positions: List of apple positions in format "(x,y)"
        avg_response_time: Average primary LLM response time in seconds
        avg_secondary_response_time: Average secondary LLM response time in seconds
        steps_per_apple: Average steps taken to eat one apple
        
    Returns:
        Dictionary containing game summary data ready for JSON serialization
    """
    # Format game end reason
    if last_collision_type == 'wall':
        game_end_reason = 'Wall collision'
    elif last_collision_type == 'self':
        game_end_reason = 'Self collision'
    elif last_collision_type == 'max_steps':
        game_end_reason = 'Maximum steps reached'
    elif last_collision_type == 'empty_moves':
        game_end_reason = f'{max_empty_moves} consecutive empty moves'
    elif last_collision_type == 'error':
        game_end_reason = 'Game aborted due to error'
    else:
        game_end_reason = 'Unknown'
    
    # Create JSON-serializable dictionary
    summary_data = {
        "game_number": game_count,
        "timestamp": timestamp,
        "score": score,
        "steps": steps,
        "last_direction": next_move,
        "performance_metrics": {
            "apples_per_step": score/(steps if steps > 0 else 1),
            "steps_per_apple": steps_per_apple,
            "final_snake_length": snake_positions_length
        },
        "game_end_reason": game_end_reason,
        "llm_info": {
            "primary_llm": f"{primary_provider} - {primary_model}" if primary_provider else "Unknown",
            "secondary_llm": f"{parser_provider} - {parser_model}" if parser_provider and parser_provider.lower() != "none" else "None",
        },
        "prompt_response_stats": {
            "total_prompts": round_count,
            "parser_usage": game_parser_usage,
            "avg_primary_response_time": avg_response_time,
            "avg_secondary_response_time": avg_secondary_response_time
        }
    }
    
    # Add apple positions if available
    if apple_positions:
        summary_data["apple_positions"] = apple_positions
    
    # Add JSON parsing statistics if available
    if json_error_stats:
        json_success_rate = (json_error_stats["successful_extractions"] / json_error_stats["total_extraction_attempts"] * 100) if json_error_stats["total_extraction_attempts"] > 0 else 0
        summary_data["json_parsing_stats"] = {
            "extraction_attempts": json_error_stats["total_extraction_attempts"],
            "success_rate": json_success_rate,
            "format_validation_errors": json_error_stats["format_validation_errors"],
            "json_decode_errors": json_error_stats.get("json_decode_errors", 0),
            "code_block_extraction_errors": json_error_stats.get("code_block_extraction_errors", 0),
            "text_extraction_errors": json_error_stats.get("text_extraction_errors", 0),
            "fallback_extraction_success": json_error_stats.get("fallback_extraction_success", 0)
        }
    
    return summary_data 