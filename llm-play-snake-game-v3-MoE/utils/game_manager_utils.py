"""
Utility module for game management.
Handles common game management tasks such as processing game over states,
error handling, and statistics reporting.
"""

import os
import json
import traceback
import pygame
from colorama import Fore
from datetime import datetime
from pathlib import Path

from utils.log_utils import generate_game_summary_json
from utils.json_utils import get_json_error_stats

def _save_game_summary(game, log_dir, args, game_number, round_count, 
                       collision_type, current_game_moves, game_parser_usage):
    """Helper function to generate and save the game summary JSON file."""
    # Get apple positions history in structured format
    apple_positions = []
    for pos in game.apple_positions_history:
        apple_positions.append({
            "x": int(pos[0]),
            "y": int(pos[1])
        })
    
    # Get performance metrics
    avg_response_time = game.get_average_response_time()
    avg_secondary_response_time = game.get_average_secondary_response_time()
    steps_per_apple = game.get_steps_per_apple()
    
    # Generate JSON summary
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_summary_data = generate_game_summary_json(
        game_number=game_number,
        timestamp=now,
        score=game.score,
        steps=game.steps,
        next_move=collision_type,  # Pass a more descriptive end reason/last move
        parser_usage_count=game_parser_usage,
        snake_length=len(game.snake_positions),
        collision_type=collision_type,
        round_count=round_count,
        primary_model=args.model,
        primary_provider=args.provider,
        parser_model=args.parser_model,
        parser_provider=args.parser_provider,
        json_error_stats=get_json_error_stats(),
        max_empty_moves=args.max_empty_moves,
        apple_positions=apple_positions,
        avg_response_time=avg_response_time,
        avg_secondary_response_time=avg_secondary_response_time,
        steps_per_apple=steps_per_apple,
        moves=current_game_moves
    )
    
    # Save JSON summary
    json_path = os.path.join(log_dir, f"game{game_number}_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary_data, f, indent=2)
    print(Fore.GREEN + f"📝 JSON summary saved to {json_path}")

def check_max_steps(game, max_steps):
    """Check if the game has exceeded the maximum number of steps.
    
    Args:
        game: Game instance
        max_steps: Maximum number of steps allowed
        
    Returns:
        True if max steps exceeded, False otherwise
    """
    return game.steps >= max_steps

def process_game_over(game, log_dir, args, game_number, round_count, 
                     collision_type, current_game_moves, game_parser_usage):
    """Process game over state and save game summary.
    
    Args:
        game: Game instance
        log_dir: Directory to save logs
        args: Command line arguments
        game_number: Current game number
        round_count: Number of rounds played
        collision_type: Type of collision that ended the game
        current_game_moves: List of moves made in the game
        game_parser_usage: Number of times parser was used
    """
    _save_game_summary(game, log_dir, args, game_number, round_count,
                      collision_type, current_game_moves, game_parser_usage)

def handle_error(error, game, log_dir, args, game_number, round_count,
                current_game_moves, game_parser_usage):
    """Handle game errors and save error summary.
    
    Args:
        error: Error that occurred
        game: Game instance
        log_dir: Directory to save logs
        args: Command line arguments
        game_number: Current game number
        round_count: Number of rounds played
        current_game_moves: List of moves made in the game
        game_parser_usage: Number of times parser was used
    """
    error_message = f"Error: {str(error)}"
    print(f"{Fore.RED}{error_message}")
    traceback.print_exc()
    
    _save_game_summary(game, log_dir, args, game_number, round_count,
                      error_message, current_game_moves, game_parser_usage)

def report_final_statistics(total_games, total_score, total_steps, json_error_stats):
    """Report final game statistics.
    
    Args:
        total_games: Total number of games played
        total_score: Total score across all games
        total_steps: Total steps taken across all games
        json_error_stats: Statistics about JSON parsing errors
    """
    print(f"\n{Fore.CYAN}=== Final Statistics ===")
    print(f"Total Games: {total_games}")
    print(f"Total Score: {total_score}")
    print(f"Total Steps: {total_steps}")
    print(f"Average Score: {total_score/total_games if total_games > 0 else 0:.2f}")
    print(f"Average Steps: {total_steps/total_games if total_games > 0 else 0:.2f}")
    
    if json_error_stats:
        print(f"\nJSON Parsing Statistics:")
        print(f"Total Attempts: {json_error_stats.get('total_extraction_attempts', 0)}")
        print(f"Successful: {json_error_stats.get('successful_extractions', 0)}")
        print(f"Failed: {json_error_stats.get('failed_extractions', 0)}")
        print(f"Success Rate: {json_error_stats.get('success_rate', 0):.2f}%") 