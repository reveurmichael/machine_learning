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

def check_max_steps(game, max_steps):
    """Check if the game has reached the maximum number of steps.
    
    Args:
        game: The snake game instance
        max_steps: Maximum number of steps allowed
        
    Returns:
        Boolean indicating if max steps has been reached
    """
    if game.steps >= max_steps:
        print(Fore.RED + f"❌ Game over! Maximum steps ({max_steps}) reached.")
        game.last_collision_type = 'max_steps'
        return True
    return False

def process_game_over(game, game_count, total_score, total_steps, 
                     game_scores, round_count, parser_usage_count, 
                     previous_parser_usage, log_dir, args, current_game_moves):
    """Process game over state and prepare for the next game.
    
    Args:
        game: The snake game instance
        game_count: Current game count
        total_score: Total score across all games
        total_steps: Total steps across all games
        game_scores: List of scores from all games
        round_count: Current round count
        parser_usage_count: Count of parser usage
        previous_parser_usage: Previous parser usage count
        log_dir: Directory for logs
        args: Command line arguments
        current_game_moves: List of moves made in the current game
        
    Returns:
        Tuple of (game_count, total_score, total_steps, game_scores, round_count, previous_parser_usage)
    """
    from utils.log_utils import generate_game_summary_json
    from utils.json_utils import get_json_error_stats
    
    game_count += 1
    print(Fore.RED + f"❌ Game over! Score: {game.score}, Steps: {game.steps}")
    
    # Update totals
    total_score += game.score
    total_steps += game.steps
    game_scores.append(game.score)
    
    # Calculate game-specific statistics
    game_parser_usage = parser_usage_count if game_count == 1 else parser_usage_count - previous_parser_usage
    previous_parser_usage = parser_usage_count
    
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
    json_summary = generate_game_summary_json(
        game_count,
        now,
        game.score,
        game.steps,
        None,  # next_move
        game_parser_usage,
        len(game.snake_positions),
        game.last_collision_type,
        round_count,
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
        moves=current_game_moves  # Add all moves made during the game
    )
    
    # Save JSON summary
    json_path = os.path.join(log_dir, f"game{game_count}_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(Fore.GREEN + f"📝 JSON summary saved to {json_path}")
    
    return game_count, total_score, total_steps, game_scores, round_count, previous_parser_usage

def handle_error(game, game_active, game_count, total_score, total_steps, 
                game_scores, round_count, parser_usage_count, previous_parser_usage, 
                log_dir, args, current_game_moves, error):
    """Handle errors that occur during the game loop.
    
    Args:
        game: The snake game instance
        game_active: Boolean indicating if game is active
        game_count: Current game count
        total_score: Total score across all games
        total_steps: Total steps across all games
        game_scores: List of scores from all games
        round_count: Current round count
        parser_usage_count: Count of parser usage
        previous_parser_usage: Previous parser usage count
        log_dir: Directory for logs
        args: Command line arguments
        current_game_moves: List of moves made in the current game
        error: The exception that occurred
        
    Returns:
        Tuple of (game_active, game_count, total_score, total_steps, game_scores, 
                 round_count, previous_parser_usage)
    """
    from utils.log_utils import generate_game_summary_json
    from utils.json_utils import get_json_error_stats
    
    print(Fore.RED + f"Error in game loop: {error}")
    traceback.print_exc()
    
    # End the current game and continue to the next one
    if game_active:
        game_active = False
        game_count += 1
        print(Fore.RED + f"❌ Game aborted due to error! Moving to game {game_count + 1}")
        
        # Update totals with current game state
        total_score += game.score
        total_steps += game.steps
        game_scores.append(game.score)
        
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
        
        # Generate JSON summary with error information
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        game.last_collision_type = 'error'
        json_summary = generate_game_summary_json(
            game_count,
            now,
            game.score,
            game.steps,
            "ERROR",
            parser_usage_count - previous_parser_usage,
            len(game.snake_positions),
            game.last_collision_type,
            round_count,
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
            moves=current_game_moves  # Add all moves made during the game
        )
        
        # Save JSON summary
        json_path = os.path.join(log_dir, f"game{game_count}_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2)
        print(Fore.GREEN + f"📝 JSON summary saved to {json_path}")
    
    return game_active, game_count, total_score, total_steps, game_scores, round_count, previous_parser_usage

def report_final_statistics(log_dir, game_count, total_score, total_steps,
                           parser_usage_count, game_scores, empty_steps, 
                           error_steps, max_empty_moves):
    """Report final statistics at the end of the game session.
    
    Args:
        log_dir: Directory for logs
        game_count: Total games played
        total_score: Total score across all games
        total_steps: Total steps across all games
        parser_usage_count: Count of parser usage
        game_scores: List of scores from all games
        empty_steps: Number of empty steps
        error_steps: Number of error steps
        max_empty_moves: Maximum allowed empty moves
    """
    from utils.json_utils import get_json_error_stats, update_experiment_info_json
    
    # Update experiment info with final statistics
    json_error_stats = get_json_error_stats()
    update_experiment_info_json(
        log_dir, 
        game_count, 
        total_score, 
        total_steps, 
        parser_usage_count, 
        game_scores, 
        empty_steps, 
        error_steps,
        json_error_stats,
        max_empty_moves=max_empty_moves
    )
    
    print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
    print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
    print(Fore.GREEN + f"🏁 Final Score: {total_score}")
    print(Fore.GREEN + f"👣 Total Steps: {total_steps}")
    print(Fore.GREEN + f"🔄 Secondary LLM was used {parser_usage_count} times")
    
    if game_count > 0:
        print(Fore.GREEN + f"📊 Average Score: {total_score/game_count:.2f}")
    
    if total_steps > 0:
        print(Fore.GREEN + f"📈 Apples per Step: {total_score/total_steps:.4f}")
        
    print(Fore.GREEN + f"📈 Empty Steps: {empty_steps}")
    print(Fore.GREEN + f"📈 Error Steps: {error_steps}")
    
    if json_error_stats['total_extraction_attempts'] > 0:
        print(Fore.GREEN + f"📈 JSON Extraction Attempts: {json_error_stats['total_extraction_attempts']}")
        success_rate = (json_error_stats['successful_extractions'] / json_error_stats['total_extraction_attempts']) * 100
        print(Fore.GREEN + f"📈 JSON Extraction Success Rate: {success_rate:.2f}%") 