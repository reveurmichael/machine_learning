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

def process_game_over(game, game_active, game_count, total_score, total_steps, game_scores, 
                    round_count, args, log_dir, current_game_moves=None, next_move=None):
    """Process game over state and prepare for the next game.
    
    Args:
        game: The game instance
        game_active: Boolean indicating if the game is active
        game_count: Count of games played
        total_score: Total score across all games
        total_steps: Total steps across all games
        game_scores: List of scores for all games
        round_count: Count of rounds in the current game
        args: Command line arguments
        log_dir: Directory for logging
        current_game_moves: List of moves made in the current game
        next_move: The last move made (or None)
        
    Returns:
        Tuple of (game_count, total_score, total_steps, game_scores, round_count)
    """
    # Update game count and statistics
    game_count += 1
    total_score += game.score
    total_steps += game.steps
    game_scores.append(game.score)
    
    # Set a reason if not already set by the game engine
    if not game.game_state.game_end_reason:
        if game.last_collision_type == 'empty_moves':
            game.game_state.record_game_end("EMPTY_MOVES")
        elif game.last_collision_type == 'max_steps':
            game.game_state.record_game_end("MAX_STEPS")
        else:
            game.game_state.record_game_end("UNKNOWN")
    
    # Save game summary
    json_path = os.path.join(log_dir, f"game{game_count}.json")
    game.game_state.save_game_summary(
        json_path,
        args.provider, 
        args.model or f"default_{args.provider}",
        args.parser_provider or args.provider,
        args.parser_model
    )
    
    return game_count, total_score, total_steps, game_scores, round_count

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
        
        # Set game end reason
        game.last_collision_type = 'error'
        game.game_state.record_game_end("ERROR")
        
        # Store moves in game state
        if current_game_moves:
            game.game_state.moves = current_game_moves
        
        # Save game summary
        json_path = os.path.join(log_dir, f"game{game_count}.json")
        game.game_state.save_game_summary(
            json_path,
            args.provider, 
            args.model or f"default_{args.provider}",
            args.parser_provider or args.provider,
            args.parser_model
        )
        print(Fore.GREEN + f"📝 Game summary saved to {json_path}")
    
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
    
    # Update experiment summary with final statistics
    json_error_stats = get_json_error_stats()
    update_experiment_info_json(
        log_dir, 
        game_count=game_count, 
        total_score=total_score, 
        total_steps=total_steps, 
        parser_usage_count=parser_usage_count, 
        game_scores=game_scores, 
        empty_steps=empty_steps, 
        error_steps=error_steps,
        json_error_stats=json_error_stats,
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