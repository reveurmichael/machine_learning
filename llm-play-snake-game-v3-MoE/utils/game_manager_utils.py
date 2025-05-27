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
        args.parser_model,
        args.max_consecutive_errors_allowed
    )
    
    return game_count, total_score, total_steps, game_scores, round_count

def handle_error(game, game_active, game_count, total_score, total_steps, 
                game_scores, round_count, parser_usage_count, previous_parser_usage, 
                log_dir, args, current_game_moves, error, consecutive_errors=0):
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
        consecutive_errors: Current count of consecutive errors
        
    Returns:
        Tuple of (game_active, game_count, total_score, total_steps, game_scores, 
                 round_count, previous_parser_usage, consecutive_errors)
    """
    print(Fore.RED + f"Error in game loop: {error}")
    traceback.print_exc()
    
    # Increment consecutive errors count
    consecutive_errors += 1
    
    # End the current game if consecutive errors exceed threshold or if this is a critical error
    if game_active and (consecutive_errors > args.max_consecutive_errors_allowed):
        game_active = False
        game_count += 1
        print(Fore.RED + f"❌ Game aborted due to {consecutive_errors} consecutive errors! Maximum allowed: {args.max_consecutive_errors_allowed}")
        print(Fore.RED + f"Moving to game {game_count + 1}")
        
        # Update totals with current game state
        total_score += game.score
        total_steps += game.steps
        game_scores.append(game.score)
        
        # Set game end reason
        game.last_collision_type = 'error'
        game.game_state.record_game_end("ERROR_THRESHOLD")
        
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
            args.parser_model,
            args.max_consecutive_errors_allowed
        )
        print(Fore.GREEN + f"📝 Game summary saved to {json_path}")
        
        # Reset consecutive errors for next game
        consecutive_errors = 0
    
    return game_active, game_count, total_score, total_steps, game_scores, round_count, previous_parser_usage, consecutive_errors

def report_final_statistics(log_dir, game_count, total_score, total_steps,
                           parser_usage_count, game_scores, empty_steps, 
                           error_steps, max_empty_moves, max_consecutive_errors_allowed=5):
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
        max_consecutive_errors_allowed: Maximum allowed consecutive errors
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
        max_empty_moves=max_empty_moves,
        max_consecutive_errors_allowed=max_consecutive_errors_allowed
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
    print(Fore.GREEN + f"📈 Max Empty Moves: {max_empty_moves}")
    print(Fore.GREEN + f"📈 Max Consecutive Errors: {max_consecutive_errors_allowed}")
    
    if json_error_stats['total_extraction_attempts'] > 0:
        print(Fore.GREEN + f"📈 JSON Extraction Attempts: {json_error_stats['total_extraction_attempts']}")
        success_rate = (json_error_stats['successful_extractions'] / json_error_stats['total_extraction_attempts']) * 100
        print(Fore.GREEN + f"📈 JSON Extraction Success Rate: {success_rate:.2f}%")

def initialize_game_manager(game_manager):
    """Initialize the game manager with necessary setup.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.json_utils import reset_json_error_stats, save_experiment_info_json
    from utils.llm_utils import check_llm_health
    import os
    import sys
    import time
    import pygame
    from datetime import datetime
    
    # Reset JSON error statistics
    reset_json_error_stats()
    
    # Initialize primary LLM client
    game_manager.llm_client = game_manager.create_llm_client(
        game_manager.args.provider, 
        game_manager.args.model
    )
    
    print(Fore.GREEN + f"Using primary LLM provider: {game_manager.args.provider}")
    if game_manager.args.model:
        print(Fore.GREEN + f"Using primary LLM model: {game_manager.args.model}")
    
    # Perform health check for primary LLM
    primary_healthy, primary_response = check_llm_health(game_manager.llm_client)
    if not primary_healthy:
        print(Fore.RED + f"❌ Primary LLM health check failed. The program cannot continue.")
        sys.exit(1)
    else:
        print(Fore.GREEN + f"✅ Primary LLM health check passed!")
        
    # Configure secondary LLM (parser) if specified
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
        print(Fore.GREEN + f"Using parser LLM provider: {game_manager.args.parser_provider}")
        parser_model = game_manager.args.parser_model
        print(Fore.GREEN + f"Using parser LLM model: {parser_model}")
        
        # Set up the secondary LLM in the client
        game_manager.llm_client.set_secondary_llm(game_manager.args.parser_provider, parser_model)
        
        # Perform health check for parser LLM
        parser_healthy, _ = check_llm_health(
            game_manager.create_llm_client(game_manager.args.parser_provider, parser_model)
        )
        if not parser_healthy:
            print(Fore.RED + f"❌ Parser LLM health check failed. Continuing without parser.")
            game_manager.args.parser_provider = "none"
            game_manager.args.parser_model = None
    else:
        print(Fore.YELLOW + "⚠️ No parser LLM specified. Using primary LLM output directly.")
        game_manager.args.parser_provider = "none"
        game_manager.args.parser_model = None
    
    # Handle sleep before launching if specified
    if game_manager.args.sleep_before_launching > 0:
        minutes = game_manager.args.sleep_before_launching
        print(Fore.YELLOW + f"💤 Sleeping for {minutes} minute{'s' if minutes > 1 else ''} before launching...")
        time.sleep(minutes * 60)
        print(Fore.GREEN + "⏰ Waking up and starting the program...")
    
    # Initialize pygame if using GUI
    if game_manager.use_gui:
        pygame.init()
        pygame.font.init()
    
    # Set up the game
    game_manager.setup_game()
    
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    
    # Set up logging directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    primary_model = game_manager.args.model if game_manager.args.model else f'default_{game_manager.args.provider}'
    primary_model = primary_model.replace(':', '-')  # Replace colon with hyphen
    game_manager.log_dir = f"{primary_model}_{timestamp}"
    game_manager.prompts_dir = os.path.join(game_manager.log_dir, "prompts")
    game_manager.responses_dir = os.path.join(game_manager.log_dir, "responses")
    
    # Save experiment information
    model_info_path = save_experiment_info_json(game_manager.args, game_manager.log_dir)
    print(Fore.GREEN + f"📝 Experiment information saved to {model_info_path}")

def process_events(game_manager):
    """Process pygame events.
    
    Args:
        game_manager: The GameManager instance
    """
    import pygame
    
    if not game_manager.use_gui:
        return
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_manager.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_manager.running = False
            elif event.key == pygame.K_r:
                # Reset game
                game_manager.game.reset()
                game_manager.game_active = True
                game_manager.need_new_plan = True
                game_manager.consecutive_empty_steps = 0  # Reset on game reset
                game_manager.current_game_moves = []  # Reset moves for new game
                print(Fore.GREEN + "🔄 Game reset")

def extract_state_for_parser(game_manager):
    """Extract state information for the parser.
    
    Args:
        game_manager: The GameManager instance
        
    Returns:
        Tuple of (head_pos, apple_pos, body_cells) as strings
    """
    # Get the game state
    head_x, head_y = game_manager.game.head
    apple_x, apple_y = game_manager.game.apple
    body_cells = game_manager.game.body
    
    # Format for parser
    head_pos = f"({head_x}, {head_y})"
    apple_pos = f"({apple_x}, {apple_y})"
    body_cells_str = str(body_cells) if body_cells else "[]"
    
    return head_pos, apple_pos, body_cells_str 