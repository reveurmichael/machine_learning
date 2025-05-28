"""
Utility module for game management.
Handles common game management tasks such as processing game over states,
error handling, and statistics reporting.
"""

import os
import traceback
import pygame
from colorama import Fore
from datetime import datetime

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

def process_game_over(game, game_state_info):
    """Process game over state and prepare for the next game.
    
    Args:
        game: The game instance
        game_state_info: Dictionary containing game state information:
            - game_active: Boolean indicating if the game is active
            - game_count: Count of games played
            - total_score: Total score across all games
            - total_steps: Total steps across all games
            - game_scores: List of scores for all games
            - round_count: Count of rounds in the current game
            - args: Command line arguments
            - log_dir: Directory for logging
            - current_game_moves: List of moves made in the current game (optional)
            - next_move: The last move made (optional)
        
    Returns:
        Tuple of (game_count, total_score, total_steps, game_scores, round_count)
    """
    # Update game count and statistics
    game_count = game_state_info["game_count"] + 1
    total_score = game_state_info["total_score"] + game.score
    total_steps = game_state_info["total_steps"] + game.steps
    game_scores = game_state_info["game_scores"].copy()
    game_scores.append(game.score)
    round_count = game_state_info["round_count"]
    args = game_state_info["args"]
    log_dir = game_state_info["log_dir"]
    
    # Set a reason if not already set by the game engine
    if not game.game_state.game_end_reason:
        if game.last_collision_type == 'empty_moves':
            game.game_state.record_game_end("EMPTY_MOVES")
        elif game.last_collision_type == 'max_steps':
            game.game_state.record_game_end("MAX_STEPS")
        else:
            game.game_state.record_game_end("UNKNOWN")
    
    # Save game summary
    json_path = os.path.join(log_dir, f"game_{game_count}.json")
    game.game_state.save_game_summary(
        json_path,
        args.provider, 
        args.model or f"default_{args.provider}",
        args.parser_provider or args.provider,
        args.parser_model,
        args.max_consecutive_errors_allowed
    )
    
    return game_count, total_score, total_steps, game_scores, round_count

def handle_error(game, error_info):
    """Handle errors that occur during the game loop.
    
    Args:
        game: The snake game instance
        error_info: Dictionary containing error handling information:
            - game_active: Boolean indicating if game is active
            - game_count: Current game count
            - total_score: Total score across all games
            - total_steps: Total steps across all games
            - game_scores: List of scores from all games
            - round_count: Current round count
            - parser_usage_count: Count of parser usage
            - previous_parser_usage: Previous parser usage count
            - log_dir: Directory for logs
            - args: Command line arguments
            - current_game_moves: List of moves made in the current game
            - error: The exception that occurred
            - consecutive_errors: Current count of consecutive errors (default: 0)
        
    Returns:
        Tuple of (game_active, game_count, total_score, total_steps, game_scores, 
                 round_count, previous_parser_usage, consecutive_errors)
    """
    print(Fore.RED + f"Error in game loop: {error_info['error']}")
    traceback.print_exc()
    
    # Initialize return values from input dictionary
    game_active = error_info["game_active"]
    game_count = error_info["game_count"]
    total_score = error_info["total_score"]
    total_steps = error_info["total_steps"]
    game_scores = error_info["game_scores"].copy()
    round_count = error_info["round_count"]
    previous_parser_usage = error_info["previous_parser_usage"]
    consecutive_errors = error_info.get("consecutive_errors", 0) + 1
    args = error_info["args"]
    log_dir = error_info["log_dir"]
    current_game_moves = error_info.get("current_game_moves", [])
    
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
        json_path = os.path.join(log_dir, f"game_{game_count}.json")
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

def report_final_statistics(stats_info):
    """Report final statistics at the end of the game session.
    
    Args:
        stats_info: Dictionary containing statistics information:
            - log_dir: Directory for logs
            - game_count: Total games played
            - total_score: Total score across all games
            - total_steps: Total steps across all games
            - parser_usage_count: Count of parser usage
            - game_scores: List of scores from all games
            - empty_steps: Number of empty steps
            - error_steps: Number of error steps
            - max_empty_moves: Maximum allowed empty moves
            - max_consecutive_errors_allowed: Maximum allowed consecutive errors (default: 5)
    """
    from utils.json_utils import get_json_error_stats, update_experiment_info_json
    
    # Extract values from input dictionary
    log_dir = stats_info["log_dir"]
    game_count = stats_info["game_count"]
    total_score = stats_info["total_score"]
    total_steps = stats_info["total_steps"]
    parser_usage_count = stats_info["parser_usage_count"]
    game_scores = stats_info["game_scores"]
    empty_steps = stats_info["empty_steps"]
    error_steps = stats_info["error_steps"]
    max_empty_moves = stats_info["max_empty_moves"]
    max_consecutive_errors_allowed = stats_info.get("max_consecutive_errors_allowed", 5)
    
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
    import os
    import time
    
    # Reset JSON error statistics
    reset_json_error_stats()
    
    # Import these functions inside the function to avoid cyclic imports
    from utils.llm_utils import check_llm_health
    from utils.setup_utils import setup_llm_clients
    
    # Set up the common setup function
    setup_llm_clients(game_manager, check_llm_health)
    
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