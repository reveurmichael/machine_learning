"""
Game management utilities.
Core functionality for the Snake game manager, handling game states, error processing,
statistics reporting, and initialization functions.
"""

import os
import traceback
import pygame
from colorama import Fore

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
    # Calculate new statistics after game completion
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
    
    # Reset round_count to 1 for the next game
    round_count = 1
    
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
        
        # Include current game stats in totals
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
        
        # Reset round_count to 1 for the next game
        round_count = 1
    
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
    from utils.json_utils import get_json_error_stats, save_session_stats
    
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
    
    # Save session statistics to summary file
    json_error_stats = get_json_error_stats()
    save_session_stats(
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
    
    Sets up the LLM clients (primary and optional secondary),
    creates session directories, and initializes game state tracking.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.json_utils import reset_json_error_stats, save_experiment_info_json
    from utils.initialization_utils import setup_log_directories, setup_llm_clients, initialize_game_state
    import os
    import time
    
    # Initialize statistics tracking
    reset_json_error_stats()
    
    # Handle sleep before launching if specified
    if game_manager.args.sleep_before_launching > 0:
        minutes = game_manager.args.sleep_before_launching
        print(Fore.YELLOW + f"💤 Sleeping for {minutes} minute{'s' if minutes > 1 else ''} before launching...")
        time.sleep(minutes * 60)
        print(Fore.GREEN + "⏰ Waking up and starting the program...")
    
    # Set up the LLM clients (primary and optional secondary)
    setup_llm_clients(game_manager)
    
    # Set up session directories
    if game_manager.args.log_dir:
        # Use provided log directory
        game_manager.log_dir = game_manager.args.log_dir
        game_manager.prompts_dir = os.path.join(game_manager.log_dir, "prompts")
        game_manager.responses_dir = os.path.join(game_manager.log_dir, "responses")
        
        # Create directories if they don't exist
        os.makedirs(game_manager.log_dir, exist_ok=True)
        os.makedirs(game_manager.prompts_dir, exist_ok=True)
        os.makedirs(game_manager.responses_dir, exist_ok=True)
    else:
        # Create new session directory
        setup_log_directories(game_manager)
    
    # Save experiment information
    model_info_path = save_experiment_info_json(game_manager.args, game_manager.log_dir)
    print(Fore.GREEN + f"📝 Experiment information saved to {model_info_path}")
    
    # Initialize game state
    initialize_game_state(game_manager)

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