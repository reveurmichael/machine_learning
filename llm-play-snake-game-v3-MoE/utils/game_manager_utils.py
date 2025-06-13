"""
Game management utilities.
Core functionality for the Snake game manager, handling game states, error processing,
statistics reporting, and initialization functions.
"""

import os
import traceback
import pygame
import numpy as np
from colorama import Fore
import json
from datetime import datetime
from utils.json_utils import NumPyJSONEncoder

def _safe_add(target: dict, key: str, delta):
    """Add delta to target[key] only if delta is truthy (skips None / 0)."""
    if delta:
        target[key] = target.get(key, 0) + delta

def check_collision(position, snake_positions, grid_size, is_eating_apple_flag=False):
    """Check if a position collides with walls or snake body.
    
    Args:
        position: Position to check as [x, y]
        snake_positions: Array of snake body positions
        grid_size: Size of the game grid
        is_eating_apple_flag: Boolean indicating if an apple is being eaten at 'position'
        
    Returns:
        Tuple of (wall_collision, body_collision) as booleans
    """
    x, y = position
    
    # Check wall collision
    wall_collision = (x < 0 or x >= grid_size or 
                     y < 0 or y >= grid_size)
    
    # Default to no collision
    body_collision = False
    
    # Handle empty snake case (shouldn't happen normally)
    if len(snake_positions) == 0:
        return wall_collision, False
    
    # Get current snake structure for clarity
    current_tail = snake_positions[0]  # First position is tail
    current_head = snake_positions[-1] # Last position is head
    
    if is_eating_apple_flag:
        # CASE: Eating an apple - tail will NOT move
        # Check collision with all segments EXCEPT the current head
        # (since the head will move to the new position)
        
        # Check all segments except the head
        body_segments = snake_positions[:-1]  # [tail, body1, body2, ..., bodyN]
        
        # Check if new head position collides with any body segment (including tail)
        body_collision = any(np.array_equal(position, pos) for pos in body_segments)
        
    else:
        # CASE: Normal move (not eating apple) - tail WILL move
        # Only need to check for collision with body segments, excluding both
        # the current tail (which will move) and the current head (which will be replaced)
        
        if len(snake_positions) > 2:
            # If snake has body segments between tail and head
            # Check segments excluding tail and head: [body1, body2, ..., bodyN]
            body_segments = snake_positions[1:-1]
            body_collision = any(np.array_equal(position, pos) for pos in body_segments)
        else:
            # Snake has only head and tail (or just head), no body segments to collide with
            body_collision = False
        
    return wall_collision, body_collision

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
    """Process game over state.
    
    Handles the game over state including:
    - Saving game statistics
    - Updating counters for the next game
    - Creating the game summary JSON file
    
    Args:
        game: The Game instance
        game_state_info: Dictionary with game state info
        
    Returns:
        Tuple of (game_count, total_score, total_steps, game_scores, round_count, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, error_steps)
    """
    from utils.json_utils import save_session_stats
    import os
    
    args = game_state_info["args"]
    log_dir = game_state_info["log_dir"]
    current_game_moves = game_state_info["current_game_moves"]
    
    # Extract or initialize counters
    game_count = game_state_info["game_count"]
    total_score = game_state_info["total_score"]
    total_steps = game_state_info["total_steps"]
    game_scores = game_state_info["game_scores"]
    round_count = game_state_info["round_count"]
    time_stats = game_state_info.get("time_stats", {})
    token_stats = game_state_info.get("token_stats", {})
    valid_steps = game_state_info.get("valid_steps", 0)
    invalid_reversals = game_state_info.get("invalid_reversals", 0)
    empty_steps = game_state_info.get("empty_steps", 0)
    error_steps = game_state_info.get("error_steps", 0)
    
    # Print game over message with reason
    if hasattr(game, "last_collision_type"):
        collision_type = game.last_collision_type
        if collision_type == "wall":
            print(Fore.RED + "❌ Game over: Snake hit the wall!")
        elif collision_type == "self":
            print(Fore.RED + "❌ Game over: Snake hit itself!")
        elif collision_type == "empty_moves":
            print(Fore.RED + f"❌ Game over: Too many empty moves ({args.max_consecutive_empty_moves_allowed})!")
    
    # Update counters for next game
    game_count += 1
    total_score += game.score
    total_steps += game.steps
    game_scores.append(game.score)
    
    # Increment valid_steps counter
    valid_steps += game.game_state.valid_steps
    
    # Update invalid_reversals counter
    # This ensures we're keeping track of invalid_reversals across all games
    invalid_reversals += game.game_state.invalid_reversals
    
    # Update empty_steps and error_steps counters - add current game's to the running total
    empty_steps += game.game_state.empty_steps
    error_steps += game.game_state.error_steps
    
    # Print game stats
    move_str = ", ".join(current_game_moves)
    print(Fore.BLUE + f"Game {game_count} Stats:")
    print(Fore.BLUE + f"- Score: {game.score}")
    print(Fore.BLUE + f"- Steps: {game.steps}")
    print(Fore.BLUE + f"- Moves: {move_str}")
    
    # Update time statistics
    if hasattr(game.game_state, "get_time_stats"):
        game_time_stats = game.game_state.get_time_stats()
        
        # ── aggregate time statistics ────────────────────────────────
        if game_time_stats:
            _safe_add(time_stats, "llm_communication_time",
                      game_time_stats.get("llm_communication_time"))
            _safe_add(time_stats, "game_movement_time",
                      game_time_stats.get("game_movement_time"))
            _safe_add(time_stats, "waiting_time",
                      game_time_stats.get("waiting_time"))
    
    # Update token statistics
    if hasattr(game.game_state, "get_token_stats"):
        game_token_stats = game.game_state.get_token_stats()
        
        # Initialize token stats if not present
        if "primary" not in token_stats:
            token_stats["primary"] = {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
            
        if "secondary" not in token_stats:
            token_stats["secondary"] = {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        
        # Add primary LLM token stats
        if "primary" in game_token_stats:
            primary_stats = game_token_stats["primary"]
            if primary_stats.get("total_tokens") is not None:
                token_stats["primary"]["total_tokens"] = token_stats["primary"].get("total_tokens", 0) + primary_stats.get("total_tokens", 0)
            if primary_stats.get("total_prompt_tokens") is not None:
                token_stats["primary"]["total_prompt_tokens"] = token_stats["primary"].get("total_prompt_tokens", 0) + primary_stats.get("total_prompt_tokens", 0)
            if primary_stats.get("total_completion_tokens") is not None:
                token_stats["primary"]["total_completion_tokens"] = token_stats["primary"].get("total_completion_tokens", 0) + primary_stats.get("total_completion_tokens", 0)
            
        # Add secondary LLM token stats
        if "secondary" in game_token_stats:
            secondary_stats = game_token_stats["secondary"]
            if secondary_stats.get("total_tokens") is not None:
                token_stats["secondary"]["total_tokens"] = token_stats["secondary"].get("total_tokens", 0) + secondary_stats.get("total_tokens", 0)
            if secondary_stats.get("total_prompt_tokens") is not None:
                token_stats["secondary"]["total_prompt_tokens"] = token_stats["secondary"].get("total_prompt_tokens", 0) + secondary_stats.get("total_prompt_tokens", 0)
            if secondary_stats.get("total_completion_tokens") is not None:
                token_stats["secondary"]["total_completion_tokens"] = token_stats["secondary"].get("total_completion_tokens", 0) + secondary_stats.get("total_completion_tokens", 0)
    
    # Update session stats
    save_session_stats(
        log_dir,
        game_count=game_count,
        total_score=total_score,
        total_steps=total_steps,
        game_scores=game_scores,
        empty_steps=empty_steps,
        error_steps=error_steps,
        valid_steps=valid_steps,
        invalid_reversals=invalid_reversals,
        time_stats=time_stats,
        token_stats=token_stats,
        max_consecutive_empty_moves_allowed=args.max_consecutive_empty_moves_allowed,
        max_consecutive_errors_allowed=args.max_consecutive_errors_allowed
    )
    
    # Use the actual number of rounds that contain data to avoid the
    # off-by-one "phantom round" that appeared after a wall/self collision.
    if hasattr(game, "game_state") and hasattr(game.game_state, "_calculate_actual_round_count"):
        round_count = game.game_state._calculate_actual_round_count()

    # Save individual game JSON file using the canonical writer
    from utils.file_utils import get_game_json_filename, join_log_path

    game_file = join_log_path(
        log_dir,
        get_game_json_filename(game_count)
    )

    parser_provider = (
        args.parser_provider
        if args.parser_provider
           and args.parser_provider.lower() != "none"
        else None
    )

    # tag the state with the right game number
    game.game_state.game_number = game_count

    game.game_state.save_game_summary(
        game_file,
        args.provider,        # SINGLE SOURCE-OF-TRUTH
        args.model,
        parser_provider,
        args.parser_model if parser_provider else None,
        args.max_consecutive_errors_allowed
    )

    print(
        Fore.GREEN +
        f"💾 Saved data for game {game_count} "
        f"(rounds: {round_count}, moves: {len(current_game_moves)})"
    )
    
    return game_count, total_score, total_steps, game_scores, round_count, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, error_steps

def handle_error(game, error_info):
    """Handle errors during gameplay.
    
    Args:
        game: The Game instance
        error_info: Dictionary containing error information:
            - game_active: Boolean indicating if game is active
            - game_count: Current game count
            - total_score: Total score across all games
            - total_steps: Total steps across all games
            - game_scores: List of scores from all games
            - round_count: Count of rounds in the current game
            - args: Command line arguments
            - log_dir: Directory for logs
            - current_game_moves: List of moves made in the current game
            - consecutive_errors: Count of consecutive errors
            - error: The exception that occurred
            - valid_steps: Total valid steps across all games (optional)
            - invalid_reversals: Total invalid reversals across all games (optional)
            
    Returns:
        Tuple of (game_active, game_count, total_score, total_steps, game_scores, round_count, consecutive_errors, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, error_steps)
    """
    import traceback
    
    args = error_info["args"]
    consecutive_errors = error_info["consecutive_errors"]
    log_dir = error_info["log_dir"]
    
    # Extract values from input dictionary
    game_active = error_info["game_active"]
    game_count = error_info["game_count"]
    total_score = error_info["total_score"]
    total_steps = error_info["total_steps"]
    game_scores = error_info["game_scores"].copy()
    round_count = error_info["round_count"]
    current_game_moves = error_info["current_game_moves"]
    
    # Get current time and token stats
    time_stats = error_info.get("time_stats", {})
    token_stats = error_info.get("token_stats", {})
    
    # Get valid steps and invalid reversals
    valid_steps = error_info.get("valid_steps", 0)
    invalid_reversals = error_info.get("invalid_reversals", 0)
    empty_steps = error_info.get("empty_steps", 0)
    error_steps = error_info.get("error_steps", 0)
    
    # Get the error details
    error = error_info["error"]
    traceback_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    
    # Print error info
    print(Fore.RED + f"❌ Error occurred: {str(error)}")
    print(Fore.RED + traceback_str)
    
    # Record the error
    if hasattr(game.game_state, "record_error_move"):
        game.game_state.record_error_move()
    
    # Increment consecutive errors
    consecutive_errors += 1
    print(Fore.RED + f"⚠️ Consecutive errors: {consecutive_errors}/{args.max_consecutive_errors_allowed}")
    
    # Update valid steps and invalid reversals from the game state
    if hasattr(game, "game_state"):
        valid_steps += game.game_state.valid_steps
        invalid_reversals += game.game_state.invalid_reversals
        
        # Also update empty_steps and error_steps
        empty_steps += game.game_state.empty_steps
        error_steps += game.game_state.error_steps
    
    # Check if we should end the game
    if consecutive_errors >= args.max_consecutive_errors_allowed:
        print(Fore.RED + f"❌ Maximum consecutive errors reached ({args.max_consecutive_errors_allowed}). Game over.")
        game_active = False
        
        # Set the game end reason
        game.game_state.record_game_end("MAX_CONSECUTIVE_ERRORS_REACHED")
        
        # Add score to game scores
        game_scores.append(game.score)
        
        # Update counters
        game_count += 1
        total_score += game.score
        total_steps += game.steps
        
        # Update time statistics
        game_time_stats = game.game_state.get_time_stats()
        if time_stats and game_time_stats:
            time_stats["llm_communication_time"] = time_stats.get("llm_communication_time", 0) + game_time_stats.get("llm_communication_time", 0)
            time_stats["game_movement_time"] = time_stats.get("game_movement_time", 0) + game_time_stats.get("game_movement_time", 0)
            time_stats["waiting_time"] = time_stats.get("waiting_time", 0) + game_time_stats.get("waiting_time", 0)
        
        # Update token statistics
        game_token_stats = game.game_state.get_token_stats()
        if token_stats and game_token_stats:
            # Initialize token stats if not present
            if "primary" not in token_stats:
                token_stats["primary"] = {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0
                }
                
            if "secondary" not in token_stats:
                token_stats["secondary"] = {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0
                }
            
            # Update primary LLM token stats
            if "primary" in token_stats and "primary" in game_token_stats:
                primary_stats = game_token_stats["primary"]
                token_stats["primary"]["total_tokens"] = token_stats["primary"].get("total_tokens", 0) + primary_stats.get("total_tokens", 0)
                token_stats["primary"]["total_prompt_tokens"] = token_stats["primary"].get("total_prompt_tokens", 0) + primary_stats.get("total_prompt_tokens", 0)
                token_stats["primary"]["total_completion_tokens"] = token_stats["primary"].get("total_completion_tokens", 0) + primary_stats.get("total_completion_tokens", 0)
            
            # Update secondary LLM token stats
            if "secondary" in token_stats and "secondary" in game_token_stats:
                secondary_stats = game_token_stats["secondary"]
                token_stats["secondary"]["total_tokens"] = token_stats["secondary"].get("total_tokens", 0) + secondary_stats.get("total_tokens", 0)
                token_stats["secondary"]["total_prompt_tokens"] = token_stats["secondary"].get("total_prompt_tokens", 0) + secondary_stats.get("total_prompt_tokens", 0)
                token_stats["secondary"]["total_completion_tokens"] = token_stats["secondary"].get("total_completion_tokens", 0) + secondary_stats.get("total_completion_tokens", 0)
        
        # Save game summary
        import os
        from utils.file_utils import get_game_json_filename, join_log_path
        
        json_filename = get_game_json_filename(game_count)
        json_path = join_log_path(log_dir, json_filename)
        parser_provider = args.parser_provider if args.parser_provider and args.parser_provider.lower() != "none" else None
        
        # Set the correct game number in the game state
        game.game_state.game_number = game_count
        
        game.game_state.save_game_summary(
            json_path,
            args.provider, 
            args.model or f"default_{args.provider}",
            parser_provider,
            args.parser_model if parser_provider else None,
            args.max_consecutive_errors_allowed
        )
    
    return game_active, game_count, total_score, total_steps, game_scores, round_count, previous_parser_usage, consecutive_errors, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, error_steps

def report_final_statistics(stats_info):
    """Report final statistics for the experiment.
    
    Args:
        stats_info: Dictionary containing statistics information:
        - log_dir: Directory containing the summary.json file
        - game_count: Number of games played
        - total_score: Total score across all games
        - total_steps: Total number of steps taken
        - game_scores: List of scores for each game
        - empty_steps: Total empty steps across all games
        - error_steps: Total error steps across all games
        - valid_steps: Total valid steps across all games (optional)
        - invalid_reversals: Total invalid reversals across all games (optional)
    """
    from utils.json_utils import save_session_stats
    import os
    import json
    
    # Extract statistics
    log_dir = stats_info["log_dir"]
    game_count = stats_info["game_count"]
    total_score = stats_info["total_score"]
    total_steps = stats_info["total_steps"]
    game_scores = stats_info["game_scores"]
    empty_steps = stats_info["empty_steps"]
    error_steps = stats_info["error_steps"]
    valid_steps = stats_info.get("valid_steps", 0)
    invalid_reversals = stats_info.get("invalid_reversals", 0)
    max_consecutive_empty_moves_allowed = stats_info["max_consecutive_empty_moves_allowed"]
    max_consecutive_errors_allowed = stats_info.get("max_consecutive_errors_allowed", 5)
    
    # Get time and token statistics from the game instance if available
    time_stats = {}
    token_stats = {}
    game = stats_info.get("game")
    
    # If we have access to the game instance, update our statistics from it
    if game and hasattr(game, "game_state"):
        game_state = game.game_state
        
        # Get time stats
        time_stats = game_state.get_time_stats()
        
        # Get token stats
        token_stats = game_state.get_token_stats()
        
        # Get step stats - these are the stats that matter for each individual game
        step_stats = game_state.get_step_stats()
    
    # Get token stats specifically from the game manager if available
    if "token_stats" in stats_info:
        token_stats = stats_info["token_stats"]
        
    # Get time stats specifically from the game manager if available
    if "time_stats" in stats_info:
        time_stats = stats_info["time_stats"]
    
    # Save session statistics to summary file
    save_session_stats(
        log_dir, 
        game_count=game_count, 
        total_score=total_score, 
        total_steps=total_steps, 
        game_scores=game_scores, 
        empty_steps=empty_steps, 
        error_steps=error_steps,
        valid_steps=valid_steps,
        invalid_reversals=invalid_reversals,
        time_stats=time_stats,
        token_stats=token_stats
    )
    
    # Print final statistics
    print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
    print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
    print(Fore.GREEN + f"🏁 Final Score: {total_score}")
    print(Fore.GREEN + f"👣 Total Steps: {total_steps}")
    
    # Calculate and print average score
    avg_score = total_score / game_count if game_count > 0 else 0
    print(Fore.GREEN + f"📊 Average Score: {avg_score:.2f}")
    
    # Calculate and print apples per step
    apples_per_step = total_score / total_steps if total_steps > 0 else 0
    print(Fore.GREEN + f"📈 Apples per Step: {apples_per_step:.4f}")
    
    # Print step statistics
    print(Fore.GREEN + f"📈 Empty Steps: {empty_steps}")
    print(Fore.GREEN + f"📈 Error Steps: {error_steps}")
    print(Fore.GREEN + f"📈 Valid Steps: {valid_steps}")
    print(Fore.GREEN + f"📈 Invalid Reversals: {invalid_reversals}")
    
    # Print move limits
    print(Fore.GREEN + f"📈 Max Empty Moves: {max_consecutive_empty_moves_allowed}")
    print(Fore.GREEN + f"📈 Max Consecutive Errors: {max_consecutive_errors_allowed}")
    
    # End message based on max games reached
    if game_count >= stats_info.get("max_games", float('inf')):
        print(Fore.GREEN + f"🏁 Reached maximum games ({game_count}). Session complete.")

def initialize_game_manager(game_manager):
    """Initialize the game manager with necessary setup.
    
    Sets up the LLM clients (primary and optional secondary),
    creates session directories, and initializes game state tracking.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.json_utils import save_experiment_info_json
    from utils.initialization_utils import setup_log_directories, setup_llm_clients, initialize_game_state
    import os
    import time

    # Set up the LLM clients (primary and optional secondary)
    setup_llm_clients(game_manager)

    # Handle sleep before launching if specified
    if game_manager.args.sleep_before_launching > 0:
        minutes = game_manager.args.sleep_before_launching
        print(Fore.YELLOW + f"💤 Sleeping for {minutes} minute{'s' if minutes > 1 else ''} before launching...")
        time.sleep(minutes * 60)
        print(Fore.GREEN + "⏰ Waking up and starting the program...")

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


def calculate_move_differences(head_pos, apple_pos):
    """Calculate the expected move differences based on head and apple positions.

    Args:
        head_pos: Position of the snake's head as [x, y]
        apple_pos: Position of the apple as [x, y]

    Returns:
        String describing the expected move differences with actual numbers
    """
    head_x, head_y = head_pos
    apple_x, apple_y = apple_pos

    # Calculate horizontal differences
    x_diff_text = ""
    if head_x <= apple_x:
        x_diff = apple_x - head_x
        x_diff_text = f"#RIGHT - #LEFT = {x_diff} (= {apple_x} - {head_x})"
    else:
        x_diff = head_x - apple_x
        x_diff_text = f"#LEFT - #RIGHT = {x_diff} (= {head_x} - {apple_x})"

    # Calculate vertical differences
    y_diff_text = ""
    if head_y <= apple_y:
        y_diff = apple_y - head_y
        y_diff_text = f"#UP - #DOWN = {y_diff} (= {apple_y} - {head_y})"
    else:
        y_diff = head_y - apple_y
        y_diff_text = f"#DOWN - #UP = {y_diff} (= {head_y} - {apple_y})"

    return f"{x_diff_text}, and {y_diff_text}"


def format_body_cells_str(body_positions):
    """Format the snake body cells as a string representation.

    Args:
        body_positions: List of [x, y] coordinates of the snake segments

    Returns:
        String representation of body cells in format: "[(x1,y1), (x2,y2), ...]"
    """
    body_cells = []

    # Format each position as a tuple string
    for x, y in body_positions:
        body_cells.append(f"({x},{y})")

    return "[" + ", ".join(body_cells) + "]"
