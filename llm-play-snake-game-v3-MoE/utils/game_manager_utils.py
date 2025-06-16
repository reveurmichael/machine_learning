"""
Game management utilities.
Core functionality for the Snake game manager, handling game states, error processing,
statistics reporting, and initialization functions.
"""

import pygame
import numpy as np
from colorama import Fore

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
        Tuple of (game_count, total_score, total_steps, game_scores, round_count, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, something_is_wrong_steps)
    """
    from utils.game_stats_utils import save_session_stats
    
    args = game_state_info["args"]
    log_dir = game_state_info["log_dir"]
    
    # Extract or initialize counters
    game_count = game_state_info["game_count"]
    total_score = game_state_info["total_score"]
    total_steps = game_state_info["total_steps"]
    game_scores = game_state_info["game_scores"]
    round_count = game_state_info["round_count"]
    # List of per-game round counts collected so far (mutated in-place)
    round_counts = game_state_info.get("round_counts", [])
    time_stats = game_state_info.get("time_stats", {})
    token_stats = game_state_info.get("token_stats", {})
    valid_steps = game_state_info.get("valid_steps", 0)
    invalid_reversals = game_state_info.get("invalid_reversals", 0)
    empty_steps = game_state_info.get("empty_steps", 0)
    something_is_wrong_steps = game_state_info.get("something_is_wrong_steps", 0)
    
    # Print game over message with reason
    if hasattr(game, "last_collision_type"):
        collision_type = game.last_collision_type
        if collision_type == "wall":
            print(Fore.RED + "❌ Game over: Snake hit the wall!")
        elif collision_type == "self":
            print(Fore.RED + "❌ Game over: Snake hit itself!")
        elif collision_type == "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED":
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
    
    # Update empty_steps and something_is_wrong_steps counters - add current game's to the running total
    empty_steps += game.game_state.empty_steps
    something_is_wrong_steps += game.game_state.something_is_wrong_steps
    
    # Print game stats using the EXECUTED moves that are stored on the GameData
    # instance to avoid counting duplicate/un-executed planned moves.
    executed_moves = game.game_state.moves  # authoritative list
    move_str = ", ".join(executed_moves)
    print(Fore.BLUE + f"Game {game_count} Stats:")
    print(Fore.BLUE + f"- Score: {game.score}")
    print(Fore.BLUE + f"- Steps: {game.steps}")
    print(Fore.BLUE + f"- Valid Steps: {game.game_state.valid_steps}")
    print(Fore.BLUE + f"- Invalid Reversals: {game.game_state.invalid_reversals}")
    print(Fore.BLUE + f"- Moves: {move_str}")
    
    # Update time statistics
    if hasattr(game.game_state, "get_time_stats"):
        game_time_stats = game.game_state.get_time_stats()
        
        # ── aggregate time statistics ────────────────────────────────
        if game_time_stats:
            _safe_add(time_stats, "llm_communication_time",
                      game_time_stats.get("llm_communication_time"))
            # game_movement_time / waiting_time were removed from the schema → no aggregation needed
            
            # Also keep track of primary vs. secondary LLM communication time
            primary_time = sum(getattr(game.game_state, "primary_response_times", []))
            secondary_time = sum(getattr(game.game_state, "secondary_response_times", []))
            
            _safe_add(time_stats, "primary_llm_communication_time", primary_time)
            _safe_add(time_stats, "secondary_llm_communication_time", secondary_time)
    
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
        
        # Add primary LLM token stats (flat-key schema)
        primary_total = game_token_stats.get("primary_total_tokens")
        primary_prompt = game_token_stats.get("primary_total_prompt_tokens")
        primary_completion = game_token_stats.get("primary_total_completion_tokens")

        if primary_total is not None:
            token_stats["primary"]["total_tokens"] = token_stats["primary"].get("total_tokens", 0) + primary_total
        if primary_prompt is not None:
            token_stats["primary"]["total_prompt_tokens"] = token_stats["primary"].get("total_prompt_tokens", 0) + primary_prompt
        if primary_completion is not None:
            token_stats["primary"]["total_completion_tokens"] = token_stats["primary"].get("total_completion_tokens", 0) + primary_completion

        # Add secondary LLM token stats (flat-key schema)
        secondary_total = game_token_stats.get("secondary_total_tokens")
        secondary_prompt = game_token_stats.get("secondary_total_prompt_tokens")
        secondary_completion = game_token_stats.get("secondary_total_completion_tokens")

        if secondary_total is not None:
            token_stats["secondary"]["total_tokens"] = token_stats["secondary"].get("total_tokens", 0) + secondary_total
        if secondary_prompt is not None:
            token_stats["secondary"]["total_prompt_tokens"] = token_stats["secondary"].get("total_prompt_tokens", 0) + secondary_prompt
        if secondary_completion is not None:
            token_stats["secondary"]["total_completion_tokens"] = token_stats["secondary"].get("total_completion_tokens", 0) + secondary_completion
    
    # Use the actual number of rounds that contain data to avoid the
    # off-by-one "phantom round" that appeared after a wall/self collision.
    if hasattr(game, "game_state") and hasattr(game.game_state, "_calculate_actual_round_count"):
        round_count = game.game_state._calculate_actual_round_count()

    # ── aggregate round counts ──────────────────────────────────────────
    round_counts.append(round_count)
    total_rounds = sum(round_counts)

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
        primary_provider=args.provider,
        primary_model=args.model,
        parser_provider=parser_provider,
        parser_model=args.parser_model if parser_provider else None,
    )

    print(
        Fore.GREEN +
        f"💾 Saved data for game {game_count} "
        f"(rounds: {round_count}, moves: {len(executed_moves)})"
    )
    
    # Update session stats (now includes round-level aggregates)
    save_session_stats(
        log_dir,
        game_count=game_count,
        total_score=total_score,
        total_steps=total_steps,
        game_scores=game_scores,
        empty_steps=empty_steps,
        something_is_wrong_steps=something_is_wrong_steps,
        valid_steps=valid_steps,
        invalid_reversals=invalid_reversals,
        time_stats=time_stats,
        token_stats=token_stats,
        round_counts=round_counts,
        total_rounds=total_rounds,
    )
    
    return game_count, total_score, total_steps, game_scores, round_count, time_stats, token_stats, valid_steps, invalid_reversals, empty_steps, something_is_wrong_steps

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
        - something_is_wrong_steps: Total something_is_wrong steps across all games
        - valid_steps: Total valid steps across all games (optional)
        - invalid_reversals: Total invalid reversals across all games (optional)
    """
    from utils.game_stats_utils import save_session_stats
    import os
    
    # Extract statistics
    log_dir = stats_info["log_dir"]
    game_count = stats_info["game_count"]
    total_score = stats_info["total_score"]
    total_steps = stats_info["total_steps"]
    game_scores = stats_info["game_scores"]
    empty_steps = stats_info["empty_steps"]
    something_is_wrong_steps = stats_info["something_is_wrong_steps"]
    valid_steps = stats_info.get("valid_steps", 0)
    invalid_reversals = stats_info.get("invalid_reversals", 0)
    
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
    
    # Get token stats specifically from the game manager if available
    if "token_stats" in stats_info:
        token_stats = stats_info["token_stats"]
        
    # Get time stats specifically from the game manager if available
    if "time_stats" in stats_info:
        time_stats = stats_info["time_stats"]
    
    # Get round counts and total rounds
    round_counts = stats_info.get("round_counts", [])
    total_rounds = stats_info.get("total_rounds", 0)
    
    # Save session statistics to summary file
    save_session_stats(
        log_dir, 
        game_count=game_count, 
        total_score=total_score, 
        total_steps=total_steps, 
        game_scores=game_scores, 
        empty_steps=empty_steps, 
        something_is_wrong_steps=something_is_wrong_steps,
        valid_steps=valid_steps,
        invalid_reversals=invalid_reversals,
        time_stats=time_stats,
        token_stats=token_stats,
        round_counts=round_counts,
        total_rounds=total_rounds,
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
    print(Fore.GREEN + f"📈 SOMETHING_IS_WRONG steps: {something_is_wrong_steps}")
    print(Fore.GREEN + f"📈 Valid Steps: {valid_steps}")
    print(Fore.GREEN + f"📈 Invalid Reversals: {invalid_reversals}")
    
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
    from utils.game_stats_utils import save_experiment_info_json
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
    
    # Skip if GUI disabled or pygame has already been quit
    if not game_manager.use_gui or not pygame.get_init():
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
                game_manager.consecutive_invalid_reversals = 0  # Reset counter
                game_manager.current_game_moves = []  # Reset moves for new game
                print(Fore.GREEN + "🔄 Game reset") 

