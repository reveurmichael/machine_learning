"""
Core game loop module for the Snake game.
Handles the main game execution logic and LLM interactions.
"""

import time
import traceback
import pygame
from colorama import Fore
from utils.game_manager_utils import check_max_steps, process_game_over, handle_error, process_events
from utils.json_utils import save_session_stats
from llm.communication_utils import get_llm_response

def run_game_loop(game_manager):
    """Run the main game loop.
    
    Executes the core game logic including:
    - Processing user input events
    - Getting moves from the LLM
    - Executing moves with appropriate timing
    - Handling game state transitions
    
    Args:
        game_manager: The GameManager instance controlling the game session
    """
    try:
        while game_manager.running and game_manager.game_count < game_manager.args.max_game:
            # Process player input and system events
            process_events(game_manager)
            
            if game_manager.game_active:
                try:
                    # Start tracking game movement time for analytics
                    game_manager.game.game_state.record_game_movement_start()
                    
                    # Check if we need a new plan from the LLM
                    if game_manager.need_new_plan:
                        # Get next move from LLM
                        next_move, game_manager.game_active = get_llm_response(game_manager)
                        
                        # Set flag to avoid requesting another plan until needed
                        game_manager.need_new_plan = False
                        
                        # Initialize apple_eaten for use in this block
                        apple_eaten = False
                        
                        # Check if maximum steps limit has been reached
                        if check_max_steps(game_manager.game, game_manager.args.max_steps):
                            game_manager.game_active = False
                            game_manager.game.game_state.record_game_end("MAX_STEPS")
                        # Execute the move if valid and game is still active
                        elif next_move and game_manager.game_active:
                            # Update UI to show LLM response and planned moves
                            game_manager.game.draw()
                            
                            # Remove the 2-second delay before snake moves
                            # Execute the move immediately after displaying the LLM response
                            game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                            
                            # Update UI to show the new state after move
                            game_manager.game.draw()
                            
                            # Reset error tracking on successful move
                            game_manager.consecutive_errors = 0
                            
                            # Standard pause between moves for gameplay rhythm
                            game_manager.game.game_state.record_waiting_start()
                            # Only sleep if there's a non-zero pause time
                            pause_time = game_manager.get_pause_between_moves()
                            if pause_time > 0:
                                time.sleep(pause_time)
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # Handle the case where no valid move was found
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            
                            # Update stats for empty move
                            game_manager.game.steps += 1
                            game_manager.total_steps += 1
                            game_manager.game.game_state.record_empty_move()
                            
                            # Record for analysis
                            game_manager.current_game_moves.append("EMPTY")
                            game_manager.game.game_state.moves.append("EMPTY")
                            
                            # Track consecutive empty moves
                            game_manager.consecutive_empty_steps += 1
                            print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_empty_moves}")
                            
                            # End game if too many consecutive empty moves
                            if game_manager.consecutive_empty_steps >= game_manager.args.max_empty_moves:
                                print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_empty_moves}). Game over.")
                                game_manager.game_active = False
                                game_manager.game.last_collision_type = 'empty_moves'
                                game_manager.game.game_state.record_game_end("EMPTY_MOVES")
                        
                        # End movement time tracking
                        game_manager.game.game_state.record_game_movement_end()
                        
                    else:
                        # Execute the next move from previously planned moves
                        next_move = game_manager.game.get_next_planned_move()
                        
                        if next_move:
                            print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count})")
                            
                            # Record move for logging
                            game_manager.current_game_moves.append(next_move)
                            
                            # Check max steps limit
                            if check_max_steps(game_manager.game, game_manager.args.max_steps):
                                game_manager.game_active = False
                                game_manager.game.game_state.record_game_end("MAX_STEPS")
                            else:
                                # Update UI before executing the move
                                game_manager.game.draw()
                                
                                # Execute the move immediately (remove 2-second delay)
                                game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                                
                                # Update UI after the move
                                game_manager.game.draw()
                            
                            # Reset error tracking on successful move
                            game_manager.consecutive_errors = 0
                            
                            # Request new plan if apple was eaten
                            if apple_eaten:
                                print(Fore.GREEN + "🍎 Apple eaten! Requesting new plan.")
                                game_manager.need_new_plan = True
                            
                            # End movement time tracking
                            game_manager.game.game_state.record_game_movement_end()
                            
                            # Standard pause between moves
                            game_manager.game.game_state.record_waiting_start()
                            # Only sleep if there's a non-zero pause time
                            pause_time = game_manager.get_pause_between_moves()
                            if pause_time > 0:
                                time.sleep(pause_time)
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # No more planned moves available, request new plan
                            game_manager.need_new_plan = True
                    
                    # Handle game over state
                    if not game_manager.game_active:
                        game_state_info = {
                            "game_active": game_manager.game_active,
                            "game_count": game_manager.game_count,
                            "total_score": game_manager.total_score,
                            "total_steps": game_manager.total_steps,
                            "game_scores": game_manager.game_scores,
                            "round_count": game_manager.round_count,
                            "args": game_manager.args,
                            "log_dir": game_manager.log_dir,
                            "current_game_moves": game_manager.current_game_moves,
                            "next_move": next_move,
                            "time_stats": game_manager.time_stats,
                            "token_stats": game_manager.token_stats
                        }
                        
                        game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.time_stats, game_manager.token_stats = process_game_over(
                            game_manager.game,
                            game_state_info
                        )
                        
                        # Make sure to update session stats after processing game over
                        save_session_stats(
                            game_manager.log_dir,
                            game_count=game_manager.game_count,
                            total_score=game_manager.total_score,
                            total_steps=game_manager.total_steps,
                            parser_usage_count=game_manager.parser_usage_count,
                            game_scores=game_manager.game_scores,
                            empty_steps=game_manager.empty_steps,
                            error_steps=game_manager.error_steps,
                            time_stats=game_manager.time_stats,
                            token_stats=game_manager.token_stats
                        )
                        
                        # Reset for next game
                        game_manager.need_new_plan = True
                        game_manager.game_active = True
                        game_manager.current_game_moves = []
                        
                        # Reset game state and counters
                        game_manager.game.reset()
                        game_manager.consecutive_empty_steps = 0
                        game_manager.consecutive_errors = 0
                    
                    # Ensure UI is updated
                    game_manager.game.draw()
                    
                except Exception as e:
                    # Handle errors during gameplay
                    error_info = {
                        "game_active": game_manager.game_active,
                        "game_count": game_manager.game_count,
                        "total_score": game_manager.total_score,
                        "total_steps": game_manager.total_steps,
                        "game_scores": game_manager.game_scores,
                        "round_count": game_manager.round_count,
                        "parser_usage_count": game_manager.parser_usage_count,
                        "previous_parser_usage": game_manager.previous_parser_usage,
                        "log_dir": game_manager.log_dir,
                        "args": game_manager.args,
                        "current_game_moves": game_manager.current_game_moves,
                        "error": e,
                        "consecutive_errors": game_manager.consecutive_errors,
                        "time_stats": game_manager.time_stats,
                        "token_stats": game_manager.token_stats
                    }
                    
                    game_manager.game_active, game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.previous_parser_usage, game_manager.consecutive_errors, game_manager.time_stats, game_manager.token_stats = handle_error(
                        game_manager.game,
                        error_info
                    )
                    
                    # Make sure to update session stats after handling errors
                    save_session_stats(
                        game_manager.log_dir,
                        game_count=game_manager.game_count,
                        total_score=game_manager.total_score,
                        total_steps=game_manager.total_steps,
                        parser_usage_count=game_manager.parser_usage_count,
                        game_scores=game_manager.game_scores,
                        empty_steps=game_manager.empty_steps,
                        error_steps=game_manager.error_steps,
                        time_stats=game_manager.time_stats,
                        token_stats=game_manager.token_stats
                    )
                    
                    # Prepare for next game if not at limit
                    if game_manager.game_count < game_manager.args.max_game and not game_manager.game_active:
                        # Only use pygame.time.delay if GUI is active
                        if game_manager.use_gui:
                            pygame.time.delay(1000)  # Brief pause for user visibility
                        game_manager.game.reset()
                        game_manager.game_active = True
                        game_manager.need_new_plan = True
                        game_manager.current_game_moves = []
                        game_manager.consecutive_errors = 0
                        print(Fore.GREEN + f"🔄 Starting game {game_manager.game_count + 1}/{game_manager.args.max_game}")
            
            # Control frame rate only in GUI mode
            if game_manager.use_gui:
                pygame.time.delay(game_manager.time_delay)
                game_manager.clock.tick(game_manager.time_tick)
        
        # Report final statistics at end of session
        from utils.game_manager_utils import report_final_statistics
        
        stats_info = {
            "log_dir": game_manager.log_dir,
            "game_count": game_manager.game_count,
            "total_score": game_manager.total_score,
            "total_steps": game_manager.total_steps,
            "parser_usage_count": game_manager.parser_usage_count,
            "game_scores": game_manager.game_scores,
            "empty_steps": game_manager.empty_steps,
            "error_steps": game_manager.error_steps,
            "max_empty_moves": game_manager.args.max_empty_moves,
            "max_consecutive_errors_allowed": game_manager.args.max_consecutive_errors_allowed
        }
        
        report_final_statistics(stats_info)
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Ensure pygame is properly shut down
        pygame.quit() 