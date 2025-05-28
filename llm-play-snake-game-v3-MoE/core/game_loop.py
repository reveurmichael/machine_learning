"""
Core game loop module for the Snake game.
Handles the main game loop and interactions with the LLM.
"""

import time
import traceback
import pygame
from colorama import Fore
from utils.game_manager_utils import check_max_steps, process_game_over, handle_error, process_events
from utils.llm_utils import get_llm_response

def run_game_loop(game_manager):
    """Run the main game loop.
    
    Args:
        game_manager: The GameManager instance
    """
    try:
        while game_manager.running and game_manager.game_count < game_manager.args.max_games:
            # Handle events
            process_events(game_manager)
            
            if game_manager.game_active:
                try:
                    # Start tracking game movement time
                    game_manager.game.game_state.record_game_movement_start()
                    
                    # Check if we need a new plan
                    if game_manager.need_new_plan:
                        # Get the next move from the LLM
                        next_move, game_manager.game_active = get_llm_response(game_manager)
                        
                        # We now have a new plan, so don't request another one until we need it
                        game_manager.need_new_plan = False
                        
                        # Check if we've reached max steps
                        if check_max_steps(game_manager.game, game_manager.args.max_steps):
                            game_manager.game_active = False
                            game_manager.game.game_state.record_game_end("MAX_STEPS")
                        # Only execute the move if we got a valid direction and game is still active
                        elif next_move and game_manager.game_active:
                            # Execute the move and check if game continues
                            game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                            
                            # Make sure UI is updated with current game state
                            game_manager.game.draw()
                            
                            # Reset consecutive errors counter on successful move
                            game_manager.consecutive_errors = 0
                            
                            # Add a pause between the first move after getting a new plan
                            # This ensures the snake doesn't move too quickly after receiving an LLM response
                            # Track waiting time for this pause
                            game_manager.game.game_state.record_waiting_start()
                            time.sleep(game_manager.get_pause_between_moves())
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # No valid move found, but we still count this as a round
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            # No movement, so the game remains active and no apple is eaten
                            game_manager.game.steps += 1
                            game_manager.total_steps += 1
                            game_manager.game.game_state.record_empty_move()
                        
                        # End tracking game movement time
                        game_manager.game.game_state.record_game_movement_end()
                        
                        # Increment round count
                        game_manager.round_count += 1
                    else:
                        # Get the next move from the existing plan
                        next_move = game_manager.game.get_next_planned_move()
                        
                        # If we have a move, execute it
                        if next_move:
                            print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count+1})")
                            
                            # Record the move
                            game_manager.current_game_moves.append(next_move)
                            
                            # Check if we've reached max steps
                            if check_max_steps(game_manager.game, game_manager.args.max_steps):
                                game_manager.game_active = False
                                game_manager.game.game_state.record_game_end("MAX_STEPS")
                            else:
                                # Execute the move and check if game continues
                                game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                                
                                # Make sure UI is updated with current game state and planned moves
                                game_manager.game.draw()
                            
                            # Reset consecutive errors counter on successful move
                            game_manager.consecutive_errors = 0
                            
                            # If we've eaten an apple, request a new plan
                            if apple_eaten:
                                print(Fore.GREEN + f"🍎 Apple eaten! Requesting new plan.")
                                game_manager.need_new_plan = True
                            
                            # Increment round count
                            game_manager.round_count += 1
                            
                            # End tracking game movement time
                            game_manager.game.game_state.record_game_movement_end()
                            
                            # Start tracking waiting time (for pause between moves)
                            game_manager.game.game_state.record_waiting_start()
                            
                            # Pause between moves for visualization
                            time.sleep(game_manager.get_pause_between_moves())
                            
                            # End tracking waiting time
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # No more planned moves, request a new plan
                            game_manager.need_new_plan = True
                    
                    # Check if game is over
                    if not game_manager.game_active:
                        game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count = process_game_over(
                            game_manager.game,
                            game_manager.game_active,
                            game_manager.game_count,
                            game_manager.total_score,
                            game_manager.total_steps,
                            game_manager.game_scores,
                            game_manager.round_count,
                            game_manager.args,
                            game_manager.log_dir,
                            game_manager.current_game_moves,
                            next_move
                        )
                        
                        # Reset for next game
                        game_manager.need_new_plan = True
                        game_manager.game_active = True
                        game_manager.current_game_moves = []
                        
                        # Reset the game
                        game_manager.game.reset()
                        game_manager.consecutive_empty_steps = 0  # Reset on new game
                        game_manager.consecutive_errors = 0  # Reset on new game
                    
                    # Draw the current state
                    game_manager.game.draw()
                    
                except Exception as e:
                    game_manager.game_active, game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.previous_parser_usage, game_manager.consecutive_errors = handle_error(
                        game_manager.game, 
                        game_manager.game_active,
                        game_manager.game_count,
                        game_manager.total_score,
                        game_manager.total_steps,
                        game_manager.game_scores,
                        game_manager.round_count,
                        game_manager.parser_usage_count,
                        game_manager.previous_parser_usage,
                        game_manager.log_dir,
                        game_manager.args,
                        game_manager.current_game_moves,
                        e,
                        game_manager.consecutive_errors  # Pass the consecutive errors counter
                    )
                    
                    # Prepare for next game if we haven't reached the limit
                    if game_manager.game_count < game_manager.args.max_games and not game_manager.game_active:
                        pygame.time.delay(1000)  # Wait 1 second
                        game_manager.game.reset()
                        game_manager.game_active = True
                        game_manager.need_new_plan = True
                        game_manager.current_game_moves = []  # Reset moves for next game
                        game_manager.consecutive_errors = 0  # Reset on new game
                        print(Fore.GREEN + f"🔄 Starting game {game_manager.game_count + 1}/{game_manager.args.max_games}")
            
            # Control game speed
            pygame.time.delay(game_manager.time_delay)
            game_manager.clock.tick(game_manager.time_tick)
        
        # Report final statistics
        from utils.game_manager_utils import report_final_statistics
        report_final_statistics(
            game_manager.log_dir,
            game_manager.game_count,
            game_manager.total_score,
            game_manager.total_steps,
            game_manager.parser_usage_count,
            game_manager.game_scores,
            game_manager.empty_steps,
            game_manager.error_steps,
            game_manager.args.max_empty_moves,
            game_manager.args.max_consecutive_errors_allowed
        )
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        pygame.quit() 