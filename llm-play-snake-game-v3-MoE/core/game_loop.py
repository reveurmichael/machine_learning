"""
Core game loop module for the Snake game.
Handles the main game execution logic and LLM interactions.
"""

import time
import traceback
import pygame
from colorama import Fore
from utils.game_manager_utils import check_max_steps, process_game_over, handle_error, process_events
from utils.llm_utils import get_llm_response

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
        while game_manager.running and game_manager.game_count < game_manager.args.max_games:
            # Process player input and system events
            process_events(game_manager)
            
            if game_manager.game_active:
                try:
                    # Start tracking game movement time for analytics
                    game_manager.game.game_state.record_game_movement_start()
                    
                    # Check if we need a new plan from the LLM
                    if game_manager.need_new_plan:
                        # Get the next move from the LLM
                        next_move, game_manager.game_active = get_llm_response(game_manager)
                        
                        # Set flag to avoid requesting another plan until needed
                        game_manager.need_new_plan = False
                        
                        # Check if maximum steps limit has been reached
                        if check_max_steps(game_manager.game, game_manager.args.max_steps):
                            game_manager.game_active = False
                            game_manager.game.game_state.record_game_end("MAX_STEPS")
                        # Execute the move if valid and game is still active
                        elif next_move and game_manager.game_active:
                            # Update UI to show LLM response and planned moves
                            game_manager.game.draw()
                            
                            # Delay to let user see LLM response before snake moves
                            game_manager.game.game_state.record_waiting_start()
                            time.sleep(2.0)  # User-friendly delay for reading LLM plans
                            game_manager.game.game_state.record_waiting_end()
                            
                            # Execute the move
                            game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                            
                            # Update UI to show the new state after move
                            game_manager.game.draw()
                            
                            # Reset error tracking on successful move
                            game_manager.consecutive_errors = 0
                            
                            # Standard pause between moves for gameplay rhythm
                            game_manager.game.game_state.record_waiting_start()
                            time.sleep(game_manager.get_pause_between_moves())
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # Handle the case where no valid move was found
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            
                            # Update stats for empty move
                            game_manager.game.steps += 1
                            game_manager.total_steps += 1
                            game_manager.game.game_state.record_empty_move()
                            
                            # Record for replay and analysis
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
                        
                        # Update round counter
                        game_manager.round_count += 1
                    else:
                        # Execute the next move from previously planned moves
                        next_move = game_manager.game.get_next_planned_move()
                        
                        if next_move:
                            print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {game_manager.game_count+1}, Round {game_manager.round_count+1})")
                            
                            # Record move for logging and replay
                            game_manager.current_game_moves.append(next_move)
                            
                            # Check max steps limit
                            if check_max_steps(game_manager.game, game_manager.args.max_steps):
                                game_manager.game_active = False
                                game_manager.game.game_state.record_game_end("MAX_STEPS")
                            else:
                                # Update UI before executing the move
                                game_manager.game.draw()
                                
                                # Delay to let user see which planned move will be executed
                                game_manager.game.game_state.record_waiting_start()
                                time.sleep(2.0)  # User-friendly delay for move visibility
                                game_manager.game.game_state.record_waiting_end()
                                
                                # Execute the move
                                game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                                
                                # Update UI after the move
                                game_manager.game.draw()
                            
                            # Reset error tracking on successful move
                            game_manager.consecutive_errors = 0
                            
                            # Request new plan if apple was eaten
                            if apple_eaten:
                                print(Fore.GREEN + f"🍎 Apple eaten! Requesting new plan.")
                                game_manager.need_new_plan = True
                            
                            # Update round counter
                            game_manager.round_count += 1
                            
                            # End movement time tracking
                            game_manager.game.game_state.record_game_movement_end()
                            
                            # Standard pause between moves
                            game_manager.game.game_state.record_waiting_start()
                            time.sleep(game_manager.get_pause_between_moves())
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # No more planned moves available, request new plan
                            game_manager.need_new_plan = True
                    
                    # Handle game over state
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
                        
                        # Reset game state and counters
                        game_manager.game.reset()
                        game_manager.consecutive_empty_steps = 0
                        game_manager.consecutive_errors = 0
                    
                    # Ensure UI is updated
                    game_manager.game.draw()
                    
                except Exception as e:
                    # Handle errors during gameplay
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
                        game_manager.consecutive_errors
                    )
                    
                    # Prepare for next game if not at limit
                    if game_manager.game_count < game_manager.args.max_games and not game_manager.game_active:
                        pygame.time.delay(1000)  # Brief pause for user visibility
                        game_manager.game.reset()
                        game_manager.game_active = True
                        game_manager.need_new_plan = True
                        game_manager.current_game_moves = []
                        game_manager.consecutive_errors = 0
                        print(Fore.GREEN + f"🔄 Starting game {game_manager.game_count + 1}/{game_manager.args.max_games}")
            
            # Control frame rate
            pygame.time.delay(game_manager.time_delay)
            game_manager.clock.tick(game_manager.time_tick)
        
        # Report final statistics at end of session
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
        # Ensure pygame is properly shut down
        pygame.quit() 