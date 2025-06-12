"""
Core game loop module for the Snake game.
Handles the main game execution logic and LLM interactions.
"""

import time
import traceback
import pygame
import os
import json
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
        while game_manager.running and game_manager.game_count < game_manager.args.max_games:
            # Process player input and system events
            process_events(game_manager)
            
            if game_manager.game_active:
                try:
                    # Start tracking game movement time for analytics
                    game_manager.game.game_state.record_game_movement_start()
                    
                    # Check if we need a new plan from the LLM
                    if game_manager.need_new_plan:
                        # Mark that we're waiting for a plan to prevent re-execution of moves
                        game_manager.awaiting_plan = True
                        # Get next move from LLM
                        next_move, game_manager.game_active = get_llm_response(game_manager)
                        
                        # We now have a response, so we're no longer waiting
                        game_manager.awaiting_plan = False
                        # Set flag to avoid requesting another plan until needed
                        game_manager.need_new_plan = False
                        
                        # Initialize apple_eaten for use in this block
                        apple_eaten = False
                        
                        # Execute the move if valid and game is still active
                        if next_move and game_manager.game_active:
                            # Update UI to show LLM response and planned moves
                            game_manager.game.draw()
                            
                            # Execute the move 3 second after displaying the LLM response
                            if game_manager.use_gui:
                                time.sleep(3)
                            game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                            
                            # Check if maximum steps limit has been reached AFTER the move
                            if check_max_steps(game_manager.game, game_manager.args.max_steps):
                                game_manager.game_active = False
                                game_manager.game.game_state.record_game_end("MAX_STEPS")
                            
                            # Update UI to show the new state after move
                            game_manager.game.draw()
                            
                            # Reset error tracking on successful move, but NOT empty move tracking
                            game_manager.consecutive_errors = 0
                            
                            # Standard pause between moves for gameplay rhythm
                            game_manager.game.game_state.record_waiting_start()
                            # Only sleep if there's a non-zero pause time
                            pause_time = game_manager.get_pause_between_moves()
                            if pause_time > 0 and game_manager.use_gui:
                                time.sleep(pause_time)
                            game_manager.game.game_state.record_waiting_end()
                        else:
                            # Handle the case where no valid move was found
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            
                            # Let record_empty_move handle the steps counter - it increments steps internally
                            game_manager.game.game_state.record_empty_move()
                            
                            # Update empty_steps counter at manager level for live stats (optional)
                            game_manager.empty_steps = game_manager.game.game_state.empty_steps
                            
                            # Record for analysis
                            game_manager.current_game_moves.append("EMPTY")
                            game_manager.game.game_state.moves.append("EMPTY")
                            
                            # Track consecutive empty moves
                            # Only increment this counter for actual empty moves, not for LLM errors
                            game_manager.consecutive_empty_steps += 1
                            print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_empty_moves_allowed}")
                            
                            # End game if too many consecutive empty moves
                            if game_manager.consecutive_empty_steps >= game_manager.args.max_empty_moves_allowed:
                                print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_empty_moves_allowed}). Game over.")
                                game_manager.game_active = False
                                game_manager.game.last_collision_type = 'empty_moves'
                                game_manager.game.game_state.record_game_end("MAX_EMPTY_MOVES_REACHED")
                        
                        # End movement time tracking
                        game_manager.game.game_state.record_game_movement_end()
                        
                    else:
                        # Skip executing planned moves if we're waiting for a new plan
                        if game_manager.awaiting_plan:
                            # Still waiting for LLM - nothing to execute this tick
                            # Close the movement-timer that was opened at the top of this loop
                            game_manager.game.game_state.record_game_movement_end()
                            continue
                        
                        # Execute the next move from previously planned moves
                        next_move = game_manager.game.get_next_planned_move()
                        
                        if next_move:
                            
                            # Record move for logging (but game_state.record_move will be called in make_move)
                            # No need to add to game_state.moves here as that will be done in make_move
                            game_manager.current_game_moves.append(next_move)
                            
                            # Update UI before executing the move
                            game_manager.game.draw()
                            
                            # Execute the move immediately
                            game_manager.game_active, apple_eaten = game_manager.game.make_move(next_move)
                            
                            # Check max steps limit AFTER the move
                            if check_max_steps(game_manager.game, game_manager.args.max_steps):
                                game_manager.game_active = False
                                game_manager.game.game_state.record_game_end("MAX_STEPS")
                            
                            # Update UI after the move
                            game_manager.game.draw()
                            
                            # Reset error tracking on successful move, but NOT empty move tracking
                            game_manager.consecutive_errors = 0
                            
                            # Request new plan if apple was eaten AND no more planned moves
                            if apple_eaten:
                                # Do NOT increment round_count when an apple is eaten
                                # Rounds should only be incremented when we get a new plan from the LLM
                                
                                # Only request new plan if there are no more planned moves
                                if not game_manager.game.planned_moves:
                                    print(Fore.YELLOW + "No more planned moves, requesting new plan.")
                                    game_manager.need_new_plan = True
                                else:
                                    print(Fore.CYAN + f"Continuing with {len(game_manager.game.planned_moves)} remaining planned moves for this round.")
                            
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
                            # No more planned moves in the current round, we need a new plan
                            game_manager.need_new_plan = True
                            print("🔄 No more planned moves in the current round, requesting new plan.")
                    
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
                            "token_stats": game_manager.token_stats,
                            "valid_steps": getattr(game_manager, "valid_steps", 0),
                            "invalid_reversals": getattr(game_manager, "invalid_reversals", 0),
                            "empty_steps": getattr(game_manager, "empty_steps", 0),
                            "error_steps": getattr(game_manager, "error_steps", 0)
                        }
                        
                        game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.time_stats, game_manager.token_stats, game_manager.valid_steps, game_manager.invalid_reversals, game_manager.empty_steps, game_manager.error_steps = process_game_over(
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
                            valid_steps=game_manager.valid_steps,
                            invalid_reversals=game_manager.invalid_reversals,
                            time_stats=game_manager.time_stats,
                            token_stats=game_manager.token_stats
                        )
                        
                        # Reset for next game
                        game_manager.need_new_plan = True
                        game_manager.game_active = True
                        game_manager.current_game_moves = []
                        
                        # Reset round_count to 1 for the new game
                        # This ensures proper round counting for each game
                        game_manager.round_count = 1
                        
                        # Update summary.json with the latest configuration
                        summary_path = os.path.join(game_manager.log_dir, "summary.json")
                        try:
                            if os.path.exists(summary_path):
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    summary_data = json.load(f)
                                
                                # Update max_games in configuration
                                if 'configuration' in summary_data:
                                    summary_data['configuration']['max_games'] = game_manager.args.max_games
                                    summary_data['configuration']['no_gui'] = game_manager.args.no_gui
                                    
                                    # Remove the continue_with_game_in_dir entry since it's confusing in the configuration
                                    if 'continue_with_game_in_dir' in summary_data['configuration']:
                                        del summary_data['configuration']['continue_with_game_in_dir']
                                
                                # Save the updated configuration
                                with open(summary_path, 'w', encoding='utf-8') as f:
                                    json.dump(summary_data, f, indent=2)
                        except Exception as e:
                            print(Fore.YELLOW + f"⚠️ Warning: Could not update configuration in summary.json: {e}")
                        
                        # Reset game state and counters, but preserve the score
                        # Only reset the game positions and movement-related variables
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
                        "token_stats": game_manager.token_stats,
                        "valid_steps": getattr(game_manager, "valid_steps", 0),
                        "invalid_reversals": getattr(game_manager, "invalid_reversals", 0),
                        "empty_steps": getattr(game_manager, "empty_steps", 0),
                        "error_steps": getattr(game_manager, "error_steps", 0)
                    }
                    
                    game_manager.game_active, game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.previous_parser_usage, game_manager.consecutive_errors, game_manager.time_stats, game_manager.token_stats, game_manager.valid_steps, game_manager.invalid_reversals, game_manager.empty_steps, game_manager.error_steps = handle_error(
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
                        valid_steps=game_manager.valid_steps,
                        invalid_reversals=game_manager.invalid_reversals,
                        time_stats=game_manager.time_stats,
                        token_stats=game_manager.token_stats
                    )
                    
                    # Prepare for next game if not at limit
                    if game_manager.game_count < game_manager.args.max_games and not game_manager.game_active:
                        # Only use pygame.time.delay if GUI is active
                        if game_manager.use_gui:
                            pygame.time.delay(1000)  # Brief pause for user visibility
                            
                        # Update summary.json with the latest configuration
                        summary_path = os.path.join(game_manager.log_dir, "summary.json")
                        try:
                            if os.path.exists(summary_path):
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    summary_data = json.load(f)
                                
                                # Update max_games in configuration
                                if 'configuration' in summary_data:
                                    summary_data['configuration']['max_games'] = game_manager.args.max_games
                                    summary_data['configuration']['no_gui'] = game_manager.args.no_gui
                                    
                                    # Remove the continue_with_game_in_dir entry since it's confusing in the configuration
                                    if 'continue_with_game_in_dir' in summary_data['configuration']:
                                        del summary_data['configuration']['continue_with_game_in_dir']
                                
                                # Save the updated configuration
                                with open(summary_path, 'w', encoding='utf-8') as f:
                                    json.dump(summary_data, f, indent=2)
                        except Exception as e:
                            print(Fore.YELLOW + f"⚠️ Warning: Could not update configuration in summary.json: {e}")
                                
                        game_manager.game.reset()
                        game_manager.game_active = True
                        game_manager.need_new_plan = True
                        game_manager.current_game_moves = []
                        game_manager.consecutive_errors = 0
                        print(Fore.GREEN + f"🔄 Starting game {game_manager.game_count + 1}/{game_manager.args.max_games}")
            
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
            "max_empty_moves_allowed": game_manager.args.max_empty_moves_allowed,
            "max_consecutive_errors_allowed": game_manager.args.max_consecutive_errors_allowed
        }
        
        report_final_statistics(stats_info)
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Ensure pygame is properly shut down
        pygame.quit() 