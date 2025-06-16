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
from utils.game_manager_utils import check_max_steps, process_game_over, process_events
from llm.communication_utils import get_llm_response
from typing import Tuple

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
            
            if game_manager.game_active and game_manager.game is not None:
                try:
                    # (Timer removed – we no longer track per-move wall-clock sections)
                    
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
                            
                            # Execute the move 3 seconds after displaying the LLM response (preview delay)
                            if game_manager.use_gui:
                                time.sleep(3)
                            
                            # After preview, remove the first move from the on-screen list so it reflects
                            # the *remaining* plan.  We keep it during the preview so the player can see the
                            # full plan sent by the LLM.
                            if game_manager.game.planned_moves:
                                game_manager.game.planned_moves.pop(0)
                            
                            # Execute the move and all associated bookkeeping in one place
                            game_manager.game_active, apple_eaten = _execute_move(game_manager, next_move)

                            # Request new plan if apple was eaten AND no more planned moves
                            if apple_eaten:
                                # Do NOT increment round_count when an apple is eaten
                                # Rounds should only be incremented when we get a new plan from the LLM
                                
                                # Only request new plan if there are no more planned moves
                                if not game_manager.game.planned_moves:
                                    # Round ends simultaneously with an apple, finish it before asking for a new plan
                                    game_manager.finish_round()
                                    print(Fore.YELLOW + "No more planned moves, requesting new plan.")
                                    game_manager.need_new_plan = True
                                else:
                                    print(Fore.CYAN + f"Continuing with {len(game_manager.game.planned_moves)} remaining planned moves for this round.")
                            
                        else:
                            # Handle the case where no valid move was found
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            
                            # Let record_empty_move handle the steps counter - it increments steps internally
                            game_manager.game.game_state.record_empty_move()
                            
                            # Update empty_steps counter at manager level for live stats (optional)
                            game_manager.empty_steps = game_manager.game.game_state.empty_steps
                            
                            # Record for analysis (game_state already appends "EMPTY")
                            game_manager.current_game_moves.append("EMPTY")
                            
                            # Track consecutive empty moves
                            # Only increment this counter for actual empty moves, not for LLM errors
                            game_manager.consecutive_empty_steps += 1
                            print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {game_manager.consecutive_empty_steps}/{game_manager.args.max_consecutive_empty_moves_allowed}")
                            
                            # End game if too many consecutive empty moves
                            if game_manager.consecutive_empty_steps >= game_manager.args.max_consecutive_empty_moves_allowed:
                                print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_consecutive_empty_moves_allowed}). Game over.")
                                game_manager.game_active = False
                                game_manager.game.last_collision_type = 'MAX_CONSECUTIVE_EMPTY_MOVES_REACHED'
                                game_manager.game.game_state.record_game_end("MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")
                        
                    else:
                        # Skip executing planned moves if we're waiting for a new plan
                        if game_manager.awaiting_plan:
                            # Still waiting for LLM - nothing to execute this tick
                            # Close the movement-timer that was opened at the top of this loop
                            # timer removed
                            continue
                        
                        # Execute the next move from previously planned moves
                        next_move = game_manager.game.get_next_planned_move()
                        
                        if next_move:
                            
                            # Record move for logging (but game_state.record_move will be called in make_move)
                            # No need to add to game_state.moves here as that will be done in make_move
                            game_manager.current_game_moves.append(next_move)
                            
                            # Update UI before executing the move
                            game_manager.game.draw()
                            
                            # Execute and bookkeeping via shared helper
                            game_manager.game_active, apple_eaten = _execute_move(game_manager, next_move)

                            # Request new plan if apple was eaten AND no more planned moves
                            if apple_eaten:
                                # Do NOT increment round_count when an apple is eaten
                                # Rounds should only be incremented when we get a new plan from the LLM
                                
                                # Only request new plan if there are no more planned moves
                                if not game_manager.game.planned_moves:
                                    # Round ends simultaneously with an apple, finish it before asking for a new plan
                                    game_manager.finish_round()
                                    print(Fore.YELLOW + "No more planned moves, requesting new plan.")
                                    game_manager.need_new_plan = True
                                else:
                                    print(Fore.CYAN + f"Continuing with {len(game_manager.game.planned_moves)} remaining planned moves for this round.")
                            
                        else:
                            # The round is finished – flush current round and bump the counter **before**
                            # we request the next LLM plan. This keeps prompts/responses, JSON logs and
                            # console banners on the very same round number (single source of truth).
                            game_manager.finish_round()

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
                            "round_counts": game_manager.round_counts,
                            "args": game_manager.args,
                            "log_dir": game_manager.log_dir,
                            "current_game_moves": game_manager.current_game_moves,
                            "next_move": next_move,
                            "time_stats": game_manager.time_stats,
                            "token_stats": game_manager.token_stats,
                            "valid_steps": getattr(game_manager, "valid_steps", 0),
                            "invalid_reversals": getattr(game_manager, "invalid_reversals", 0),
                            "empty_steps": getattr(game_manager, "empty_steps", 0),
                            "something_is_wrong_steps": getattr(game_manager, "something_is_wrong_steps", 0)
                        }
                        
                        game_manager.game_count, game_manager.total_score, game_manager.total_steps, game_manager.game_scores, game_manager.round_count, game_manager.time_stats, game_manager.token_stats, game_manager.valid_steps, game_manager.invalid_reversals, game_manager.empty_steps, game_manager.something_is_wrong_steps = process_game_over(
                            game_manager.game,
                            game_state_info
                        )
                        
                        # Session statistics are now updated within process_game_over()
                        
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
                        game_manager.consecutive_something_is_wrong = 0
                        
                        # Update total_rounds
                        game_manager.total_rounds = sum(game_manager.round_counts)
                    
                    # Ensure UI is updated
                    game_manager.game.draw()
                    
                except Exception:
                    pass
            # Control frame rate only in GUI mode
            if game_manager.use_gui:
                pygame.time.delay(game_manager.time_delay)
                game_manager.clock.tick(game_manager.time_tick)
        
        # Final statistics are now reported once, from GameManager.report_final_statistics().
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Ensure pygame is properly shut down
        pygame.quit()

# ---------------------------------------------------------------------------
# Internal utilities (module-private)
# ---------------------------------------------------------------------------

def _execute_move(manager, direction: str) -> Tuple[bool, bool]:
    """Run *one* snake move and handle all common bookkeeping.

    This wraps the repeated code that appears in both the *new-plan* branch
    (first move after an LLM response) and the *planned-moves* branch.  It
    purposefully contains **no** game-flow decisions so it can be called from
    either place without altering behaviour.

    Args:
        manager: GameManager instance (passed explicitly for clarity).
        direction: The direction key to execute ("UP", "DOWN", etc.).

    Returns:
        Tuple(bool game_active, bool apple_eaten) – same as GameController.
    """

    # Keep a before-snapshot of invalid-reversal counter so we can detect
    # whether the upcoming make_move() call was blocked.
    prev_invalid_rev = manager.game.game_state.invalid_reversals

    game_active, apple_eaten = manager.game.make_move(direction)

    # ---------------- Invalid reversal tracking -----------------
    if manager.game.game_state.invalid_reversals > prev_invalid_rev:
        manager.consecutive_invalid_reversals += 1
        print(
            Fore.YELLOW +
            f"⚠️ Invalid reversal detected. Consecutive invalid reversals: "
            f"{manager.consecutive_invalid_reversals}/"
            f"{manager.args.max_consecutive_invalid_reversals_allowed}"
        )
    else:
        manager.consecutive_invalid_reversals = 0

    # Game-over due to invalid-reversal threshold
    if (
        manager.consecutive_invalid_reversals >=
        manager.args.max_consecutive_invalid_reversals_allowed
    ):
        print(
            Fore.RED +
            f"❌ Maximum consecutive invalid reversals reached "
            f"({manager.args.max_consecutive_invalid_reversals_allowed}). Game over."
        )
        game_active = False
        manager.game.last_collision_type = 'MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED'
        manager.game.game_state.record_game_end("MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED")

    # ---------------- Max-step check ---------------------------------------
    if game_active and check_max_steps(manager.game, manager.args.max_steps):
        game_active = False
        manager.game.game_state.record_game_end("MAX_STEPS")

    # ---------------- UI refresh & per-move pause --------------------------
    manager.game.draw()
    pause = manager.get_pause_between_moves()
    if pause > 0:
        time.sleep(pause)

    # Any successful move (even if blocked by invalid reversal) resets the
    # "something is wrong" counter.
    manager.consecutive_something_is_wrong = 0

    # Return the possibly-updated game_active flag plus apple status.
    manager.game_active = game_active  # maintain historic side-effect
    return game_active, apple_eaten 