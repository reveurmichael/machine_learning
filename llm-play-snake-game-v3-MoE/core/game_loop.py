"""
Core game loop module for the Snake game.
Handles the main game execution logic and LLM interactions.
"""

import time
import traceback
import pygame
from colorama import Fore
from utils.game_manager_utils import check_max_steps, process_game_over, process_events
from llm.communication_utils import get_llm_response
from typing import Tuple

def run_game_loop(game_manager):
    """Main orchestrator that keeps the frame/GUI timing tidy and delegates
    all decision making to :func:`_process_active_game` so this function stays
    small and readable.

    Args:
        game_manager: The active :class:`core.game_manager.GameManager`.
    """

    try:
        while game_manager.running and game_manager.game_count < game_manager.args.max_games:
            # Handle user / window events first
            process_events(game_manager)

            # Advance the actual game logic (LLM, moves, game-over handling …)
            if game_manager.game_active and game_manager.game is not None:
                _process_active_game(game_manager)

            # GUI timing / FPS – only when a window is displayed
            if game_manager.use_gui:
                pygame.time.delay(game_manager.time_delay)
                game_manager.clock.tick(game_manager.time_tick)

        # Session-level statistics banner printed from GameManager.

    except Exception as exc:
        print(Fore.RED + f"Fatal error: {exc}")
        traceback.print_exc()
    finally:
        pygame.quit()


# ------------------------------------------------------------------
# Heavy-lifting helpers (private to this module) – keep run_game_loop short
# ------------------------------------------------------------------


def _process_active_game(mgr):
    """One tick of active gameplay – plan management, move execution, game-over."""

    # ------------------------------------------------------------------
    # 1. Need a fresh LLM plan?
    # ------------------------------------------------------------------
    if mgr.need_new_plan:
        _request_and_execute_first_move(mgr)
    else:
        _execute_next_planned_move(mgr)

    # ------------------------------------------------------------------
    # 2. Post-move book-keeping – game over, UI refresh, etc.
    # ------------------------------------------------------------------
    if not mgr.game_active:
        _handle_game_over(mgr)

    # Always refresh the GUI after handling a tick
    mgr.game.draw()


def _request_and_execute_first_move(mgr):
    """Ask the LLM for a new plan and execute its first move if possible."""

    mgr.awaiting_plan = True
    next_move, mgr.game_active = get_llm_response(mgr)
    mgr.awaiting_plan = False
    mgr.need_new_plan = False

    if not next_move or not mgr.game_active:
        _handle_no_move(mgr)
        return

    # Optional preview delay so humans can read the LLM answer
    mgr.game.draw()
    if mgr.use_gui:
        time.sleep(3)

    if mgr.game.planned_moves:
        mgr.game.planned_moves.pop(0)

    game_active, apple_eaten = _execute_move(mgr, next_move)

    if apple_eaten:
        _post_apple_logic(mgr)


def _execute_next_planned_move(mgr):
    """Pop the next pre-computed move off the queue and run it."""

    if mgr.awaiting_plan:
        return  # still waiting for LLM – nothing to do

    next_move = mgr.game.get_next_planned_move()
    if not next_move:
        # Round finished – flush & ask for a new plan next tick
        mgr.finish_round()
        mgr.need_new_plan = True
        print("🔄 No more planned moves in the current round, requesting new plan.")
        return

    mgr.current_game_moves.append(next_move)
    mgr.game.draw()
    game_active, apple_eaten = _execute_move(mgr, next_move)

    if apple_eaten:
        _post_apple_logic(mgr)


def _post_apple_logic(mgr):
    """Common branch after an apple was eaten – decide if we need a new plan."""

    if mgr.game.planned_moves:
        print(Fore.CYAN + f"Continuing with {len(mgr.game.planned_moves)} remaining planned moves for this round.")
    else:
        mgr.finish_round()
        print(Fore.YELLOW + "No more planned moves, requesting new plan.")
        mgr.need_new_plan = True


def _handle_no_move(mgr):
    """Fallback when LLM returned no valid move."""

    print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")

    mgr.game.game_state.record_empty_move()
    mgr.empty_steps = mgr.game.game_state.empty_steps
    mgr.current_game_moves.append("EMPTY")

    mgr.consecutive_empty_steps += 1
    print(Fore.YELLOW + f"⚠️ No valid moves found. Empty steps: {mgr.consecutive_empty_steps}/{mgr.args.max_consecutive_empty_moves_allowed}")

    if mgr.consecutive_empty_steps >= mgr.args.max_consecutive_empty_moves_allowed:
        print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({mgr.args.max_consecutive_empty_moves_allowed}). Game over.")
        mgr.game_active = False
        mgr.game.last_collision_type = 'MAX_CONSECUTIVE_EMPTY_MOVES_REACHED'
        mgr.game.game_state.record_game_end("MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")


def _handle_game_over(mgr):
    """Delegate heavy game-over processing to utils.game_manager_utils then reset for next game."""

    game_state_info = {
        "game_active": mgr.game_active,
        "game_count": mgr.game_count,
        "total_score": mgr.total_score,
        "total_steps": mgr.total_steps,
        "game_scores": mgr.game_scores,
        "round_count": mgr.round_count,
        "round_counts": mgr.round_counts,
        "args": mgr.args,
        "log_dir": mgr.log_dir,
        "current_game_moves": mgr.current_game_moves,
        "next_move": None,
        "time_stats": mgr.time_stats,
        "token_stats": mgr.token_stats,
        "valid_steps": getattr(mgr, "valid_steps", 0),
        "invalid_reversals": getattr(mgr, "invalid_reversals", 0),
        "empty_steps": getattr(mgr, "empty_steps", 0),
        "something_is_wrong_steps": getattr(mgr, "something_is_wrong_steps", 0),
    }

    (
        mgr.game_count,
        mgr.total_score,
        mgr.total_steps,
        mgr.game_scores,
        mgr.round_count,
        mgr.time_stats,
        mgr.token_stats,
        mgr.valid_steps,
        mgr.invalid_reversals,
        mgr.empty_steps,
        mgr.something_is_wrong_steps,
    ) = process_game_over(mgr.game, game_state_info)

    # Reset per-game flags/counters for the upcoming game
    mgr.need_new_plan = True
    mgr.game_active = True
    mgr.current_game_moves = []
    mgr.round_count = 1
    mgr.game.reset()
    mgr.consecutive_empty_steps = 0
    mgr.consecutive_something_is_wrong = 0
    mgr.total_rounds = sum(mgr.round_counts)


# ----------------------------------------
# Internal utilities (module-private)
# ----------------------------------------

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