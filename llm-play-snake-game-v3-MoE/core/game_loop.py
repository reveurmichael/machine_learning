"""Core game loop for the LLM-controlled Snake game.

This module keeps the real-time frame pacing tidy and delegates all
decision-making (LLM calls, move execution, game-over handling …) to private
helpers so the public entry point stays small and readable.
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Tuple

import pygame
from colorama import Fore

from utils.game_manager_utils import check_max_steps, process_game_over, process_events

# -------------------------------
# Type-checking helpers (avoid heavyweight imports at runtime)
# -------------------------------

if TYPE_CHECKING:  # pragma: no cover – imports only needed for static analysis
    from core.game_manager import GameManager
    from llm.communication_utils import get_llm_response  # noqa: F401


# -------------------------------
# Public entry point
# -------------------------------


def run_game_loop(game_manager: "GameManager") -> None:
    """Main orchestrator for a *single session* of games.

    It keeps GUI timing consistent, hands off heavy logic to helper functions
    and terminates once the requested number of games has been played.

    Args:
        game_manager: Active :class:`core.game_manager.GameManager` instance.
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


# --------------------------------
# Heavy-lifting helpers (private to this module) – keep run_game_loop short
# --------------------------------


def _process_active_game(game_manager: "GameManager") -> None:
    """One tick of active gameplay – plan management, move execution, game-over."""

    # --------------------------------
    # 1. Need a fresh LLM plan?
    # --------------------------------
    if game_manager.need_new_plan:
        _request_and_execute_first_move(game_manager)
    else:
        _execute_next_planned_move(game_manager)

    # --------------------------------
    # 2. Post-move book-keeping – game over, UI refresh, etc.
    # --------------------------------
    if not game_manager.game_active:
        _handle_game_over(game_manager)

    # Always refresh the GUI after handling a tick
    game_manager.game.draw()


def _request_and_execute_first_move(game_manager: "GameManager") -> None:
    """Ask the LLM for a new plan and execute its first move if possible."""

    game_manager.awaiting_plan = True
    # Deferred import to mitigate circular-import issues.  The actual call
    # happens only when we *need* an LLM response, so importing here is safe
    # and incurs negligible overhead compared to the network latency.
    from llm.communication_utils import get_llm_response  # local import

    next_move, game_manager.game_active = get_llm_response(game_manager)
    game_manager.awaiting_plan = False
    game_manager.need_new_plan = False

    # ----------------
    # NO_PATH_FOUND sentinel handling.
    # If the previous LLM response indicated NO_PATH_FOUND, record a
    # dedicated move so that analytics and replay can distinguish it from a
    # plain EMPTY step.
    # ----------------

    if game_manager.last_no_path_found:
        _handle_no_path_found(game_manager)
        # _handle_no_path_found() clears the flag internally.

        # If the NO_PATH_FOUND response itself ended the game because the
        # max-consecutive limit was reached, stop processing here to avoid
        # logging additional EMPTY moves after game over.
        if not game_manager.game_active:
            return

    # If game ended for any other reason during the LLM call, stop.
    if not game_manager.game_active:
        return

    # Still active but the LLM produced no usable move → treat as EMPTY unless
    # communication_utils already logged a sentinel for this tick.
    if not next_move:
        if getattr(game_manager, "skip_empty_this_tick", False):
            # Reset flag and bail out without adding another EMPTY move.
            game_manager.skip_empty_this_tick = False
            return
        _handle_no_move(game_manager)
        return

    # IMPORTANT: preview delay of 3 seconds so humans can read the LLM answer
    # For guarenteeing that the human can read the LLM answer before seeing the actual moves 
    game_manager.game.draw()
    if game_manager.use_gui:
        time.sleep(3)

    if game_manager.game.planned_moves:
        game_manager.game.planned_moves.pop(0)

    game_active, apple_eaten = _execute_move(game_manager, next_move)

    if apple_eaten:
        _post_apple_logic(game_manager)


def _execute_next_planned_move(game_manager: "GameManager") -> None:
    """Pop the next pre-computed move off the queue and run it."""

    if game_manager.awaiting_plan:
        return  # still waiting for LLM – nothing to do

    next_move = game_manager.game.get_next_planned_move()
    if not next_move:
        # Round finished – flush & ask for a new plan next tick
        game_manager.finish_round()
        game_manager.need_new_plan = True
        print("🔄 No more planned moves in the current round, requesting new plan.")
        return

    game_manager.current_game_moves.append(next_move)
    game_manager.game.draw()
    game_active, apple_eaten = _execute_move(game_manager, next_move)

    if apple_eaten:
        _post_apple_logic(game_manager)


def _post_apple_logic(game_manager: "GameManager") -> None:
    """Common branch after an apple was eaten – decide if we need a new plan."""

    if game_manager.game.planned_moves:
        print(Fore.CYAN + f"Continuing with {len(game_manager.game.planned_moves)} remaining planned moves for this round.")
    else:
        game_manager.finish_round()
        print(Fore.YELLOW + "No more planned moves, requesting new plan.")
        game_manager.need_new_plan = True


def _handle_no_move(game_manager: "GameManager") -> None:
    """Fallback when LLM returned no valid move."""

    print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")

    game_manager.game.game_state.record_empty_move()
    game_manager.empty_steps = game_manager.game.game_state.empty_steps
    game_manager.current_game_moves.append("EMPTY")

    game_manager.consecutive_empty_steps += 1
    # Any tick that is not an explicit SOMETHING_IS_WRONG error resets that
    # particular streak so the error counter only reflects consecutive
    # parser/LLM failures.
    game_manager.consecutive_something_is_wrong = 0
    print(Fore.YELLOW + f"⚠️ No valid moves found. Empty moves: {game_manager.consecutive_empty_steps}/{game_manager.args.max_consecutive_empty_moves_allowed}")

    # An EMPTY step naturally breaks a sequence of invalid reversals, so make
    # sure to reset that counter here.  Otherwise a pattern like
    # INVALID_REVERSAL → EMPTY → INVALID_REVERSAL would incorrectly be
    # counted as two *consecutive* invalid reversals.
    game_manager.consecutive_invalid_reversals = 0

    if game_manager.consecutive_empty_steps >= game_manager.args.max_consecutive_empty_moves_allowed:
        print(Fore.RED + f"❌ Maximum consecutive empty moves reached ({game_manager.args.max_consecutive_empty_moves_allowed}). Game over.")
        game_manager.game_active = False
        game_manager.game.last_collision_type = 'MAX_CONSECUTIVE_EMPTY_MOVES_REACHED'
        game_manager.game.game_state.record_game_end("MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")

    _apply_empty_move_delay(game_manager)


def _handle_game_over(game_manager: "GameManager") -> None:
    """Delegate heavy game-over processing to utils.game_manager_utils then reset for next game."""

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
        "next_move": None,
        "time_stats": game_manager.time_stats,
        "token_stats": game_manager.token_stats,
        "valid_steps": getattr(game_manager, "valid_steps", 0),
        "invalid_reversals": getattr(game_manager, "invalid_reversals", 0),
        "empty_steps": getattr(game_manager, "empty_steps", 0),
        "something_is_wrong_steps": getattr(game_manager, "something_is_wrong_steps", 0),
        "no_path_found_steps": getattr(game_manager, "no_path_found_steps", 0),
    }

    (
        game_manager.game_count,
        game_manager.total_score,
        game_manager.total_steps,
        game_manager.game_scores,
        game_manager.round_count,
        game_manager.time_stats,
        game_manager.token_stats,
        game_manager.valid_steps,
        game_manager.invalid_reversals,
        game_manager.empty_steps,
        game_manager.something_is_wrong_steps,
        game_manager.no_path_found_steps,
    ) = process_game_over(game_manager.game, game_state_info)

    # Reset per-game flags/counters for the upcoming game
    game_manager.need_new_plan = True
    game_manager.game_active = True
    game_manager.current_game_moves = []
    game_manager.round_count = 1
    game_manager.game.reset()
    game_manager.consecutive_empty_steps = 0
    game_manager.consecutive_something_is_wrong = 0
    game_manager.total_rounds = sum(game_manager.round_counts)


# ----------------------------------------
# Internal utilities (module-private)
# ----------------------------------------

def _execute_move(manager: "GameManager", direction: str) -> Tuple[bool, bool]:
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
        manager.game.game_state.record_game_end("MAX_STEPS_REACHED")

    # ---------------- UI refresh & per-move pause --------------------------
    manager.game.draw()
    pause = manager.get_pause_between_moves()
    if pause > 0:
        time.sleep(pause)

    # Any successful move (even if blocked by invalid reversal) resets the
    # "something is wrong" counter.
    manager.consecutive_something_is_wrong = 0

    # Reset consecutive *empty* move counter when we successfully execute a
    # non-EMPTY move.  Without this, the counter accumulated across unrelated
    # moves and could prematurely trigger the game-over condition even when
    # empty moves were not consecutive.
    if direction != "EMPTY":
        manager.consecutive_empty_steps = 0
        manager.consecutive_no_path_found = 0

    # Return the possibly-updated game_active flag plus apple status.
    manager.game_active = game_active  # maintain historic side-effect
    return game_active, apple_eaten 

# --------------------------------
# Sentinel NO_PATH_FOUND handling – completely separate from EMPTY & ERRORS
# --------------------------------


def _handle_no_path_found(game_manager: "GameManager") -> None:
    """Log a NO_PATH_FOUND step (distinct from EMPTY)."""

    print(Fore.YELLOW + "⚠️ LLM reported NO_PATH_FOUND. Snake stays in place.")

    game_manager.game.game_state.record_no_path_found_move()
    game_manager.current_game_moves.append("NO_PATH_FOUND")

    # Break the SOMETHING_IS_WRONG streak because this tick is a distinct
    # sentinel cause.
    game_manager.consecutive_something_is_wrong = 0

    # A NO_PATH_FOUND tick also breaks any chain of invalid reversals because
    # the snake did not even attempt to move.  Reset that counter to keep the
    # invalid-reversal streak strictly consecutive.
    game_manager.consecutive_invalid_reversals = 0

    # Reset the flag so subsequent EMPTY logic behaves normally.
    game_manager.last_no_path_found = False 

# ------------------------------------------------------------------
# Per-move helpers – sleep handling for EMPTY sentinel
# ------------------------------------------------------------------


def _apply_empty_move_delay(manager: "GameManager") -> None:
    """Apply the pause configured via ``--sleep-after-empty-step`` (minutes).

    Motivation
    ----------
    An **EMPTY** sentinel means the parser did not find *any* valid move in
    the LLM response.  Two common real-world causes are:

    1. *Model overload* – the provider replies with an empty string or
       connection resets; backing off reduces pressure and gives the service
       a chance to recover.
    2. *Rate limiting* – cloud APIs may throttle if we hit them too quickly;
       inserting a pause lowers the request frequency and avoids further
       HTTP 429 or truncated responses.

    By sleeping only on EMPTY (and **not** on the structurally different
    NO_PATH_FOUND sentinel) we retain distinct behaviours: path-finding
    failures are retried immediately, whereas suspected capacity/ratelimit
    issues trigger a back-off.
    """

    pause_min: float = getattr(manager.args, "sleep_after_empty_step", 0.0)

    # Skip pause when disabled or when the preceding turn was NO_PATH_FOUND.
    if pause_min <= 0 or getattr(manager, "last_no_path_found", False):
        return

    plural = "s" if pause_min != 1 else ""
    print(Fore.CYAN + f"⏸️ Sleeping {pause_min} minute{plural} after EMPTY step …")
    import time as _time
    _time.sleep(pause_min * 60) 