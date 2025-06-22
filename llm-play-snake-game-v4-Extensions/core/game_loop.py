"""Core game loop for the LLM-controlled Snake game.

This module keeps the real-time frame pacing tidy and delegates all
decision-making (LLM calls, move execution, game-over handling …) to private
helpers so the public entry point stays small and readable.

IMPORTANT: KEEP THIS FILE OOP. KEEP THE BASE CLASS AS WELL AS THE DERIVED CLASS.
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Tuple

import pygame
from colorama import Fore
from config.game_constants import PAUSE_PREVIEW_BEFORE_MAKING_FIRST_MOVE_SECONDS
from core.game_manager_helper import BaseGameManagerHelper, GameManagerHelper


# ---------------------
# Type-checking helpers (avoid heavyweight imports at runtime)
# ---------------------

if TYPE_CHECKING:  # pragma: no cover – imports only needed for static analysis
    from core.game_manager import BaseGameManager  # use narrow, LLM-agnostic base class
    from llm.communication_utils import get_llm_response  # noqa: F401


# ---------------------
# OOP refactor – GameLoop class encapsulates the helpers
# ---------------------


class BaseGameLoop:
    """Orchestrate **one entire session** of Snake games.

    Key design goals (identical to the original functional version):

    1. **Frame pacing / GUI timing** remain the responsibility of this layer
        – the heavy-weight logic lives in private helpers so the public
        :py:meth:`run` method stays readable.
    2. **LLM-agnostic**: the base class *does not* import or reference any
        network code directly.  Task-0 subclasses keep the HTTP / websocket
        specifics so future heuristic / RL tasks can reuse the loop without
        modification.
    3. **Open/Closed Principle**: every significant step (_new-plan_,
        _execute-move_, _apple-logic_, …) is factored into its own protected
        method – subclasses may override just the bits they need.

    The body below is a 1-to-1 transcription of the historical procedural
    implementation – only wrapped in a class to facilitate subclassing.
    """

    def __init__(self, manager: "BaseGameManager") -> None:
        self.manager = manager

    # ---- Public entry point ---------------------

    def run(self) -> None:
        """Execute the session until the requested number of games completes."""

        manager = self.manager  # local alias for brevity

        try:
            while manager.running and manager.game_count < manager.args.max_games:
                # Handle OS / pygame events first so the window remains responsive.
                BaseGameManagerHelper.process_events(manager)

                if manager.game_active and manager.game is not None:
                    if getattr(manager, "agent", None) is not None:
                        self._process_agent_game()
                    else:
                        self._process_active_game()

                if manager.use_gui:
                    pygame.time.delay(manager.time_delay)
                    manager.clock.tick(manager.time_tick)

        except Exception as exc:  # pragma: no cover – safety net
            print(Fore.RED + f"Fatal error: {exc}")
            traceback.print_exc()
        finally:
            pygame.quit()

    # ---------------------
    # Former top-level helpers – now instance methods (identical bodies)
    # ---------------------

    def _process_active_game(self) -> None:
        """One tick of live gameplay for *LLM / planned-move* sessions."""
        manager = self.manager

        # If a new plan is needed, fetch it first. This method will set
        # the manager's internal state (`planned_moves`).
        if manager.need_new_plan:
            self._get_new_plan()

        # Now, execute the next move from the queue, regardless of whether
        # the plan is fresh or we're continuing an old one. This unifies
        # the execution path and kills the "double-first-move" bug.
        self._execute_next_planned_move()

        if not manager.game_active:
            self._handle_game_over()

        manager.game.draw()

    def _get_new_plan(self) -> None:
        """Subclasses (LLM, heuristic, RL…) must implement their own plan-fetching logic."""
        raise NotImplementedError
        
    def _execute_next_planned_move(self) -> None:
        """Grab the next pre-computed move from :pyattr:`GameLogic.planned_moves`."""

        manager = self.manager
        if manager.awaiting_plan:
            return
        next_move = manager.game.get_next_planned_move()
        # ------------------------------------------------------------------
        # 1. Duplicate-first-move bug (historical)
        # ------------------------------------------------------------------
        # In the legacy procedural loop the *first* element of every freshly
        # downloaded plan was executed immediately **and** left inside the
        # queue, so it was executed *again* one tick later – polluting the
        # JSON output with duplicates.
        #
        # The refactor solves this by funnelling *all* moves – even the very
        # first of a new plan – through **this** method.  When
        # ``get_next_planned_move()`` returns a value we *remove* it from the
        # queue and execute it exactly once.  No special-casing, no risk of
        # the queue getting out of sync.
        # ------------------------------------------------------------------
        if not next_move:
            # Plan queue exhausted → round finished.
            print("🔄 No more planned moves in the current round — finishing round and requesting new plan.")

            # Persist buffered data & advance the authoritative round counter.
            manager.finish_round()

            # Ask for a new plan on the next tick.
            manager.need_new_plan = True
            return
            
        # At this point ``next_move`` is guaranteed to be a **single** string
        # representing a direction (UP/DOWN/LEFT/RIGHT).  It has been
        # *destructively* removed from ``planned_moves`` by
        # ``get_next_planned_move()`` so there is no possibility of re-use.
        #
        # Keeping an explicit copy in ``current_game_moves`` allows us to
        # compute post-hoc statistics (valid vs. invalid, apples per step …)
        # without touching the RoundManager's buffers.

        manager.current_game_moves.append(next_move)
        manager.game.draw()
        _, apple_eaten = self._execute_move(next_move)
        if apple_eaten:
            self._post_apple_logic()
    
    def _post_apple_logic(self) -> None:
        """Decide whether the current round continues after an apple was eaten."""
        # This logic remains sound. If the plan is empty after eating,
        # we need a new one.
        manager = self.manager
        if not manager.game.planned_moves:
            print(Fore.YELLOW + "No more planned moves after apple – round completed. Requesting new plan.")
            manager.finish_round()
            manager.need_new_plan = True

    def _handle_no_move(self) -> None:
        """Stub – subclasses may implement their own *no-move* sentinel handling."""
        return None

    def _handle_game_over(self) -> None:
        """Delegate heavy game-over processing to utils.game_manager_utils then reset for next game."""
        return None

    # Agent path – unchanged behaviour
    def _process_agent_game(self) -> None:
        """Path for **non-LLM agents** (heuristic, RL, human, …)."""

        manager = self.manager
        move = manager.agent.get_move(manager.game)  # type: ignore[arg-type]
        if not move:
            self._handle_no_move()
        else:
            manager.current_game_moves.append(move)
            manager.game.draw()
            _, apple_eaten = self._execute_move(move)
            if apple_eaten:
                print(Fore.CYAN + "🍎 Apple eaten!")
        if not manager.game_active:
            self._handle_game_over()
        manager.game.draw()

    # ---- Low-level helpers ---------------------

    def _execute_move(self, direction: str) -> Tuple[bool, bool]:
        """Run *one* snake move and perform all shared bookkeeping."""

        manager = self.manager

        prev_invalid_rev = manager.game.game_state.invalid_reversals
        game_active, apple_eaten = manager.game.make_move(direction)

        # Check if an invalid reversal occurred
        if manager.game.game_state.invalid_reversals > prev_invalid_rev:
            # -------------------
            # Elegant Limits Management for Invalid Reversals
            # -------------------
            from core.game_state_adapter import create_game_state_adapter

            game_state_adapter = create_game_state_adapter(
                manager, override_game_active=game_active
            )

            # Use elegant limits manager to handle INVALID_REVERSAL
            game_should_continue = manager.limits_manager.record_move(
                "INVALID_REVERSAL", game_state_adapter
            )

            if not game_should_continue:
                game_active = False

            # -------------------
            # Legacy Counter Updates (for backward compatibility)
            # -------------------
            manager.consecutive_invalid_reversals += 1
        else:
            manager.consecutive_invalid_reversals = 0

        # -------------------
        # Elegant Max Steps Management
        # -------------------
        if game_active:
            from core.game_state_adapter import create_game_state_adapter

            game_state_adapter = create_game_state_adapter(manager)
            game_active = manager.limits_manager.check_step_limit(
                manager.game.steps, game_state_adapter
            )

        manager.game.draw()
        pause = manager.get_pause_between_moves()
        if pause > 0:
            time.sleep(pause)

        # -------------------
        # Elegant Limits Management for Valid Moves
        # -------------------
        # For valid directional moves, use the limits manager to intelligently reset counters
        if direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            from core.game_state_adapter import create_game_state_adapter

            game_state_adapter = create_game_state_adapter(
                manager, override_game_active=True
            )
            manager.limits_manager.record_move(direction, game_state_adapter)

        # -------------------
        # Legacy Counter Updates (for backward compatibility)
        # -------------------
        manager.consecutive_something_is_wrong = 0
        if direction != "EMPTY":
            manager.consecutive_empty_steps = 0
            manager.consecutive_no_path_found = 0

        manager.game_active = game_active
        return game_active, apple_eaten

    def _handle_no_path_found(self) -> None:
        """Stub – subclasses may implement *NO_PATH_FOUND* sentinel handling."""
        return None  # override in Task-0

    def _apply_empty_move_delay(self) -> None:
        """
        Legacy method for applying empty move delay.

        This functionality is now handled elegantly by the ConsecutiveLimitsManager,
        but we keep this method for backward compatibility with any code that might
        still call it directly.
        """
        manager = self.manager
        pause_min: float = getattr(manager.args, "sleep_after_empty_step", 0.0)
        if pause_min <= 0 or getattr(manager, "last_no_path_found", False):
            return
        plural = "s" if pause_min != 1 else ""
        print(
            Fore.CYAN
            + f"⏸️ Legacy sleep: {pause_min} minute{plural} after EMPTY step..."
        )
        time.sleep(pause_min * 60)


class GameLoop(BaseGameLoop):
    """Task-0 (LLM) loop, which injects network-specific logic."""

    # ---------------------
    # **Override the main loop**
    # ---------------------
    def run(self) -> None:  # noqa: D401
        """Run the LLM-controlled game session.

        The base implementation distinguishes between *agent* vs. *planned-move*
        sessions by checking for the existence of ``manager.agent``.  Task-0,
        however, **always** provides an :class:`llm.agent_llm.AgentLLM` instance
        for compatibility, which would incorrectly funnel execution through the
        *agent* branch and ultimately try to *pull* a move from the LLM on every
        tick.

        In the refactored architecture we only query the LLM **once per round**
        to obtain a *plan* and then simply consume the pre-computed
        ``planned_moves`` queue.  Therefore we bypass the *agent* branch
        entirely and always delegate to :meth:`_process_active_game`.
        """
        import pygame  # local import to avoid hard dependency for head-less tests

        manager = self.manager  # local alias for brevity

        try:
            while manager.running and manager.game_count < manager.args.max_games:
                # Keep the GUI responsive.
                BaseGameManagerHelper.process_events(manager)

                if manager.game_active and manager.game is not None:
                    self._process_active_game()

                if manager.use_gui:
                    pygame.time.delay(manager.time_delay)
                    manager.clock.tick(manager.time_tick)

        except Exception as exc:  # pragma: no cover – final safety net
            print(Fore.RED + f"Fatal error: {exc}")
            traceback.print_exc()
        finally:
            pygame.quit()

    # ---------------------
    # Method Overrides
    # ---------------------
    def _get_new_plan(self) -> None:  # noqa: D401
        """Ask the LLM for a *new* plan and store it in ``game.planned_moves``.

        Design contract
        ----------------
        1.  **Round start = LLM request**  – We *must* bump the authoritative
            round counter *before* sending the prompt so that filenames like
            ``game_1_round_4_prompt.txt`` and the data injected into
            ``RoundManager`` stay in perfect lock-step.

        2.  **Plan lives in one place**  – `get_llm_response()` *only* populates
            ``planned_moves``; it no longer returns the first element.  All
            execution goes through :meth:`_execute_next_planned_move`, which
            means *every* move (including the first) flows through a single
            well-tested code path.
        """
        from llm.communication_utils import get_llm_response

        manager = self.manager
        
        # Increment the round counter *before* querying the LLM.
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("new round start")

        manager.awaiting_plan = True

        # Retrieve plan from LLM – synchronous call; may take a few seconds.
        get_llm_response(manager, round_id=manager.round_count)

        # Give humans a brief preview window to read the LLM output before the
        # snake starts moving.  Only relevant for interactive GUI sessions.
        manager.game.draw()
        if manager.use_gui:
            time.sleep(PAUSE_PREVIEW_BEFORE_MAKING_FIRST_MOVE_SECONDS)

        manager.need_new_plan = False # We have a plan now.

    def _handle_game_over(self) -> None:
        """Extend with Task-0 specific game-over processing."""
        manager = self.manager
        game = manager.game

        # -------------------
        # Prepare game state information to pass to the helper
        # -------------------
        game_state_info = {
            "args": manager.args,
            "log_dir": manager.log_dir,
            "game_count": manager.game_count,
            "total_score": manager.total_score,
            "total_steps": manager.total_steps,
            "game_scores": manager.game_scores,
            "round_count": manager.round_count,
            "round_counts": manager.round_counts,
            "time_stats": manager.time_stats,
            "token_stats": manager.token_stats,
            "valid_steps": manager.valid_steps,
            "invalid_reversals": manager.invalid_reversals,
            "empty_steps": manager.empty_steps,
            "something_is_wrong_steps": manager.something_is_wrong_steps,
            "no_path_found_steps": manager.no_path_found_steps,
        }

        # -------------------
        # Use the singleton helper to process game over logic
        # -------------------
        (
            manager.game_count,
            manager.total_score,
            manager.total_steps,
            manager.game_scores,
            manager.round_count,
            manager.time_stats,
            manager.token_stats,
            manager.valid_steps,
            manager.invalid_reversals,
            manager.empty_steps,
            manager.something_is_wrong_steps,
            manager.no_path_found_steps,
        ) = GameManagerHelper().process_game_over(game, game_state_info)

        # -------------------
        # Reset game state for the next game
        # -------------------
        # Reset per-game flags/counters for the upcoming game
        manager.need_new_plan = True
        manager.game_active = True
        manager.current_game_moves = []
        manager.round_count = 1
        manager.game.reset()

        # Reset first-plan flag for the new game
        manager._first_plan = True

        # -------------------
        # Elegant Limits Management Reset
        # -------------------
        # Reset all consecutive counters in the elegant limits manager
        if hasattr(manager, 'limits_manager'):
            manager.limits_manager.reset_all_counters()
        
        # -------------------
        # Legacy Counter Resets (for backward compatibility)
        # -------------------
        manager.consecutive_empty_steps = 0
        manager.consecutive_something_is_wrong = 0
        manager.consecutive_invalid_reversals = 0
        manager.consecutive_no_path_found = 0

    def _handle_no_move(self) -> None:  # noqa: D401
        """Handle EMPTY move sentinel with elegant limits management."""
        manager = self.manager

        from core.game_state_adapter import create_game_state_adapter

        game_state_adapter = create_game_state_adapter(manager)

        # Use elegant limits manager to handle EMPTY moves
        game_should_continue = manager.limits_manager.record_move(
            "EMPTY", game_state_adapter
        )

        if not game_should_continue:
            manager.game_active = False

        # -------------------
        # Legacy Counter Updates (for backward compatibility)
        # -------------------
        manager.consecutive_empty_steps += 1

        # If the game is still active, execute the EMPTY move
        if manager.game_active:
            self._execute_move("EMPTY")

    def _handle_no_path_found(self) -> None:  # noqa: D401
        """Handle NO_PATH_FOUND sentinel from LLM."""
        manager = self.manager

        from core.game_state_adapter import create_game_state_adapter

        game_state_adapter = create_game_state_adapter(manager)

        # Use elegant limits manager to handle NO_PATH_FOUND
        game_should_continue = manager.limits_manager.record_move(
            "NO_PATH_FOUND", game_state_adapter
        )

        if not game_should_continue:
            manager.game_active = False

        # -------------------
        # Legacy Counter Updates (for backward compatibility)
        # -------------------
        manager.consecutive_no_path_found += 1

        # If the game is still active, execute an EMPTY move
        if manager.game_active:
            self._execute_move("EMPTY")


def run_game_loop(manager: "BaseGameManager") -> None:
    """Procedural-style entry point.

    This factory function preserves the original API, which keeps all existing
    call-sites in `scripts/` working without modification.

    It detects whether the manager is a base or extended (Task-0 LLM) type and
    instantiates the correct loop implementation automatically.

    Args:
        manager: The active game manager instance.
    """
    # Choose the correct GameLoop implementation based on the manager's type.
    # We check for an LLM-specific attribute to decide.
    if hasattr(manager, "llm_client"):
        loop = GameLoop(manager)
    else:
        loop = BaseGameLoop(manager)

    loop.run()
