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

# --------------------------
# Type-checking helpers (avoid heavyweight imports at runtime)
# --------------------------

if TYPE_CHECKING:  # pragma: no cover – imports only needed for static analysis
    from core.game_manager import BaseGameManager  # use narrow, LLM-agnostic base class
    from llm.communication_utils import get_llm_response  # noqa: F401


# --------------------------
# OOP refactor – GameLoop class encapsulates the helpers
# --------------------------


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

    # ---- Public entry point ------------------------------------------------

    def run(self) -> None:
        """Execute the session until the requested number of games completes."""

        manager = self.manager  # local alias for brevity

        try:
            while manager.running and manager.game_count < manager.args.max_games:
                # Handle OS / pygame events first so the window remains responsive.
                process_events(manager)

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

    # -----------------------------------------------------------------------
    # Former top-level helpers – now instance methods (identical bodies)
    # -----------------------------------------------------------------------

    def _process_active_game(self) -> None:
        """One tick of live gameplay for *LLM / planned-move* sessions."""

        manager = self.manager  # local alias keeps lines short
        if manager.need_new_plan:
            self._request_and_execute_first_move()
        else:
            self._execute_next_planned_move()

        if not manager.game_active:
            self._handle_game_over()

        manager.game.draw()

    def _request_and_execute_first_move(self) -> None:
        """Subclasses (LLM, heuristic, RL…) must implement strategy-specific first-move logic."""
        raise NotImplementedError

    def _execute_next_planned_move(self) -> None:
        """Grab the next pre-computed move from :pyattr:`GameLogic.planned_moves`."""

        manager = self.manager
        if manager.awaiting_plan:
            return
        next_move = manager.game.get_next_planned_move()
        if not next_move:
            manager.finish_round()
            manager.need_new_plan = True
            print("🔄 No more planned moves in the current round, requesting new plan.")
            return
        manager.current_game_moves.append(next_move)
        manager.game.draw()
        _, apple_eaten = self._execute_move(next_move)
        if apple_eaten:
            self._post_apple_logic()

    def _post_apple_logic(self) -> None:
        """Decide whether the current round continues after an apple was eaten."""

        manager = self.manager
        if manager.game.planned_moves:
            print(Fore.CYAN + f"Continuing with {len(manager.game.planned_moves)} remaining planned moves for this round.")
        else:
            manager.finish_round()
            print(Fore.YELLOW + "No more planned moves, requesting new plan.")
            manager.need_new_plan = True

    def _handle_no_move(self) -> None:
        """Stub – subclasses may implement their own *no-move* sentinel handling."""
        return None

    def _handle_game_over(self) -> None:
        """Delegate post-mortem aggregation to :pymod:`utils.game_manager_utils`."""

        manager = self.manager
        process_game_over(manager.game, manager)
        manager.game_active = False

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

    # ---- Low-level helpers --------------------------------------------------

    def _execute_move(self, direction: str) -> Tuple[bool, bool]:
        """Run *one* snake move and perform all shared bookkeeping."""

        manager = self.manager

        prev_invalid_rev = manager.game.game_state.invalid_reversals
        game_active, apple_eaten = manager.game.make_move(direction)
        if manager.game.game_state.invalid_reversals > prev_invalid_rev:
            manager.consecutive_invalid_reversals += 1
            print(
                Fore.YELLOW +
                f"⚠️ Invalid reversal detected. Consecutive invalid reversals: "
                f"{manager.consecutive_invalid_reversals}/{manager.args.max_consecutive_invalid_reversals_allowed}"
            )
        else:
            manager.consecutive_invalid_reversals = 0

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

        if game_active and check_max_steps(manager.game, manager.args.max_steps):
            game_active = False
            manager.game.game_state.record_game_end("MAX_STEPS_REACHED")

        manager.game.draw()
        pause = manager.get_pause_between_moves()
        if pause > 0:
            time.sleep(pause)

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
        manager = self.manager
        pause_min: float = getattr(manager.args, "sleep_after_empty_step", 0.0)
        if pause_min <= 0 or getattr(manager, "last_no_path_found", False):
            return
        plural = "s" if pause_min != 1 else ""
        print(Fore.CYAN + f"⏸️ Sleeping {pause_min} minute{plural} after EMPTY step …")
        time.sleep(pause_min * 60)


# --------------------------
# Back-compat thin wrapper – keeps existing call-sites unchanged
# --------------------------


def run_game_loop(game_manager: "BaseGameManager") -> None:  # pragma: no cover
    """Historical function wrapper – instantiates :class:`GameLoop` and runs it."""

    GameLoop(game_manager).run()


# --------------------------
# Thin Task-0 subclass – inherits full behaviour
# --------------------------


class GameLoop(BaseGameLoop):
    """Task-0 implementation that *keeps* the LLM plumbing.

    The base class is fully LLM-agnostic.  This subclass restores the
    provider-specific helpers so behaviour remains 100 % identical for
    Task-0 while paving the way for heuristic / RL variants that can
    reuse :class:`BaseGameLoop` without the network dependency.
    """

    # ------------------------------------------------------------------
    # LLM-specific overrides – these were abstract in the base so that
    # heuristic / RL loops don't inherit any provider coupling.
    # ------------------------------------------------------------------

    def _request_and_execute_first_move(self) -> None:  # noqa: D401
        """Ask the LLM for a fresh plan and play its first move.

        Implements the behaviour previously baked into the procedural
        `_request_and_execute_first_move` helper.  Nothing changed except
        that the code now lives only in the Task-0 subclass.
        """

        manager = self.manager
        manager.awaiting_plan = True
        from llm.communication_utils import get_llm_response  # local import

        next_move, manager.game_active = get_llm_response(manager, round_id=manager.round_count)  # type: ignore[arg-type]
        manager.awaiting_plan = False
        manager.need_new_plan = False

        # Handle NO_PATH_FOUND sentinel recorded from the previous tick.
        if manager.last_no_path_found:
            self._handle_no_path_found()
            if not manager.game_active:
                return

        # Early-out if the LLM call ended the game (e.g. max steps).
        if not manager.game_active:
            return

        # Empty response → treat as EMPTY sentinel unless the communication
        # utils already appended a sentinel (retry case).
        if not next_move:
            if getattr(manager, "skip_empty_this_tick", False):
                manager.skip_empty_this_tick = False
                return
            self._handle_no_move()
            return

        # Optional 3-second preview so humans can read the plan.
        manager.game.draw()
        if manager.use_gui:
            import time as _t
            _t.sleep(3)

        # Drop the first element from planned_moves because we execute it now.
        if manager.game.planned_moves:
            manager.game.planned_moves.pop(0)

        _, apple_eaten = self._execute_move(next_move)
        if apple_eaten:
            self._post_apple_logic()

    # ------------------------------------------------------------------
    # Sentinel-specific handlers
    # ------------------------------------------------------------------

    def _handle_no_move(self) -> None:  # noqa: D401
        """LLM returned *no* usable move ⇒ record EMPTY sentinel."""

        manager = self.manager
        from colorama import Fore

        print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")

        # Record EMPTY sentinel in game state & analytics.
        manager.game.game_state.record_empty_move()
        manager.empty_steps = manager.game.game_state.empty_steps
        manager.current_game_moves.append("EMPTY")

        manager.consecutive_empty_steps += 1
        manager.consecutive_something_is_wrong = 0  # break SOMETHING_IS_WRONG streak

        print(
            Fore.YELLOW +
            f"⚠️ No valid moves found. Empty moves: {manager.consecutive_empty_steps}/{manager.args.max_consecutive_empty_moves_allowed}"
        )

        # EMPTY also breaks any chain of invalid reversals – reset counter.
        manager.consecutive_invalid_reversals = 0

        # Game-over condition: too many consecutive EMPTY moves.
        if (
            manager.consecutive_empty_steps >=
            manager.args.max_consecutive_empty_moves_allowed
        ):
            print(
                Fore.RED +
                "❌ Maximum consecutive empty moves reached "
                f"({manager.args.max_consecutive_empty_moves_allowed}). Game over."
            )
            manager.game_active = False
            manager.game.last_collision_type = 'MAX_CONSECUTIVE_EMPTY_MOVES_REACHED'
            manager.game.game_state.record_game_end("MAX_CONSECUTIVE_EMPTY_MOVES_REACHED")

        # Optional back-off sleep as configured via CLI.
        self._apply_empty_move_delay()

    def _handle_no_path_found(self) -> None:  # noqa: D401
        """Log the NO_PATH_FOUND sentinel and reset related counters."""

        manager = self.manager
        from colorama import Fore

        print(Fore.YELLOW + "⚠️ LLM reported NO_PATH_FOUND. Snake stays in place.")
        manager.game.game_state.record_no_path_found_move()
        manager.current_game_moves.append("NO_PATH_FOUND")
        manager.consecutive_something_is_wrong = 0
        manager.consecutive_invalid_reversals = 0
        manager.last_no_path_found = False

        # NO_PATH_FOUND does not affect EMPTY or reversal counters beyond the
        # resets above – no further action. 