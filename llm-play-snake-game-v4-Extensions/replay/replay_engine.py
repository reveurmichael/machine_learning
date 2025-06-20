"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pygame
from pygame.locals import *  # noqa: F403 – Pygame constants

from core.game_controller import BaseGameController, GameController
from config.ui_constants import TIME_DELAY, TIME_TICK
from replay.replay_utils import load_game_json, parse_game_data
from replay.replay_data import ReplayData
from utils.file_utils import get_total_games
from config.game_constants import END_REASON_MAP, SENTINEL_MOVES

# ---------------------
# Generic replay skeleton – future tasks can inherit from this base and plug
# in their own data-loading logic while re-using the event loop helpers.
# ---------------------


class BaseReplayEngine(BaseGameController):
    """Headless-capable replay engine skeleton (LLM-agnostic).

    Only contains functionality that is independent of Task-0 specifics
    (e.g. no log file parsing, no LLM meta-data).  Concrete subclasses are
    expected to implement the abstract *load_game_data*, *update*,
    *handle_events*, and *run* methods.
    """

    # ---------------------
    # Construction & basic state
    # ---------------------

    def __init__(
        self,
        log_dir: str,
        pause_between_moves: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:  # noqa: D401 – simple init
        super().__init__(use_gui=use_gui)

        self.log_dir: str = log_dir
        self.pause_between_moves: float = pause_between_moves
        self.auto_advance: bool = auto_advance

        # Replay-specific runtime state ---------------------
        self.game_number: int = 1
        self.apple_positions: List[List[int]] = []
        self.apple_index: int = 0
        self.moves: List[str] = []
        self.move_index: int = 0
        self.moves_made: List[str] = []
        self.planned_moves: List[str] = []
        self.game_stats: Dict[str, Any] = {}

        # Timing helper so that the very first move respects *pause_between_moves*.
        self.last_move_time: float = time.time()

        # Generic replay flags ---------------------
        self.running: bool = True
        self.paused: bool = False

        # Game-over meta-data – may stay *None* for subclasses that do not use them.
        self.game_end_reason: Optional[str] = None

    # ---------------------
    # Abstract hooks – must be provided by concrete subclasses
    # ---------------------

    def load_game_data(self, game_number: int):  # pragma: no cover – interface only
        raise NotImplementedError

    def update(self):  # pragma: no cover – interface only
        raise NotImplementedError

    def handle_events(self):  # pragma: no cover – interface only
        raise NotImplementedError

    def run(self):  # pragma: no cover – interface only
        raise NotImplementedError

    # ---------------------
    # Generic helpers that *can* be shared across replay implementations
    # ---------------------

    def set_gui(self, gui_instance):  # type: ignore[override]
        """Attach a GUI and keep its *paused* flag in sync."""
        super().set_gui(gui_instance)
        if gui_instance and hasattr(gui_instance, "set_paused"):
            gui_instance.set_paused(self.paused)

    def load_next_game(self) -> None:
        """Advance *game_number* by one and attempt to load that game."""
        self.game_number += 1
        if not self.load_game_data(self.game_number):
            print("No more games to load. Replay complete.")
            self.running = False

    def execute_replay_move(self, direction_key: str) -> bool:
        """Execute *direction_key* during a replay step.

        Keeps logic identical to the original GameController implementation so
        the replay matches exactly what happened during recording.
        """
        # ---------------------
        # Handle *sentinel* pseudo-moves that encode timing or errors in
        # the original session but do *not* move the snake.
        # ---------------------
        if direction_key in SENTINEL_MOVES:
            if direction_key == "INVALID_REVERSAL":
                self.game_state.record_invalid_reversal()
            elif direction_key == "EMPTY":
                # LLM-specific sentinel – only call helper when the subclass implements it
                if hasattr(self.game_state, "record_empty_move"):
                    self.game_state.record_empty_move()  # type: ignore[attr-defined]
            elif direction_key == "SOMETHING_IS_WRONG":
                # LLM-specific sentinel – guard for non-LLM tasks
                if hasattr(self.game_state, "record_something_is_wrong_move"):
                    self.game_state.record_something_is_wrong_move()  # type: ignore[attr-defined]
            elif direction_key == "NO_PATH_FOUND":
                self.game_state.record_no_path_found_move()
            return True  # Game continues, snake stays put

        # Regular move – delegate to *make_move* from BaseGameController
        game_active, apple_eaten = super().make_move(direction_key)

        # If an apple was eaten, advance *apple_index* so the replay keeps the
        # pre-recorded apple sequence instead of generating random ones.
        if apple_eaten and self.apple_index + 1 < len(self.apple_positions):
            self.apple_index += 1
            self.set_apple_position(self.apple_positions[self.apple_index])

        return game_active

    def _build_state_base(self) -> Dict[str, Any]:
        """Return the *generic* replay state shared by all tasks.

        LLM-specific metadata is intentionally *not* included here so that the
        base class remains agnostic.  Task-0 (or any other specialised task)
        can enrich the returned dict in its own override.
        """
        return {
            "snake_positions": self.snake_positions,
            "apple_position": self.apple_position,
            "game_number": self.game_number,
            "score": self.score,
            "steps": self.steps,
            "move_index": self.move_index,
            "total_moves": len(self.moves),
            "planned_moves": getattr(self, "planned_moves", []),
            "paused": self.paused,
            "speed": 1.0 / self.pause_between_moves if self.pause_between_moves else 1.0,
            "game_end_reason": getattr(self, "game_end_reason", None),
            "total_games": getattr(self, "total_games", None),
        }


# ---------------------
# Concrete Task-0 implementation – identical behaviour to the *previous* code
# ---------------------


class ReplayEngine(BaseReplayEngine, GameController):
    """Task-0 replay engine that consumes *game_N.json* artefacts."""

    def __init__(
        self,
        log_dir: str,
        pause_between_moves: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:
        super().__init__(log_dir=log_dir, pause_between_moves=pause_between_moves, auto_advance=auto_advance, use_gui=use_gui)

        # Task-0 meta-data ---------------------
        self.primary_llm: Optional[str] = None
        self.secondary_llm: Optional[str] = None
        self.game_timestamp: Optional[str] = None
        self.llm_response: Optional[str] = None
        self.total_games: int = get_total_games(log_dir)

    # ---------------------
    # Drawing
    # ---------------------

    def draw(self) -> None:  # noqa: D401 – simple wrapper
        if self.use_gui and self.gui:
            self.gui.draw(self._build_state_base())

    # ---------------------
    # Core update loop – verbatim from the historical Task-0 implementation
    # ---------------------

    def update(self) -> None:  # type: ignore[override]
        if self.paused:
            return

        current_time = time.time()
        if (
            current_time - self.last_move_time >= self.pause_between_moves
            and self.move_index < len(self.moves)
        ):
            try:
                next_move = self.moves[self.move_index]
                print(f"Move {self.move_index + 1}/{len(self.moves)}: {next_move}")

                # House-keeping ---------------------
                self.move_index += 1
                self.moves_made.append(next_move)

                if self.planned_moves:
                    self.planned_moves = self.planned_moves[1:] if len(self.planned_moves) > 1 else []

                # Execute and post-process ---------------------
                game_continues = self.execute_replay_move(next_move)
                self.last_move_time = current_time

                if not game_continues:
                    print(
                        f"Game {self.game_number} over. Score: {self.score}, Steps: {self.steps}, End reason: {self.game_end_reason}"
                    )
                    self.move_index = len(self.moves)

                    if self.auto_advance:
                        pygame.time.delay(1000)
                        self.load_next_game()

                # Immediate redraw so the GUI stays responsive
                if self.use_gui and self.gui:
                    self.draw()

            except Exception as exc:
                print(f"Error during replay: {exc}")
                traceback.print_exc()

    # ---------------------
    # Data loading – parses the *game_N.json* structure produced by Task-0
    # ---------------------

    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:  # noqa: D401 – simple loader
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None

        try:
            print(f"Loading game data from {game_file}")
            parsed: ReplayData | None = parse_game_data(game_data)
            if parsed is None:
                return None

            # Unpack parsed fields ---------------------
            self.apple_positions = parsed.apple_positions
            self.moves = parsed.moves
            self.planned_moves = parsed.planned_moves
            raw_reason = parsed.game_end_reason
            self.game_end_reason = END_REASON_MAP.get(raw_reason, raw_reason)
            self.primary_llm = parsed.primary_llm
            self.secondary_llm = parsed.secondary_llm
            self.game_timestamp = parsed.timestamp
            self.llm_response = parsed.llm_response or "No LLM response data available for this game."

            # Reset runtime counters ---------------------
            self.move_index = 0
            self.apple_index = 0
            self.moves_made = []
            self.game_stats = parsed.full_json

            loaded_score = game_data.get("score", 0)
            print(
                f"Game {game_number}: Score: {loaded_score}, Steps: {len(self.moves)}, "
                f"End reason: {self.game_end_reason}, LLM: {self.primary_llm}"
            )

            # Re-initialise board ---------------------
            print("Initializing game state…")
            self.reset()
            self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
            self.head_position = self.snake_positions[-1]
            self.set_apple_position(self.apple_positions[0])
            self._update_board()

            # GUI book-keeping ---------------------
            if self.use_gui and self.gui and hasattr(self.gui, "move_history"):
                self.gui.move_history = []

            print(f"Game {game_number} loaded successfully")
            return game_data

        except Exception as exc:
            print(f"Error loading game data: {exc}")
            traceback.print_exc()
            return None

    # ---------------------
    # Event handling & main loop – identical to the historical Task-0 code
    # ---------------------

    def handle_events(self) -> None:  # noqa: D401 – event loop
        redraw_needed = False
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if self.gui and hasattr(self.gui, "set_paused"):
                        self.gui.set_paused(self.paused)
                    print(f"Replay {'paused' if self.paused else 'resumed'}")
                    redraw_needed = True
                elif event.key in (pygame.K_UP, pygame.K_s):
                    self.pause_between_moves = max(0.1, self.pause_between_moves * 0.75)
                    print(f"Speed increased: {1 / self.pause_between_moves:.1f}x")
                    redraw_needed = True
                elif event.key in (pygame.K_DOWN, pygame.K_d):
                    self.pause_between_moves = min(3.0, self.pause_between_moves * 1.25)
                    print(f"Speed decreased: {1 / self.pause_between_moves:.1f}x")
                    redraw_needed = True
                elif event.key == pygame.K_r:
                    self.load_game_data(self.game_number)
                    print(f"Restarting game {self.game_number}")
                    redraw_needed = True
                elif event.key in (pygame.K_RIGHT, pygame.K_n):
                    self.game_number += 1
                    if not self.load_game_data(self.game_number):
                        print("No more games to load. Staying on current game.")
                        self.game_number -= 1
                    redraw_needed = True
                elif event.key in (pygame.K_LEFT, pygame.K_p):
                    if self.game_number > 1:
                        self.game_number -= 1
                        self.load_game_data(self.game_number)
                        print(f"Going to previous game {self.game_number}")
                    else:
                        print("Already at the first game")
                    redraw_needed = True

        if redraw_needed and self.use_gui and self.gui:
            self.draw()

    def run(self) -> None:  # noqa: D401 – main loop
        if not pygame.get_init():
            pygame.init()

        clock = pygame.time.Clock()

        # Try to load the first game; fallback to subsequent ones if necessary
        if not self.load_game_data(self.game_number):
            print(f"Could not load game {self.game_number}. Trying next game.")
            self.game_number += 1
            if not self.load_game_data(self.game_number):
                print("No valid games found in log directory.")
                return

        while self.running:
            self.handle_events()
            self.update()
            if self.use_gui and self.gui:
                self.draw()
            pygame.time.delay(TIME_DELAY)
            clock.tick(TIME_TICK)

        pygame.quit()

    # ---------------------
    # Enrich the generic state with LLM-specific metadata for Task-0
    # ---------------------

    def _build_state_base(self) -> Dict[str, Any]:  # type: ignore[override]
        base_state = super()._build_state_base()

        # Add the Task-0 extras – harmlessly ignored by future tasks
        base_state.update(
            {
                "llm_response": self.llm_response,
                "primary_llm": self.primary_llm,
                "secondary_llm": self.secondary_llm,
                "timestamp": self.game_timestamp,
            }
        )

        return base_state 