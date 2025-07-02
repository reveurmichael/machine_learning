"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.

Enhanced with limits manager integration to provide the same console output
that would have appeared during the original game session.

Lazy Pygame Import
---------------------
`pygame` is now loaded **only when needed** (``use_gui=True``).  Head-less
CI jobs or data-pipeline scripts can import and use the replay engine without
installing SDL/pygame.  All references go through ``self._pygame`` which is
`None` in head-less mode.
"""

from __future__ import annotations

import time
import traceback
import argparse
from typing import Any, Dict, List, Optional

import importlib
import numpy as np

from core.game_logic import GameLogic
from core.game_limits_manager import create_limits_manager
from config.ui_constants import TIME_DELAY, TIME_TICK
from replay.replay_utils import load_game_json, parse_game_data
from replay.replay_data import ReplayData
from config.game_constants import END_REASON_MAP, SENTINEL_MOVES
from core.game_file_manager import FileManager

# Initialize file manager for replay operations
_file_manager = FileManager()


class ReplayGameStateProvider:
    """
    Mock game state provider for replay mode limits manager integration.
    
    This class provides the necessary interface for the limits manager to
    function during replay, allowing it to show the same console output
    that would have appeared during the original game session.
    
    Design Pattern: Adapter Pattern
    - Adapts replay engine interface to limits manager expectations
    - Provides consistent game state management during replay
    """
    
    def __init__(self, replay_engine: 'BaseReplayEngine') -> None:
        self.replay_engine = replay_engine
    
    def record_game_end(self, reason: str) -> None:
        """Record game end reason and stop current game (not the entire replay)."""
        self.replay_engine.game_end_reason = reason
        self.replay_engine.game_active = False  # Only stop current game, not the replay loop
        print(f"🎮 Replay: Game ended with reason: {reason}")
    
    def is_game_active(self) -> bool:
        """Check if current game is still active."""
        return self.replay_engine.game_active

# ---------------------
# Generic, headless-capable replay skeleton (LLM-agnostic).
#
# It derives directly from *GameLogic* so that we can reuse the full snake
# implementation (board state, collision handling, etc.) without pulling in
# the newer *GameManager* / *GameController* abstractions that are not needed
# for offline replays.
#
# Enhanced with limits manager integration to provide authentic console output
# that matches what would have appeared during the original game session.
# ---------------------


class BaseReplayEngine(GameLogic):
    """Headless-capable replay engine skeleton (LLM-agnostic).

    Only contains functionality that is independent of Task-0 specifics
    (e.g. no log file parsing, no LLM meta-data).  Concrete subclasses are
    expected to implement the abstract *load_game_data*, *update*,
    *handle_events*, and *run* methods.
    
    Enhanced with limits manager integration to provide the same console output
    that would have appeared during the original game session, including:
    - Invalid reversal warnings and limits
    - Empty move warnings and limits  
    - LLM error warnings and limits
    - NO_PATH_FOUND warnings and limits
    
    Design Pattern: Template Method Pattern
    - Common replay structure with type-specific console output
    - Maintains consistency with live game sessions
    """

    # ---------------------
    # GUI plumbing – identical public API as before so that *scripts/replay.py*
    # continues to work unchanged.
    # ---------------------

    def set_gui(self, gui_instance):  # type: ignore[override]
        """Attach a GUI instance and propagate the current *paused* state."""

        super().set_gui(gui_instance)

        # Keep the GUI's internal flag in sync so it can adapt its display.
        if gui_instance and hasattr(gui_instance, "set_paused"):
            gui_instance.set_paused(self.paused)

    # ---------------------
    # Utility so the replay can *force-set* an apple position taken from the
    # JSON artefacts (random generation would desynchronise the playback).
    # ---------------------

    def set_apple_position(self, position):  # noqa: D401 – keep simple sig
        """Force-set the apple to a specific position (for replay determinism)."""
        self.apple_position = np.array(position)
        self._update_board()

    def _generate_apple(self) -> NDArray[np.int_]:  # type: ignore[override]
        """Override apple generation to use pre-recorded positions during replay.
        
        This ensures that the replay uses the exact same apple sequence as the
        original game, maintaining deterministic behavior.
        """
        # During replay, advance to the next pre-recorded apple position
        if hasattr(self, 'apple_positions') and hasattr(self, 'apple_index'):
            # Advance to next apple position
            self.apple_index += 1
            if self.apple_index < len(self.apple_positions):
                next_apple = self.apple_positions[self.apple_index]
                return np.array(next_apple)
        
        # Fallback to parent implementation if no pre-recorded positions available
        return super()._generate_apple()

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
        # Initialise the underlying game engine first so that attributes like
        # ``snake_positions`` are available before we start loading JSON data.
        super().__init__(use_gui=use_gui)

        # ---------------------
        # Lazy Pygame import – only when GUI is requested
        # ---------------------
        self._pygame = None  # type: ignore[assignment]
        self.use_gui = use_gui  # store for convenience (Base class already has it)

        if self.use_gui:
            try:
                self._pygame = importlib.import_module("pygame")
                # Local constants module (pygame.locals) is needed for event codes
                self._pygame_locals = importlib.import_module("pygame.locals")  # type: ignore[attr-defined]
                self.clock = self._pygame.time.Clock()
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "GUI mode requested in replay but pygame is not installed. "
                    "Install it or use --no-gui/headless mode."
                ) from exc
        else:
            self.clock = None

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
        self.game_active: bool = True  # Controls current game progress (separate from replay loop)

        # Game-over meta-data – may stay *None* for subclasses that do not use them.
        self.game_end_reason: Optional[str] = None

        # ---------------------
        # Enhanced Console Output: Limits Manager Integration
        # ---------------------
        # Initialize limits manager with default LLM game configuration
        # to provide the same console output that would have appeared
        # during the original game session
        self._init_limits_manager()

    def _init_limits_manager(self) -> None:
        """
        Initialize the limits manager for authentic console output during replay.
        
        This method creates a limits manager with default LLM game configuration
        to ensure that replay mode shows the same warning messages and game-over
        messages that would have appeared during the original game session.
        
        IMPORTANT: Sleep penalties are disabled for replay mode since we're showing
        past events, not actively waiting for LLM responses.
        
        Design Pattern: Factory Pattern
        - Creates properly configured limits manager for replay context
        - Maintains consistency with live game behavior (minus sleep delays)
        """
        # Create mock args with typical LLM game limits
        args = argparse.Namespace()
        args.max_consecutive_empty_moves_allowed = 5  # Typical default
        args.max_consecutive_something_is_wrong_allowed = 3
        args.max_consecutive_invalid_reversals_allowed = 10
        args.max_consecutive_no_path_found_allowed = 5
        args.max_steps = 1000
        
        # CRITICAL: Disable sleep penalties for replay mode
        # Replay should show what happened, not simulate real-time delays
        args.sleep_after_empty_step = 0.0  # No sleep during replay
        
        # Initialize limits manager
        self.limits_manager = create_limits_manager(args)
        
        # Create mock game state provider for limits manager
        self.game_state_provider = ReplayGameStateProvider(self)

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

    def load_next_game(self) -> None:
        """Advance *game_number* by one and attempt to load that game."""
        self.game_number += 1
        if not self.load_game_data(self.game_number):
            print("No more games to load. Replay complete.")
            self.running = False
        else:
            # Reset limits manager counters for the new game
            if hasattr(self, 'limits_manager'):
                self.limits_manager.reset_all_counters()

    def execute_replay_move(self, direction_key: str) -> bool:
        """Execute *direction_key* during a replay step.

        Enhanced with limits manager integration to provide the same console output
        that would have appeared during the original game session, including warning
        messages and game-over conditions for consecutive limit violations.
        
        Keeps logic identical to the original GameController implementation so
        the replay matches exactly what happened during recording.
        """
        # ---------------------
        # Handle *sentinel* pseudo-moves that encode timing or errors in
        # the original session but do *not* move the snake.
        # 
        # Enhanced: Use limits manager to show authentic console output
        # ---------------------
        if direction_key in SENTINEL_MOVES:
            # Record the move in game state (for statistics)
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
            
            # ---------------------
            # Enhanced Console Output: Use limits manager for authentic warnings/game-over messages
            # ---------------------
            # Process the sentinel move through limits manager to show the same
            # console output that would have appeared during the original game session
            game_should_continue = self.limits_manager.record_move(direction_key, self.game_state_provider)
            
            # If limits manager says game should end, respect that decision
            if not game_should_continue:
                return False
            
            return True  # Game continues, snake stays put

        # Regular move – delegate to *make_move* from BaseGameController
        game_active, apple_eaten = super().make_move(direction_key)
        
        # For valid moves, reset appropriate counters in limits manager
        if direction_key in ["UP", "DOWN", "LEFT", "RIGHT"]:
            self.limits_manager.record_move(direction_key, self.game_state_provider)

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
            "game_active": self.game_active,  # Include game_active flag for consistency
            "speed": (
                1.0 / self.pause_between_moves if self.pause_between_moves else 1.0
            ),
            "game_end_reason": getattr(self, "game_end_reason", None),
            "total_games": getattr(self, "total_games", None),
        }

    def reset(self) -> None:  # type: ignore[override]
        """Reset the game state without generating a random apple.
        
        During replay, we use pre-recorded apple positions instead of
        generating random ones.
        """
        # Reset game state (same as parent, but without apple generation)
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]

        # Reset game state tracker
        self.game_state.reset()

        # NOTE: We do NOT call _generate_apple() here like the parent does
        # The apple position will be set manually after loading replay data

        # Update the board (will be updated again after apple is set)
        self._update_board()

        # Draw if GUI is available
        if self.use_gui and self.gui:
            self.draw()

        # Clear runtime direction/collision trackers for the new game
        self.current_direction = None
        self.last_collision_type = None

        # Reset apple history (will be populated with first apple after setup)
        self.apple_positions_history = []

        # Sync initial snake body into GameData so snake_length starts correct
        self.game_state.snake_positions = self.snake_positions.tolist()
        
        # Clear planned moves
        self.planned_moves = []
        
        # Reset limits manager counters for the new game
        if hasattr(self, 'limits_manager'):
            self.limits_manager.reset_all_counters()


# ---------------------
# Concrete Task-0 implementation – identical behaviour to the *previous* code
# ---------------------


class ReplayEngine(BaseReplayEngine):
    """Task-0 replay engine that consumes *game_N.json* artefacts."""

    def __init__(
        self,
        log_dir: str,
        pause_between_moves: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            pause_between_moves=pause_between_moves,
            auto_advance=auto_advance,
            use_gui=use_gui,
        )

        # Task-0 meta-data ---------------------
        self.primary_llm: Optional[str] = None
        self.secondary_llm: Optional[str] = None
        self.game_timestamp: Optional[str] = None
        self.llm_response: Optional[str] = None
        self.total_games: int = _file_manager.get_total_games(log_dir)

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
        if self.paused or not self.game_active:
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
                    self.planned_moves = (
                        self.planned_moves[1:] if len(self.planned_moves) > 1 else []
                    )

                # Execute and post-process ---------------------
                game_continues = self.execute_replay_move(next_move)
                self.last_move_time = current_time

                if not game_continues:
                    print(
                        f"Game {self.game_number} over. Score: {self.score}, Steps: {self.steps}, End reason: {self.game_end_reason}"
                    )
                    self.game_active = False  # Stop current game logic
                    self.move_index = len(self.moves)  # Prevent further moves

                    if self.auto_advance:
                        if self._pygame:
                            self._pygame.time.delay(1000)
                        else:
                            time.sleep(1)
                        self.load_next_game()
                    else:
                        # Enhanced user feedback for manual control
                        print(f"Replay paused at game end. Controls: SPACE=pause/resume, R=restart, LEFT/RIGHT=navigate, ESC=quit")

                # Immediate redraw so the GUI stays responsive
                if self.use_gui and self.gui:
                    self.draw()

            except Exception as exc:
                print(f"Error during replay: {exc}")
                traceback.print_exc()

    # ---------------------
    # Data loading – parses the *game_N.json* structure produced by Task-0
    # ---------------------

    def load_game_data(
        self, game_number: int
    ) -> Optional[Dict[str, Any]]:  # noqa: D401 – simple loader
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
            self.llm_response = (
                parsed.llm_response or "No LLM response data available for this game."
            )

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
            self.snake_positions = np.array(
                [[self.grid_size // 2, self.grid_size // 2]]
            )
            self.head_position = self.snake_positions[-1]
            # Set initial apple position without advancing the index
            # The apple at index 0 is now on the board
            self.apple_position = np.array(self.apple_positions[0])
            # Initialize apple history with the first apple
            self.apple_positions_history = [self.apple_position.copy()]
            self._update_board()

            # GUI book-keeping ---------------------
            if self.use_gui and self.gui and hasattr(self.gui, "move_history"):
                self.gui.move_history = []

            # Ensure replay is running and not paused at the start of a new game
            self.running = True
            self.paused = False
            self.game_active = True  # Reset game active flag for new game

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
        if not self.use_gui or not self._pygame:
            return

        pygame = self._pygame  # local alias for brevity

        redraw_needed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
                    # Next game - check boundary first
                    if self.game_number >= self.total_games:
                        print("Already at the last game")
                    else:
                        self.game_number += 1
                        if not self.load_game_data(self.game_number):
                            print("No more games to load. Staying on current game.")
                            self.game_number -= 1
                    redraw_needed = True
                elif event.key in (pygame.K_LEFT, pygame.K_p):
                    # Previous game - check boundary first
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
        if self.use_gui and self._pygame:
            if not self._pygame.get_init():
                self._pygame.init()
            clock = self._pygame.time.Clock()
        else:
            clock = None

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

            if self.use_gui and self._pygame:
                self._pygame.time.delay(TIME_DELAY)
                clock.tick(TIME_TICK)
            else:
                # Headless mode – simple sleep to avoid busy loop
                time.sleep(TIME_DELAY / 1000.0)

        if self.use_gui and self._pygame:
            self._pygame.quit()

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

