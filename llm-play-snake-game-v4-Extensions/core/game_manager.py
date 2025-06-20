"""High-level manager for multi-game sessions of the LLM-controlled Snake."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pygame
from colorama import Fore
from collections import defaultdict

# Core game components – generic vs Task-0
from core.game_logic import BaseGameLogic, GameLogic
from core.game_loop import run_game_loop
from gui.game_gui import GameGUI
from llm.client import LLMClient
from config.ui_constants import TIME_DELAY, TIME_TICK

# Utils imports - organized by functionality
from utils.game_stats_utils import save_session_stats
from utils.continuation_utils import (
    continue_from_directory,
    handle_continuation_game_state,
    setup_continuation_session,
)
from utils.game_manager_utils import (
    initialize_game_manager,
    process_events,
    report_final_statistics,
)

# noqa: F401 to silence unused-import warnings – the runtime availability of
# SnakeAgent is required for eval()-based type hint resolution in some
# introspection utilities.
from core.game_agents import SnakeAgent  # noqa: F401

if TYPE_CHECKING:  # avoid heavy imports for runtime
    import argparse


# ------------------
# Generic session state (no LLM specifics) – shared by future managers
# ------------------


class BaseGameManager:
    """Lightweight session scaffold that future tasks can extend.

    The base class intentionally contains only the *generic* session state
    (CLI arguments and the primary running flag) so that specialised
    derivatives – Task-0ʼs LLM manager, RL managers, heuristic runners, … –
    can mix in their own orchestration without touching the common core.
    """

    def __init__(self, args: "argparse.Namespace") -> None:  # noqa: D401 – simple base
        self.args = args

        # ---------------- General session counters ----------------
        self.game_count: int = 0
        self.round_count: int = 1  # human-friendly 1-based index

        self.total_score: int = 0
        self.total_steps: int = 0

        # Sentinel / error counters aggregated across games
        self.valid_steps: int = 0
        self.invalid_reversals: int = 0
        self.consecutive_invalid_reversals: int = 0
        self.consecutive_no_path_found: int = 0

        # NO_PATH_FOUND tracking (kept generic as other tasks may reuse)
        self.last_no_path_found: bool = False
        self.no_path_found_steps: int = 0

        # Per-game score list and round stats
        self.game_scores: List[int] = []
        self.round_counts: List[int] = []
        self.total_rounds: int = 0

        self.need_new_plan = True

        # ---- Per-session state flags --------------------------
        self.game: Optional["BaseGameLogic"] = None  # set by subclasses
        self.game_active: bool = True

        # Move history for current game (purely cosmetic)
        self.current_game_moves: List[str] = []

        # GUI mode flag (set by CLI argument)
        self.use_gui: bool = not getattr(args, "no_gui", False)

        # Main loop control flag
        self.running: bool = True

    # ---- Hooks meant to be overridden --------------------------

    def initialize(self) -> None:  # pragma: no cover – interface stub
        """Prepare the session (LLM clients, log dirs, etc.)."""

    def run(self) -> None:  # pragma: no cover – interface stub
        """Start the main event loop."""

    # ------------------
    # Factory helpers – subclasses override the *CLS* attributes to inject
    # specialised components without rewriting the full methods.
    # ------------------

    GAME_LOGIC_CLS = BaseGameLogic  # override in Task-0 and others

    def setup_game(self):
        """Initialise game logic and optional GUI."""

        # Instantiate the chosen game-logic class (Base by default).
        self.game = self.GAME_LOGIC_CLS(use_gui=self.use_gui)  # type: ignore[call-arg]

        # Attach a GUI if the session runs with visual mode enabled.
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)

    def get_pause_between_moves(self) -> float:
        """Get the pause time between moves.
        
        Returns:
            Float representing pause time in seconds, 0 if no GUI is enabled
        """

        # Skip pause in standard no-gui batch mode
        if not self.use_gui:
            return 0.0

        # GUI mode (pygame or web gui) – use configured pause
        return self.args.move_pause

    # ------------------------------------------------------------------
    # Generic *round* helpers – shared across all tasks (0-5)
    # ------------------------------------------------------------------

    def increment_round(self, reason: str = "") -> None:  # noqa: D401 – generic helper
        """Flush current round buffer and start the next one.

        The base implementation is completely LLM-agnostic; subclasses may
        override to add extra logging but *should* call ``super()`` so that
        :pyclass:`core.game_rounds.RoundManager` remains the single source of
        truth for the round counter.
        """

        # Some Task-0 states (e.g. awaiting_plan) do not exist in every task.
        if hasattr(self, "awaiting_plan") and getattr(self, "awaiting_plan"):
            return  # avoid duplicate increments while still waiting for plan

        if not self.game or not hasattr(self.game, "game_state"):
            return

        gs = self.game.game_state  # type: ignore[attr-defined]

        # Seed the upcoming round with the *current* apple so replays never
        # contain null positions.  Works for all agent types.
        apple_pos = getattr(self.game, "apple_position", None)
        gs.round_manager.start_new_round(apple_pos)

        # Keep the public counter in sync with RoundManager
        self.round_count = gs.round_manager.round_count

        # Persist buffer so dashboards / JS overlays see live data
        gs.round_manager.sync_round_data()

        # Console feedback is useful for long-running experiments; harmless
        # for headless tasks.
        if reason:
            print(Fore.BLUE + f"📊 Advanced to round {self.round_count} ({reason})")
        else:
            print(Fore.BLUE + f"📊 Advanced to round {self.round_count}")

    def finish_round(self, reason: str = "round completed") -> None:  # noqa: D401 – thin wrapper
        """Public shortcut used by the game loop when a round ends."""

        self.increment_round(reason)


class GameManager(BaseGameManager):
    """Run one or many LLM-driven Snake games and collect aggregate stats."""
    
    # Plug in Task-0 specific game logic.
    GAME_LOGIC_CLS = GameLogic
    
    def __init__(self, args: "argparse.Namespace", agent: "SnakeAgent | None" = None) -> None:
        """Initialize the game manager.
        
        Args:
            args: Command line arguments
            agent: Optional pluggable policy for the game loop
        """
        super().__init__(args)
        
        self.empty_steps = 0
        self.something_is_wrong_steps = 0
        self.consecutive_empty_steps = 0
        self.consecutive_something_is_wrong = 0
        
        # Time and token statistics (auto-zeroing via defaultdict)
        self.time_stats = _make_time_stats()
        self.token_stats = _make_token_stats()
        
        self.awaiting_plan = (
            False  # Whether we're currently waiting for a new plan from the LLM
        )
        
        # Pygame and timing
        self.clock = pygame.time.Clock()
        self.time_delay = TIME_DELAY
        self.time_tick = TIME_TICK
        
        # LLM clients
        self.llm_client = None
        self.parser_provider = None
        self.parser_model = None
        
        # Logging directories
        self.log_dir = None
        self.prompts_dir = None
        self.responses_dir = None
        

        # Guard to avoid double-recording EMPTY after an exception already
        # appended a sentinel in communication_utils.
        self.skip_empty_this_tick = False

        # Optional pluggable policy – when provided, the game loop will use
        # this agent *instead of* the built-in LLM planning pipeline.
        # Import inside __init__ to avoid a hard dependency at module import
        # time (keeps startup fast and sidesteps circular-import edge cases).
        self.agent = agent
        
    def create_llm_client(self, provider: str, model: str | None = None) -> LLMClient:
        """Create an LLM client with the specified provider and model.
        
        Args:
            provider: LLM provider name
            model: Model name (optional)
            
        Returns:
            LLMClient instance
        """
        return LLMClient(provider=provider, model=model)
    

    def initialize(self) -> None:
        """Initialize the game, LLM clients, and logging directories."""
        initialize_game_manager(self)
    
    def process_events(self) -> None:
        """Process pygame events."""
        process_events(self)
    
    def run_game_loop(self) -> None:
        """Run the main game loop."""
        run_game_loop(self)
    
    def report_final_statistics(self) -> None:
        """Report final statistics at the end of the game session."""
        # Only report if games were played
        if self.game_count == 0:
            return
            
        # Update summary.json metadata at session end (no JSON-parser stats anymore)
        save_session_stats(self.log_dir)
        
        # --------------------------
        # Use the counters that have been **aggregated across games**
        # throughout the session (updated in process_game_over).
        # Previously this method overwrote them with the last game's values,
        # which zero-ed the numbers in summary.json and the console banner.
        # --------------------------
        valid_steps = self.valid_steps
        invalid_reversals = self.invalid_reversals
        
        # Create stats dictionary
        stats_info = {
            "log_dir": self.log_dir,
            "game_count": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "empty_steps": self.empty_steps,
            "something_is_wrong_steps": self.something_is_wrong_steps,
            "valid_steps": valid_steps,
            "invalid_reversals": invalid_reversals,
            "game": self.game,
            "time_stats": self.time_stats,
            "token_stats": self.token_stats,
            "round_counts": self.round_counts,
            "total_rounds": self.total_rounds,
            "max_games": self.args.max_games,
            "no_path_found_steps": self.no_path_found_steps,
        }
        
        # Report statistics to console and save to files
        report_final_statistics(stats_info)
    
        # Mark manager as no longer running so front-end can display
        # the "Session Finished" banner.  (Used by main_web / JS.)
        self.running = False

    def continue_from_session(self, log_dir: str, start_game_number: int) -> None:
        """Continue from a previous game session.
        
        Args:
            log_dir: Directory containing the previous session logs
            start_game_number: Game number to start from
        """
        print(Fore.GREEN + f"🔄 Continuing experiment from directory: {log_dir}")
        print(Fore.GREEN + f"🔄 Starting from game number: {start_game_number}")
        
        # Set up continuation session (prepares log dirs & stats)
        setup_continuation_session(self, log_dir, start_game_number)
        
        # Create LLM clients using original configuration (needed before game loop).
        # The helper performs a health-check, so we **sleep only after** it
        # succeeds to avoid wasting time when credentials are wrong.
        from utils.initialization_utils import setup_llm_clients, enforce_launch_sleep

        setup_llm_clients(self)  # includes health check

        # Start-delay – shared helper keeps behaviour in sync
        enforce_launch_sleep(self.args)
        
        # Handle game state for continuation (sets up board, counters, etc.)
        handle_continuation_game_state(self)
        
        # Run the game loop
        self.run()
        
    @classmethod
    def continue_from_directory(cls, args: "argparse.Namespace") -> "GameManager":
        """Factory method to create a GameManager instance for continuation.
        
        Args:
            args: Command line arguments with continue_with_game_in_dir set
            
        Returns:
            GameManager instance configured for continuation
        """
        return continue_from_directory(cls, args)
    
    def run(self) -> None:
        """Initialize and run the game session."""
        try:
            # Skip initialization if this is a continuation
            if (
                not hasattr(self.args, "is_continuation")
                or not self.args.is_continuation
            ):
                # Initialize the game and LLM clients
                self.initialize()
            
            # Run games until we reach max_games
            while self.game_count < self.args.max_games and self.running:
                # Run the game loop
                self.run_game_loop()
                
                # Check if we've reached the max games
                if self.game_count >= self.args.max_games:
                    # We only need the break; final banner will be printed in report_final_statistics()
                    break
            
        finally:
            # Final cleanup
            if self.use_gui and pygame.get_init():
                pygame.quit()
            
            # Report final statistics
            self.report_final_statistics() 

# --------------------------
# Utility factories for auto-initialised stats dictionaries
# --------------------------


def _make_time_stats() -> defaultdict[str, int]:
    """Return a defaultdict that auto-zeros missing time fields."""
    return defaultdict(int)


def _make_token_stats() -> dict[str, defaultdict[str, int]]:
    """Return nested defaultdicts for primary/secondary token counters."""
    return {
        "primary": defaultdict(int),
        "secondary": defaultdict(int),
    }
