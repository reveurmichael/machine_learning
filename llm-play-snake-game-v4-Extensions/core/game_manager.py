"""Session management for Snake game tasks (0-5).

This module implements a clean, future-proof architecture where:
- BaseGameManager provides all generic functionality for Tasks 1-5
- GameManager (Task-0) adds only LLM-specific features

Design Philosophy:
- Tasks 0-5 inherit BaseGameManager directly
- Each task gets exactly what it needs, nothing more
- Clean separation of concerns, no historical baggage

Testability & head-less execution
---------------------
Now *pygame* is **lazy-loaded**.  We only import/initialise
it inside :pyclass:`BaseGameManager` **when** the caller explicitly requests a
GUI session (``use_gui=True``, default for Task-0).  This brings two major
benefits:

1. **Head-less CI pipelines** – unit-tests can exercise the full planning &
   game-logic stack on platforms where SDL/pygame is unavailable.
2. **Lower coupling / faster import time** – every non-visual extension
   (heuristics, RL, dataset generation, …) remains free of the heavyweight
   dependency.

The pattern uses ``importlib.import_module("pygame")`` and stores the module
on :pyattr:`BaseGameManager._pygame`.  All downstream code gates GUI calls via
``self.use_gui`` **and** ``self._pygame is not None``.
"""

from __future__ import annotations

import os
import importlib
from typing import TYPE_CHECKING, List, Optional
from collections import defaultdict

from colorama import Fore

# Core components - all future-ready
from core.game_logic import BaseGameLogic, GameLogic
from core.game_loop import run_game_loop
from config.ui_constants import TIME_DELAY, TIME_TICK

# LLM components - only for Task-0
from llm.client import LLMClient

# Utilities - organized by purpose
from core.game_stats_manager import GameStatsManager
# Continuation utilities imported locally to avoid circular dependency
from core.game_manager_helper import GameManagerHelper

# Agent protocol for all tasks
from core.game_agents import BaseAgent

if TYPE_CHECKING:
    import argparse


# -------------------
# BASE CLASS FOR ALL TASKS (0-5) - Pure Generic Implementation
# -------------------


class BaseGameManager:
    """Generic session manager for all Snake game tasks.
    
    This class contains ONLY attributes and methods that are useful
    across Tasks 1-5. No LLM-specific code, no legacy patterns.
    
    Perfect for:
    - Task-1 (Heuristics): BFS, A*, Hamiltonian cycles
    - Task-2 (Supervised): Neural network training on game data  
    - Task-3 (Reinforcement): DQN, PPO, actor-critic agents
    - Task-4 (LLM Fine-tuning): Custom fine-tuned models
    - Task-5 (Distillation): Model compression techniques
    """

    # Factory hook - subclasses specify their game logic type
    GAME_LOGIC_CLS = BaseGameLogic

    def __init__(self, args: "argparse.Namespace") -> None:
        """Initialize generic session state for any task type."""
        self.args = args

        # -------------------
        # Core session metrics (used by ALL tasks)
        # -------------------
        self.game_count: int = 0
        self.round_count: int = 1
        self.total_score: int = 0
        self.total_steps: int = 0
        self.total_rounds: int = 0

        # Per-game data tracking
        self.game_scores: List[int] = []
        self.round_counts: List[int] = []
        self.current_game_moves: List[str] = []

        # Error tracking (generic across all algorithms)
        self.valid_steps: int = 0
        self.invalid_reversals: int = 0
        self.consecutive_invalid_reversals: int = 0
        self.consecutive_no_path_found: int = 0
        self.no_path_found_steps: int = 0
        self.last_no_path_found: bool = False

        # -------------------
        # Game state management (used by ALL tasks)
        # -------------------
        self.game: Optional[BaseGameLogic] = None
        self.game_active: bool = True
        self.need_new_plan: bool = True
        self.running: bool = True
        self._first_plan: bool = True  # Track first planning cycle for round management

        # -------------------
        # Visualization & timing (used by ALL tasks)
        # -------------------
        self.use_gui: bool = not getattr(args, "no_gui", False)
        self.pause_between_moves: float = getattr(args, "pause_between_moves", 0.0)
        self.auto_advance: bool = getattr(args, "auto_advance", False)

        # Lazy-load pygame ONLY when GUI is requested.
        # This keeps head-less extensions (heuristics, RL, …) completely
        # free of the heavyweight SDL dependency and avoids opening any
        # graphical window when ``use_gui`` is False.

        self._pygame = None  # type: ignore[assignment]

        if self.use_gui:
            try:
                # Import inside the branch so that *headless* runs never even
                # attempt to import pygame.
                self._pygame = importlib.import_module("pygame")
                self.clock = self._pygame.time.Clock()  # type: ignore[attr-defined]
                self.time_delay = TIME_DELAY
                self.time_tick = TIME_TICK
            except ModuleNotFoundError as exc:  # pragma: no cover – dev machines without pygame
                raise RuntimeError(
                    "GUI mode requested but pygame is not installed. "
                    "Install it or re-run with --no-gui."
                ) from exc
        else:
            # Headless – initialise dummies so the rest of the code can rely on them.
            self.clock = None
            self.time_delay = 0
            self.time_tick = 0

        # -------------------
        # Logging infrastructure (used by ALL tasks)
        # -------------------
        self.log_dir: Optional[str] = None

    # -------------------
    # CORE LIFECYCLE METHODS - All tasks implement these
    # -------------------

    def initialize(self) -> None:
        """Initialize the task-specific components.
        
        Override in subclasses to set up:
        - Logging directories
        - Models/algorithms  
        - Dataset connections
        - Agent configurations
        """
        raise NotImplementedError("Subclasses must implement initialize()")

    def run(self) -> None:
        """Execute the main task workflow.
        
        Override in subclasses to implement:
        - Training loops (RL, Supervised)
        - Evaluation protocols (Heuristics)  
        - Fine-tuning pipelines (LLM tasks)
        """
        raise NotImplementedError("Subclasses must implement run()")

    # -------------------
    # GENERIC GAME SETUP - Reusable across all tasks
    # -------------------

    def setup_game(self) -> None:
        """Create game logic and optional GUI interface.
        
        Extensions can pass a `--grid_size` CLI argument to have the game
        instantiate with the requested grid size without overriding this method.
        """
        # Use the specified game logic class (BaseGameLogic by default)
        grid_size = getattr(self.args, "grid_size", None)
        if grid_size is not None:
            self.game = self.GAME_LOGIC_CLS(grid_size=grid_size, use_gui=self.use_gui)
        else:
            self.game = self.GAME_LOGIC_CLS(use_gui=self.use_gui)

        # Attach GUI if visual mode is requested
        if self.use_gui:
            # Lazy import keeps headless extensions free of pygame.
            from gui.game_gui import GameGUI  # noqa: WPS433 – intentional local import
            gui = GameGUI()
            # Ensure GUI pixel scaling matches the *actual* game grid size
            if hasattr(self.game, "grid_size"):
                gui.resize(self.game.grid_size)  # auto-adjust cell size & grid lines
            self.game.set_gui(gui)

    def get_pause_between_moves(self) -> float:
        """Get pause duration between moves.
        
        Returns:
            Pause time in seconds (0.0 for no-GUI mode)
        """
        return self.pause_between_moves if self.use_gui else 0.0


    # -------------------
    # ROUND MANAGEMENT - Generic for all planning-based tasks
    # -------------------

    def start_new_round(self, reason: str = "") -> None:
        """Begin a new planning round.
        
        All tasks use rounds to track planning cycles:
        - Heuristics: Each path-finding attempt
        - RL: Each action selection
        - LLM: Each prompt/response cycle
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        game_state = self.game.game_state
        apple_pos = getattr(self.game, "apple_position", None)
        game_state.round_manager.start_new_round(apple_pos)

        # Sync public counter
        self.round_count = game_state.round_manager.round_count
        game_state.round_manager.sync_round_data()

        # Console feedback for long experiments
        if reason:
            print(Fore.BLUE + f"📊 Round {self.round_count} started ({reason})")
        else:
            print(Fore.BLUE + f"📊 Round {self.round_count} started")

    def flush_buffer_with_game_state(self) -> None:
        """Flush the current round buffer and sync data safely.
        
        Extensions can call this to ensure any buffered round data is persisted
        before writing files or computing summaries.
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return
        game_state = self.game.game_state
        if hasattr(game_state, "round_manager"):
            game_state.round_manager.flush_buffer()
            game_state.round_manager.sync_round_data()

    def increment_round(self, reason: str = "") -> None:
        """Increment the round counter and flush the current round data.
        
        This method is called to finish the current round and start a new one.
        Generic across all tasks that use planning rounds.
        
        Args:
            reason: Optional description of why the round is incrementing
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        # Prevent double-increment during certain conditions
        if hasattr(self, "awaiting_plan") and self.awaiting_plan:
            return

        game_state = self.game.game_state
        
        # Flush current round data before incrementing
        game_state.round_manager.flush_buffer()
        
        # Update session-level round tracking
        self.round_counts.append(self.round_count)
        
        # Start new round with current apple position
        apple_pos = getattr(self.game, "apple_position", None)
        game_state.round_manager.start_new_round(apple_pos)
        
        # Sync public counter
        self.round_count = game_state.round_manager.round_count
        game_state.round_manager.sync_round_data()

        # Console feedback for long experiments  
        if reason:
            print(Fore.BLUE + f"📊 Round {self.round_count} started ({reason})")
        else:
            print(Fore.BLUE + f"📊 Round {self.round_count} incremented")

    def finish_round(self, reason: str = "") -> None:  # noqa: D401
        """Finalize the **current** round without starting a new one.

        This helper is called by :pyclass:`core.game_loop.GameLoop` when the
        pre-computed *plan* has been fully executed (i.e. our move queue is
        empty).  Its sole responsibility is to **persist** whatever is left in
        the volatile :class:`core.game_rounds.RoundManager` buffer so that JSON
        outputs and in-memory statistics remain consistent.

        Crucially, the method deliberately **does not** bump
        :pyattr:`round_count` – that is handled at the *beginning* of the next
        planning cycle via :meth:`increment_round`.  This design keeps filenames
        like ``game_2_round_5_prompt.txt`` in perfect sync with the data stored
        in ``rounds_data`` while avoiding the off-by-one issues observed in the
        legacy procedural loop.

        Args:
            reason: Optional text explaining why the round is being closed.
                Primarily useful for verbose logging during long experiments.
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        # Flush any pending data for the *current* round.
        game_state = self.game.game_state
        game_state.round_manager.flush_buffer()
        game_state.round_manager.sync_round_data()

        if reason:
            print(Fore.BLUE + f"📁 Round {self.round_count} finished ({reason})")
        else:
            # Lightweight marker; keeps console noise minimal during large runs.
            print(Fore.BLUE + f"📁 Round {self.round_count} finished")

    # -------------------
    # LOGGING INFRASTRUCTURE - Used by all tasks
    # -------------------

    def setup_logging(self, base_dir: str, task_name: str) -> None:
        """Set up logging directory structure.
        
        Args:
            base_dir: Base logs directory (e.g., "logs/")
            task_name: Task identifier (e.g., "heuristics", "rl", "llm")
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, f"{task_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

    def save_session_summary(self) -> None:
        """Save session-level statistics to JSON."""
        if self.log_dir:
            stats_manager = GameStatsManager()
            stats_manager.save_session_stats(self.log_dir)

    # -------------------
    # EXTENSION CONVENIENCE HELPERS - Optional utilities for extensions
    # -------------------

    def get_game_json_filename(self, game_number: Optional[int] = None) -> str:
        """Return canonical filename for a game_N.json.
        
        Args:
            game_number: 1-based game number; defaults to current counter or 1 if unset.
        """
        from core.game_file_manager import FileManager  # local import to avoid cycles
        file_manager = FileManager()
        number = game_number if game_number is not None else max(1, int(self.game_count))
        return file_manager.get_game_json_filename(number)

    def get_game_json_path(self, game_number: Optional[int] = None) -> str:
        """Return absolute path to the canonical game_N.json under current log_dir."""
        if not self.log_dir:
            raise RuntimeError("log_dir is not set. Call setup_logging() first.")
        from core.game_file_manager import FileManager  # local import
        file_manager = FileManager()
        filename = self.get_game_json_filename(game_number)
        return file_manager.join_log_path(self.log_dir, filename)

    def display_basic_results(self) -> None:
        """Print minimal per-game results (score and steps)."""
        if not self.game or not hasattr(self.game, "game_state"):
            return
        print(Fore.BLUE + f"📊 Score: {self.game.game_state.score}, Steps: {self.game.game_state.steps}")

    def reset_for_next_game(self) -> None:
        """Standard reset routine to prepare for the next game.
        
        Safe for extensions to call at the end of a game.
        """
        # Reset per-game flags/counters for the upcoming game
        self.need_new_plan = True
        self.game_active = True
        self.current_game_moves = []
        self.round_count = 1
        if self.game:
            self.game.reset()
        # Reset first-plan flag for the new game
        self._first_plan = True
        # Reset all consecutive counters if present (Task-0 exposes limits_manager)
        if hasattr(self, 'limits_manager'):
            try:
                self.limits_manager.reset_all_counters()
            except Exception:
                pass
        # Generic counters used broadly
        self.consecutive_empty_steps = 0
        self.consecutive_something_is_wrong = 0
        self.consecutive_invalid_reversals = 0
        self.consecutive_no_path_found = 0


# -------------------
# TASK-0 SPECIFIC CLASS - LLM Snake Game
# -------------------


class GameManager(BaseGameManager):
    """LLM-powered Snake game manager (Task-0).
    
    Extends BaseGameManager with LLM-specific functionality:
    - Language model clients
    - Prompt/response logging  
    - Token usage tracking
    - LLM-specific error handling
    
    This is the ONLY class that should import LLM modules.
    """

    # Use LLM-capable game logic
    GAME_LOGIC_CLS = GameLogic

    def __init__(
        self, args: "argparse.Namespace", agent: Optional[BaseAgent] = None
    ) -> None:
        """Initialize LLM-specific session."""
        super().__init__(args)

        # -------------------
        # Elegant Consecutive Limits Management System
        # -------------------
        from core.game_limits_manager import create_limits_manager
        self.limits_manager = create_limits_manager(args)

        # -------------------
        # LLM-specific counters and state
        # -------------------
        self.empty_steps: int = 0
        self.something_is_wrong_steps: int = 0
        self.consecutive_empty_steps: int = 0
        self.consecutive_something_is_wrong: int = 0
        self.awaiting_plan: bool = False
        self.skip_empty_this_tick: bool = False

        # -------------------
        # LLM performance tracking
        # -------------------
        self.time_stats: defaultdict[str, int] = defaultdict(int)
        self.token_stats: dict[str, defaultdict[str, int]] = {
            "primary": defaultdict(int),
            "secondary": defaultdict(int),
        }

        # -------------------
        # LLM infrastructure
        # -------------------
        self.llm_client: Optional[LLMClient] = None
        self.parser_provider: Optional[str] = None
        self.parser_model: Optional[str] = None
        self.agent: Optional[BaseAgent] = agent

        # LLM-specific logging directories
        self.prompts_dir: Optional[str] = None
        self.responses_dir: Optional[str] = None

    def initialize(self) -> None:
        """Initialize LLM clients and logging infrastructure."""
        helper = GameManagerHelper()
        helper.initialize_game_manager(self)

    def setup_logging(self, base_dir: str, task_name: str = "llm") -> None:
        """Set up LLM-specific logging directories."""
        super().setup_logging(base_dir, task_name)
        if self.log_dir:
            # Create LLM-specific directories using centralized utilities
            from llm.log_utils import ensure_llm_directories
            self.prompts_dir, self.responses_dir = ensure_llm_directories(self.log_dir)
            # Convert Path objects to plain strings for JSON serialisation
            self.prompts_dir = str(self.prompts_dir)
            self.responses_dir = str(self.responses_dir)

    def create_llm_client(self, provider: str, model: Optional[str] = None) -> LLMClient:
        """Create LLM client for the specified provider."""
        return LLMClient(provider=provider, model=model)

    def run(self) -> None:
        """Execute LLM game session."""
        try:
            # For continuation mode, we need to set up LLM clients but skip other initialization
            if getattr(self.args, "is_continuation", False):
                # Set up LLM clients for continuation mode
                from utils.initialization_utils import setup_llm_clients, initialize_game_state
                setup_llm_clients(self)
                initialize_game_state(self)
            else:
                # Full initialization for new sessions
                self.initialize()

            # Run game loop until completion
            while self.game_count < self.args.max_games and self.running:
                run_game_loop(self)
                if self.game_count >= self.args.max_games:
                    break

        finally:
            # Cleanup and reporting
            # Graceful SDL shutdown (only if we ever initialised it)
            if self.use_gui and self._pygame and self._pygame.get_init():
                self._pygame.quit()
            self.report_final_statistics()

    def process_events(self) -> None:
        """Handle pygame events and user input."""
        helper = GameManagerHelper()
        helper.process_events(self)

    def report_final_statistics(self) -> None:
        """Generate comprehensive LLM session report."""
        if self.game_count == 0:
            return

        # Update session metadata
        self.save_session_summary()

        # Compile LLM-specific statistics
        stats_info = {
            "log_dir": self.log_dir,
            "game_count": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "empty_steps": self.empty_steps,
            "something_is_wrong_steps": self.something_is_wrong_steps,
            "valid_steps": self.valid_steps,
            "invalid_reversals": self.invalid_reversals,
            "game": self.game,
            "time_stats": self.time_stats,
            "token_stats": self.token_stats,
            "round_counts": self.round_counts,
            "total_rounds": self.total_rounds,
            "max_games": self.args.max_games,
            "no_path_found_steps": self.no_path_found_steps,
        }

        # Generate report and mark session complete
        helper = GameManagerHelper()
        helper.report_final_statistics(stats_info)
        self.running = False

    # -------------------
    # CONTINUATION SUPPORT - LLM-specific feature
    # -------------------

    def continue_from_session(self, log_dir: str, start_game_number: int) -> None:
        """Resume LLM session from previous checkpoint."""
        from utils.continuation_utils import setup_continuation_session, handle_continuation_game_state

        print(Fore.GREEN + f"🔄 Resuming LLM session from: {log_dir}")
        print(Fore.GREEN + f"🔄 Starting at game: {start_game_number}")

        setup_continuation_session(self, log_dir, start_game_number)

        # Initialize LLM clients with health check
        from utils.initialization_utils import setup_llm_clients, enforce_launch_sleep
        setup_llm_clients(self)
        enforce_launch_sleep(self.args)

        # Restore game state
        handle_continuation_game_state(self)
        self.run()

    @classmethod
    def continue_from_directory(cls, args: "argparse.Namespace") -> "GameManager":
        """Factory method for creating continuation sessions."""
        from utils.continuation_utils import continue_from_directory
        return continue_from_directory(cls, args)
