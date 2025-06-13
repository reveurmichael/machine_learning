"""
Game manager module for the Snake game.
Handles game session management, initialization, and statistics tracking.
"""

import pygame
from colorama import Fore

# Core game components
from core.game_logic import GameLogic
from core.game_loop import run_game_loop
from gui.game_gui import GameGUI
from llm.client import LLMClient
from config import TIME_DELAY, TIME_TICK

# Utils imports - organized by functionality
from utils.json_utils import save_session_stats
from utils.continuation_utils import continue_from_directory, setup_continuation_session, handle_continuation_game_state, setup_llm_clients
from utils.game_manager_utils import (
    report_final_statistics,
    initialize_game_manager,
    process_events
)


class GameManager:
    """Manages the overall game session, including multiple games and statistics."""
    
    def __init__(self, args):
        """Initialize the game manager.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Game counters and statistics
        self.game_count = 0
        self.round_count = 1  # Start at 1 for intuitive round numbering
        self.total_score = 0
        self.total_steps = 0
        self.empty_steps = 0
        self.something_is_wrong_steps = 0
        self.valid_steps = 0
        self.invalid_reversals = 0
        self.consecutive_empty_steps = 0
        self.consecutive_something_is_wrong = 0
        self.game_scores = []
        self.previous_parser_usage = 0
        
        # Time and token statistics
        self.time_stats = {
            "llm_communication_time": 0,
            "game_movement_time": 0,
            "waiting_time": 0
        }
        
        self.token_stats = {
            "primary": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        }
        
        # Game state
        self.game = None
        self.game_active = True
        self.need_new_plan = True
        self.awaiting_plan = False     # Whether we're currently waiting for a new plan from the LLM
        self.running = True
        
        # Track moves for this game
        self.current_game_moves = []
        
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
        
        # GUI settings
        self.use_gui = not args.no_gui
    
    def create_llm_client(self, provider, model=None):
        """Create an LLM client with the specified provider and model.
        
        Args:
            provider: LLM provider name
            model: Model name (optional)
            
        Returns:
            LLMClient instance
        """
        return LLMClient(provider=provider, model=model)
    
    def setup_game(self):
        """Set up the game logic and GUI."""
        # Initialize game logic
        self.game = GameLogic(use_gui=self.use_gui)
        
        # Set up the GUI if enabled
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)
    
    def get_pause_between_moves(self):
        """Get the pause time between moves.
        
        Returns:
            Float representing pause time in seconds, 0 if no GUI is enabled
        """
        # Skip pause in no-gui mode
        if not self.use_gui:
            return 0.0
        
        return self.args.move_pause
    
    def initialize(self):
        """Initialize the game, LLM clients, and logging directories."""
        initialize_game_manager(self)
    
    def process_events(self):
        """Process pygame events."""
        process_events(self)
    
    def run_game_loop(self):
        """Run the main game loop."""
        run_game_loop(self)
    
    def report_final_statistics(self):
        """Report final statistics at the end of the game session."""
        # Only report if games were played
        if self.game_count == 0:
            return
            
        # Update summary.json metadata at session end (no JSON-parser stats anymore)
        save_session_stats(self.log_dir)
        
        # Initialize step statistics for valid steps and invalid reversals
        valid_steps = 0
        invalid_reversals = 0
        
        # Get step statistics from the game state if available
        if self.game and hasattr(self.game, "game_state"):
            valid_steps = self.game.game_state.valid_steps
            invalid_reversals = self.game.game_state.invalid_reversals
        
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
            "max_consecutive_empty_moves_allowed": self.args.max_consecutive_empty_moves_allowed,
            "max_consecutive_something_is_wrong_allowed": self.args.max_consecutive_something_is_wrong_allowed,
            "game": self.game,
            "time_stats": self.time_stats,
            "token_stats": self.token_stats
        }
        
        # Report statistics to console and save to files
        report_final_statistics(stats_info)
    
    def increment_round(self, reason=""):
        """Increment the round counter and synchronize with game state.
        
        This centralized method ensures consistent round counting across the codebase.
        It properly increments round_count and synchronizes the count between
        the game manager and game state for accurate replay functionality.
        
        IMPORTANT: Rounds should ONLY be incremented when:
        1. We get a new plan from the LLM
        
        Rounds should NOT be incremented when:
        1. An apple is eaten during planned moves
        2. A game ends
        
        This ensures the number of rounds in game_N.json matches the
        number of prompts/responses in the logs directory.
        
        Args:
            reason: Optional reason for incrementing the round (for logging)
        """
        # Skip if we're still awaiting a plan (prevents duplicate increment)
        if self.awaiting_plan:
            return
            
        # Store old round count to check if it actually changed
        old_round_count = self.round_count

        # -----------------------------------------------------------------
        # Persist and reset the outgoing round BEFORE we bump the counter
        # This guarantees moves from round N never leak into round N+1.
        # -----------------------------------------------------------------
        if self.game and hasattr(self.game, "game_state"):
            try:
                self.game.game_state._flush_current_round()
            except Exception as e:
                print(Fore.YELLOW + f"⚠️  Could not flush round data: {e}")

        # Increment round counter – must happen AFTER the flush
        # so that new moves land in a fresh buffer
        self.round_count += 1
        
        # Sync with game state
        if self.game and hasattr(self.game, "game_state"):
            self.game.game_state.round_count = self.round_count

            # The new (empty) buffer is already in place; still call sync to
            # guarantee any external fields (apple_position, etc.) stay aligned
            self.game.game_state.sync_round_data()
        
        # Only print the banner if the round count actually changed
        if self.round_count != old_round_count:
            # Log the round increment with reason if provided
            reason_text = f" ({reason})" if reason else ""
            print(Fore.BLUE + f"📊 Advanced to round {self.round_count}{reason_text}")
    
    def continue_from_session(self, log_dir, start_game_number):
        """Continue from a previous game session.
        
        Args:
            log_dir: Directory containing the previous session logs
            start_game_number: Game number to start from
        """
        print(Fore.GREEN + f"🔄 Continuing experiment from directory: {log_dir}")
        print(Fore.GREEN + f"🔄 Starting from game number: {start_game_number}")
        
        # Set up continuation session
        setup_continuation_session(self, log_dir, start_game_number)
        
        # Set up LLM clients with the configuration from the original experiment
        setup_llm_clients(self)
        
        # Handle game state for continuation
        handle_continuation_game_state(self)
        
        # Run the game loop
        self.run()
        
    @classmethod
    def continue_from_directory(cls, args):
        """Factory method to create a GameManager instance for continuation.
        
        Args:
            args: Command line arguments with continue_with_game_in_dir set
            
        Returns:
            GameManager instance configured for continuation
        """
        return continue_from_directory(cls, args)
    
    def run(self):
        """Initialize and run the game session."""
        try:
            # Skip initialization if this is a continuation
            if not hasattr(self.args, 'is_continuation') or not self.args.is_continuation:
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