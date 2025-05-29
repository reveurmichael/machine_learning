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
from utils.json_utils import get_json_error_stats, save_session_stats
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
        self.round_count = 1  # Start at 1 to make round numbering more intuitive (1, 2, 3, ...)
        self.total_score = 0
        self.total_steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.consecutive_empty_steps = 0
        self.consecutive_errors = 0
        self.game_scores = []
        self.parser_usage_count = 0
        self.previous_parser_usage = 0
        
        # Game state
        self.game = None
        self.game_active = True
        self.need_new_plan = True
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
            
        # Update experiment info JSON
        save_session_stats(
            self.log_dir,
            json_error_stats=get_json_error_stats()
        )
        
        # Create stats dictionary
        stats_info = {
            "log_dir": self.log_dir,
            "game_count": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "parser_usage_count": self.parser_usage_count,
            "game_scores": self.game_scores,
            "empty_steps": self.empty_steps,
            "error_steps": self.error_steps,
            "max_empty_moves": self.args.max_empty_moves,
            "max_consecutive_errors_allowed": self.args.max_consecutive_errors_allowed
        }
        
        # Report statistics to console and save to files
        report_final_statistics(stats_info)
    
    def run(self):
        """Initialize and run the game session."""
        try:
            # Initialize the game and LLM clients
            self.initialize()
            
            # Run games until we reach max_game
            while self.game_count < self.args.max_game and self.running:
                # Run the game loop
                self.run_game_loop()
                
                # Check if we've reached the max games
                if self.game_count >= self.args.max_game:
                    print(Fore.GREEN + f"🏁 Reached maximum games ({self.args.max_game}). Session complete.")
                    break
            
        finally:
            # Final cleanup
            if self.use_gui and pygame.get_init():
                pygame.quit()
            
            # Report final statistics
            self.report_final_statistics() 