"""
Game manager module for the Snake game.
Handles game session management, initialization, and statistics reporting.
"""

import os
import time
import pygame
import traceback
import json
import sys
import glob
from datetime import datetime
from colorama import Fore
from core.game_logic import GameLogic
from gui.game_gui import GameGUI
from llm_client import LLMClient
from config import TIME_DELAY, TIME_TICK, PAUSE_BETWEEN_MOVES_SECONDS
from utils import (
    # Log utilities
    save_to_file,
    format_raw_llm_response,
    format_parsed_llm_response,
    generate_game_summary_json,
    
    # JSON utilities
    get_json_error_stats,
    reset_json_error_stats,
    save_experiment_info_json,
    update_experiment_info_json,
    
    # Game management utilities
    check_max_steps,
    process_game_over,
    handle_error,
    report_final_statistics
)

from utils.llm_utils import handle_llm_response, check_llm_health, parse_and_format
from utils.game_manager_utils import check_max_steps as utils_check_max_steps
from utils.game_manager_utils import process_game_over as utils_process_game_over
from utils.game_manager_utils import handle_error as utils_handle_error
from utils.game_manager_utils import report_final_statistics as utils_report_final_statistics
from utils.game_manager_utils import initialize_game_manager, process_events
from utils.continuation_utils import setup_continuation_session, setup_llm_clients, handle_continuation_game_state, continue_from_directory
from core.game_loop import run_game_loop


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
        self.round_count = 0
        self.total_score = 0
        self.total_steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.consecutive_empty_steps = 0
        self.consecutive_errors = 0  # Track consecutive errors
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
        # Set up the game
        self.game = GameLogic(use_gui=self.use_gui)
        
        # Set up the GUI if needed
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)
    
    def get_pause_between_moves(self):
        """Get the pause time between moves.
        
        Returns:
            Float representing pause time in seconds
        """
        return PAUSE_BETWEEN_MOVES_SECONDS
    
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
        
        # Check if this is a continuation mode
        is_continuation = False
        if hasattr(self.game.game_state, 'is_continuation'):
            is_continuation = self.game.game_state.is_continuation
            
        # Update experiment info JSON
        update_experiment_info_json(
            self.log_dir,
            is_continuation=is_continuation,
            json_error_stats=get_json_error_stats()
        )
        
        # Call the utility function
        report_final_statistics(
            self.log_dir,
            self.game_count,
            self.total_score,
            self.total_steps,
            self.parser_usage_count,
            self.game_scores,
            self.empty_steps,
            self.error_steps,
            self.args.max_empty_moves,
            self.args.max_consecutive_errors_allowed
        )
    
    def run(self):
        """Initialize and run the game session."""
        try:
            # Initialize the game and LLM clients
            self.initialize()
            
            # Run the game loop
            self.run_game_loop()
            
        finally:
            # Final cleanup
            if self.use_gui and pygame.get_init():
                pygame.quit()
            
            # Report final statistics
            self.report_final_statistics()
    
    def continue_from_session(self, log_dir, start_game_number):
        """Continue from a previous game session.
        
        Args:
            log_dir: Path to the log directory to continue from
            start_game_number: The game number to start from
        """
        # Set up continuation session
        setup_continuation_session(self, log_dir, start_game_number)
        
        # Set up LLM clients
        setup_llm_clients(self)
        
        # Handle game state for continuation
        handle_continuation_game_state(self)
        
        # Run the game loop
        self.run_game_loop()
        
        # Report final statistics
        self.report_final_statistics()

    @classmethod
    def continue_from_directory(cls, args):
        """Factory method to create a GameManager instance for continuation.
        
        Args:
            args: Command-line arguments with continue_with_game_in_dir set
            
        Returns:
            GameManager instance set up for continuation
        """
        return continue_from_directory(cls, args) 