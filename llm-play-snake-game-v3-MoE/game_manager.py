"""
Game manager module for the Snake game.
Manages game sessions, initialization, and statistics reporting.
"""

import os
import json
import time
import pygame
import datetime
from colorama import Fore
from core.game_engine import GameEngine
from core.snake_game import SnakeGame
from gui.game_gui import GameGUI
from llm_client import LLMClient, LLMOutputParser
from utils.log_utils import save_to_file
from utils.game_stats_utils import update_experiment_info, report_final_statistics


class GameManager:
    """Manages game sessions and initialization."""

    def __init__(self, config):
        """Initialize the game manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.game = None
        self.gui = None
        self.game_engine = None
        self.llm_client = None
        self.parser_client = None
        self.log_dir = None
        self.game_count = 0
        self.total_score = 0
        self.total_steps = 0
        self.parser_usage_count = 0
        self.game_scores = []
        self.empty_steps = 0
        self.error_steps = 0
        self.max_empty_moves = config.get("max_empty_moves", 3)
        self.json_error_stats = {"count": 0, "responses": []}

    def check_llm_health(self):
        """Check if the LLM clients are healthy."""
        try:
            # Initialize primary LLM client
            self.llm_client = LLMClient(
                provider=self.config.get("llm_provider", "openai"),
                model=self.config.get("llm_model")
            )
            
            # Initialize parser LLM client
            self.parser_client = LLMOutputParser(
                provider=self.config.get("parser_provider", "openai"),
                model=self.config.get("parser_model")
            )
            
            # Test the clients
            test_response = self.llm_client.generate_response("Test")
            if not test_response:
                raise Exception("Primary LLM client test failed")
                
            test_parsed, _ = self.parser_client.parse_and_format(
                test_response,
                (0, 0),
                (1, 1),
                [(0, 0)]
            )
            if not test_parsed:
                raise Exception("Parser LLM client test failed")
                
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error checking LLM health: {str(e)}{Fore.RESET}")
            return False

    def initialize_game(self):
        """Initialize the game and GUI."""
        # Initialize pygame
        pygame.init()
        
        # Create game instance
        self.game = SnakeGame(
            grid_size=self.config.get("grid_size", 10),
            cell_size=self.config.get("cell_size", 40)
        )
        
        # Create GUI
        self.gui = GameGUI(
            grid_size=self.config.get("grid_size", 10),
            cell_size=self.config.get("cell_size", 40)
        )
        
        # Create game engine
        self.game_engine = GameEngine(
            game=self.game,
            llm_client=self.llm_client,
            parser_client=self.parser_client,
            move_delay=self.config.get("move_delay", 0.5),
            max_steps=self.config.get("max_steps", 100),
            max_empty_moves=self.max_empty_moves
        )
        
        # Create log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(
            self.config.get("log_dir", "logs"),
            f"game_session_{timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        config_file = os.path.join(self.log_dir, "config.json")
        save_to_file(config_file, self.config)

    def run(self):
        """Run the game session."""
        # Check LLM health
        if not self.check_llm_health():
            print(f"{Fore.RED}LLM health check failed. Exiting.{Fore.RESET}")
            return
            
        # Initialize game
        self.initialize_game()
        
        # Run game sessions
        num_sessions = self.config.get("num_sessions", 1)
        for session in range(num_sessions):
            print(f"\n{Fore.CYAN}Starting game session {session + 1}/{num_sessions}{Fore.RESET}")
            
            # Run the game
            self.game_engine.run_game()
            
            # Update statistics
            self.game_count += 1
            self.total_score += self.game.score
            self.total_steps += self.game.steps
            self.parser_usage_count += self.game_engine.parser_usage_count
            self.game_scores.append(self.game.score)
            self.empty_steps += self.game_engine.empty_steps
            self.error_steps += self.game_engine.error_steps
            
            # Update experiment info
            update_experiment_info(
                self.log_dir,
                self.game_count,
                self.total_score,
                self.total_steps,
                self.json_error_stats,
                self.parser_usage_count,
                self.game_scores,
                self.empty_steps,
                self.error_steps,
                self.max_empty_moves
            )
            
            # Reset game for next session
            self.game.reset()
            
        # Report final statistics
        report_final_statistics(
            self.log_dir,
            self.game_count,
            self.total_score,
            self.total_steps,
            self.parser_usage_count,
            self.game_scores,
            self.empty_steps,
            self.error_steps,
            self.max_empty_moves
        )
        
        # Clean up
        pygame.quit() 