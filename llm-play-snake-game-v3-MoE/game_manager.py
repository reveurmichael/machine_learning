"""
Game manager module for the Snake game.
Manages game sessions, initialization, and statistics reporting.
"""

import os
import json
import time
import pygame
import datetime
import traceback
from colorama import Fore
from core.game_engine import GameEngine
from core.snake_game import SnakeGame
from gui.game_gui import GameGUI
from llm_client import LLMClient, LLMOutputParser
from utils.log_utils import save_to_file, save_experiment_info_json
from utils.game_stats_utils import update_experiment_info, report_final_statistics


class GameManager:
    """Manages game sessions and initialization."""

    def __init__(self, config):
        """Initialize the game manager.
        
        Args:
            config: Configuration dictionary or Namespace from argparse
        """
        # Convert Namespace to dictionary if needed
        self.config = vars(config) if not isinstance(config, dict) else config
        
        # Main game objects
        self.game = None
        self.gui = None
        self.game_engine = None
        self.llm_client = None
        self.parser_client = None
        
        # Logging
        self.log_dir = None
        
        # Statistics
        self.game_count = 0
        self.total_score = 0
        self.total_steps = 0
        self.parser_usage_count = 0
        self.game_scores = []
        self.empty_steps = 0
        self.error_steps = 0
        self.max_empty_moves = self.config.get("max_empty_moves", 3)
        self.json_error_stats = {"count": 0, "responses": []}

    def check_llm_health(self):
        """Check if the LLM clients are healthy."""
        try:
            # Get provider and model info from config
            provider = self.config.get("provider")
            model = self.config.get("model")
            parser_provider = self.config.get("parser_provider") if self.config.get("parser_provider") else provider
            parser_model = self.config.get("parser_model")
            
            print(f"{Fore.CYAN}Initializing primary LLM client: {provider} / {model}{Fore.RESET}")
            
            # Initialize primary LLM client
            self.llm_client = LLMClient(
                provider=provider,
                model=model
            )
            
            print(f"{Fore.CYAN}Initializing parser LLM client: {parser_provider} / {parser_model}{Fore.RESET}")
            
            # Initialize parser LLM client
            self.parser_client = LLMOutputParser(
                provider=parser_provider,
                model=parser_model
            )
            
            # Skip test for 'none' provider
            if provider == 'none':
                print(f"{Fore.YELLOW}Primary LLM provider is 'none' - skipping health check{Fore.RESET}")
                return True
                
            # Test the primary client with a simple prompt
            print(f"{Fore.CYAN}Testing primary LLM client...{Fore.RESET}")
            test_response = self.llm_client.generate_response("Test prompt for health check.")
            if test_response.startswith("ERROR"):
                raise Exception(f"Primary LLM client test failed: {test_response}")
            
            # Skip parser test if using 'none' provider
            if parser_provider == 'none':
                print(f"{Fore.YELLOW}Parser provider is 'none' - skipping parser health check{Fore.RESET}")
                return True
                
            # Test the parser client
            print(f"{Fore.CYAN}Testing parser LLM client...{Fore.RESET}")
            test_parsed, _ = self.parser_client.parse_and_format(
                "Example response for test",
                (0, 0),
                (1, 1),
                [(0, 0)]
            )
            if test_parsed.startswith("ERROR"):
                raise Exception(f"Parser LLM client test failed: {test_parsed}")
                
            print(f"{Fore.GREEN}LLM health check passed{Fore.RESET}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error checking LLM health: {str(e)}{Fore.RESET}")
            traceback.print_exc()
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
            move_delay=self.config.get("move_pause", 1.0),
            max_steps=self.config.get("max_steps", 400),
            max_empty_moves=self.max_empty_moves
        )
        
        # Create log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(
            self.config.get("log_dir", "logs"),
            f"game_session_{timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set the log directory for the game engine
        self.game_engine.set_log_dir(self.log_dir)
        
        # Save experiment configuration and metadata to info.json file
        save_experiment_info_json(self.config, self.log_dir)

    def run(self):
        """Run the game session."""
        # Check LLM health
        if not self.check_llm_health():
            print(f"{Fore.RED}LLM health check failed. Exiting.{Fore.RESET}")
            return
            
        # Initialize game
        self.initialize_game()
        
        # Run game sessions
        max_games = self.config.get("max_games", 1)
        print(f"{Fore.CYAN}Will run up to {max_games} games{Fore.RESET}")
        
        for game_num in range(max_games):
            print(f"\n{Fore.CYAN}Starting game {game_num + 1}/{max_games}{Fore.RESET}")
            
            # Run the game
            self.game_engine.run_game()
            
            # Get statistics
            game_stats = self.game_engine.get_statistics()
            
            # Update statistics
            self.game_count += 1
            self.total_score += self.game.score
            self.total_steps += self.game.steps
            self.parser_usage_count += self.game_engine.parser_usage_count
            self.game_scores.append(self.game.score)
            self.empty_steps += self.game_engine.empty_steps
            self.error_steps += self.game_engine.error_steps
            
            # Calculate steps per apple (for performance metrics)
            steps_per_apple = self.game.steps / max(1, self.game.score)
            
            # Get token statistics for this game
            primary_token_stats = game_stats.get("primary_token_stats", {})
            secondary_token_stats = game_stats.get("secondary_token_stats", {})
            
            # Get response time statistics
            primary_response_stats = game_stats.get("primary_response_stats", {})
            secondary_response_stats = game_stats.get("secondary_response_stats", {})
            
            # Save game summary
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            game_summary_path = os.path.join(self.log_dir, f"game{game_num + 1}_summary.json")
            
            # Create summary with the most important info at the top
            game_summary = {
                "score": self.game.score,
                "steps": self.game.steps,
                "game_end_reason": self.game.collision_type,
                "snake_length": len(self.game.body) + 1,  # +1 for the head
                "timestamp": timestamp,
                "game_number": game_num + 1,
                "primary_provider": self.config.get("provider"),
                "primary_model": self.config.get("model"),
                "parser_provider": self.config.get("parser_provider"),
                "parser_model": self.config.get("parser_model"),
                "performance_metrics": {
                    "steps_per_apple": steps_per_apple
                },
                "prompt_response_stats": {
                    "avg_primary_response_time": primary_response_stats.get("avg_primary_response_time", 0),
                    "min_primary_response_time": primary_response_stats.get("min_primary_response_time", 0),
                    "max_primary_response_time": primary_response_stats.get("max_primary_response_time", 0),
                    "avg_secondary_response_time": secondary_response_stats.get("avg_secondary_response_time", 0),
                    "min_secondary_response_time": secondary_response_stats.get("min_secondary_response_time", 0),
                    "max_secondary_response_time": secondary_response_stats.get("max_secondary_response_time", 0)
                },
                "token_stats": {
                    "primary": primary_token_stats,
                    "secondary": secondary_token_stats
                },
                "parser_usage_count": self.game_engine.parser_usage_count,
                "rounds_data": game_stats.get("rounds_data", {})
            }
            
            # Save the summary
            with open(game_summary_path, 'w', encoding='utf-8') as f:
                json.dump(game_summary, f, indent=2)
            
            # Update experiment info with token statistics
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
                self.max_empty_moves,
                token_stats={
                    "primary": {
                        **primary_token_stats,
                        "response_times": self.game_engine.primary_response_times
                    },
                    "secondary": {
                        **secondary_token_stats,
                        "response_times": self.game_engine.secondary_response_times
                    }
                }
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