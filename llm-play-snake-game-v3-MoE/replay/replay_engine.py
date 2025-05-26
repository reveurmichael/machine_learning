"""
Replay engine module for the Snake game.
Handles the replay engine functionality.
"""

import os
import json
import time
import pygame
import datetime
from colorama import Fore
from core.snake_game import SnakeGame
from gui.replay_gui import ReplayGUI


class ReplayEngine:
    """Handles the replay engine functionality."""

    def __init__(self, game, log_dir, game_number=None, speed=1.0):
        """Initialize the replay engine.
        
        Args:
            game: SnakeGame instance
            log_dir: Directory containing game logs
            game_number: Specific game number to replay
            speed: Replay speed multiplier
        """
        self.game = game
        self.log_dir = log_dir
        self.game_number = game_number
        self.speed = speed
        
        # Game state
        self.current_move = 0
        self.paused = False
        self.game_data = None
        
        # Load game data
        self.load_game_data()

    def load_game_data(self):
        """Load game data from logs."""
        # Find available game logs
        game_logs = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("game_") and filename.endswith(".json"):
                game_logs.append(os.path.join(self.log_dir, filename))
        
        if not game_logs:
            print(f"{Fore.RED}No game data found{Fore.RESET}")
            return
            
        # Sort logs by timestamp
        game_logs.sort()
        
        # Load specific game if requested
        if self.game_number is not None:
            if self.game_number < 0 or self.game_number >= len(game_logs):
                print(f"{Fore.RED}Invalid game number: {self.game_number}{Fore.RESET}")
                return
            game_logs = [game_logs[self.game_number]]
        
        # Load game data
        self.game_data = []
        for log_file in game_logs:
            with open(log_file, "r") as f:
                data = json.load(f)
                self.game_data.extend(data["moves"])

    def process_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT and self.current_move < len(self.game_data):
                    if not self.paused:
                        move = self.game_data[self.current_move]
                        self.game.move(move)
                        self.current_move += 1
                elif event.key == pygame.K_LEFT and self.current_move > 0:
                    if not self.paused:
                        self.game.reset()
                        self.current_move = 0
                        for i in range(self.current_move):
                            self.game.move(self.game_data[i])
                elif event.key == pygame.K_r:
                    self.game.reset()
                    self.current_move = 0
                elif event.key == pygame.K_q:
                    return False
        return True

    def run(self):
        """Run the replay engine."""
        if not self.game_data:
            return
            
        # Main replay loop
        running = True
        while running:
            # Process events
            running = self.process_events()
            
            # Draw the current state
            if self.game.gui:
                self.game.gui.draw(self.game, 1, 1, self.current_move)
                pygame.display.flip()
            
            # Control replay speed
            pygame.time.delay(int(1000 / self.speed)) 