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


class ReplayEngine:
    """Handles the replay engine functionality."""

    def __init__(self, game: SnakeGame, gui, log_dir: str, game_number: int = None, speed: float = 1.0):
        """Initialize the replay engine.
        
        Args:
            game: SnakeGame instance
            gui: GUI instance (e.g., ReplayGUI)
            log_dir: Directory containing game logs
            game_number: Specific game number to replay
            speed: Replay speed multiplier
        """
        self.game = game
        self.gui = gui # Store the gui instance
        self.log_dir = log_dir
        self.game_to_load = game_number
        self.speed = speed
        
        # Game state
        self.current_move_index = 0
        self.paused = False
        self.all_moves_for_game = [] # Will hold moves for the loaded game
        
        self.game_number_display = 1 # For display purposes in GUI
        self.round_number_display = 1 # For display purposes in GUI

        # Load game data immediately
        if not self._load_game_data():
            # Handle case where game data couldn't be loaded
            print(f"{Fore.RED}Failed to load game data. ReplayEngine will not run.{Fore.RESET}")
            self.all_moves_for_game = [] # Ensure it's empty

    def _load_game_data(self) -> bool:
        """Load game data from logs. Returns True if successful, False otherwise."""
        game_summary_files = []  # List to store game summary files
        try:
            for filename in os.listdir(self.log_dir):
                if filename.startswith("game") and filename.endswith("_summary.json"):
                    game_summary_files.append(os.path.join(self.log_dir, filename))
        except FileNotFoundError:
            print(f"{Fore.RED}Log directory not found: {self.log_dir}{Fore.RESET}")
            return False
        
        if not game_summary_files:
            print(f"{Fore.YELLOW}No game summary files found in {self.log_dir}{Fore.RESET}")
            return False
            
        # Sort files by game number
        game_summary_files.sort(key=lambda x: int(os.path.basename(x).split("game")[1].split("_")[0]))
        
        summary_file_to_load = None
        if self.game_to_load is not None:
            if 0 <= self.game_to_load < len(game_summary_files):
                summary_file_to_load = game_summary_files[self.game_to_load]
                self.game_number_display = self.game_to_load + 1  # Update display number
            else:
                print(f"{Fore.RED}Invalid game number: {self.game_to_load}. Valid range: 0-{len(game_summary_files)-1}{Fore.RESET}")
                return False
        elif game_summary_files:  # Default to the first game if no specific game is requested
            summary_file_to_load = game_summary_files[0]
            self.game_number_display = 1

        if not summary_file_to_load:
            print(f"{Fore.YELLOW}No specific game summary file to load.{Fore.RESET}")
            return False
        
        try:
            with open(summary_file_to_load, "r") as f:
                data = json.load(f)
                
                # Check if using old format with "moves" array
                if "moves" in data:
                    # Check if moves is a list
                    if isinstance(data["moves"], list):
                        self.all_moves_for_game = data["moves"]
                    # Check if moves is a dictionary with round keys
                    elif isinstance(data["moves"], dict):
                        # Convert dictionary of moves by round to flat list of moves
                        moves_list = []
                        # Sort round keys numerically
                        sorted_rounds = sorted(data["moves"].keys(), 
                                              key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                        for round_key in sorted_rounds:
                            move = data["moves"][round_key]
                            moves_list.append(move)
                        self.all_moves_for_game = moves_list
                    else:
                        print(f"{Fore.RED}Unexpected format for 'moves' in {os.path.basename(summary_file_to_load)}{Fore.RESET}")
                        return False
                
                # Check if using new format with "rounds_data"
                elif "rounds_data" in data:
                    # Extract moves from rounds_data and flatten into a list
                    moves_list = []
                    # Sort round keys numerically
                    sorted_rounds = sorted(data["rounds_data"].keys(),
                                         key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                    for round_key in sorted_rounds:
                        round_data = data["rounds_data"][round_key]
                        if "moves" in round_data:
                            moves_list.append(round_data["moves"])
                    self.all_moves_for_game = moves_list
                else:
                    print(f"{Fore.RED}Game summary file {os.path.basename(summary_file_to_load)} is missing 'moves' and 'rounds_data'.{Fore.RESET}")
                    return False
                
                if not self.all_moves_for_game:
                    print(f"{Fore.YELLOW}Loaded game summary {os.path.basename(summary_file_to_load)} has no moves.{Fore.RESET}")
                return True
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error decoding JSON from {summary_file_to_load}{Fore.RESET}")
            return False
        except Exception as e:
            print(f"{Fore.RED}Error loading game data from {summary_file_to_load}: {e}{Fore.RESET}")
            return False

    def _apply_next_move(self):
        """Applies the next move from the loaded game data."""
        if self.current_move_index < len(self.all_moves_for_game):
            move = self.all_moves_for_game[self.current_move_index]
            self.game.move(move)
            self.current_move_index += 1

    def _revert_to_previous_move(self):
        """Reverts to the state before the current_move_index by replaying from start."""
        if self.current_move_index > 0:
            self.game.reset()
            # Go to the state before the current index by replaying up to index-1
            previous_target_index = self.current_move_index - 1
            self.current_move_index = 0 # Reset internal counter before replaying
            for _ in range(previous_target_index):
                if self.current_move_index < len(self.all_moves_for_game):
                     self._apply_next_move() # Use the internal method that increments index
                else:
                    break

    def _process_events(self) -> bool:
        """Process pygame events. Returns False if replay should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{Fore.CYAN}Replay {'Paused' if self.paused else 'Resumed'}{Fore.RESET}")
                elif event.key == pygame.K_RIGHT:
                    if not self.paused:
                        self._apply_next_move()
                    else:
                        # Allow step-forward even if paused
                        self._apply_next_move()
                        print(f"{Fore.CYAN}Stepped forward to move {self.current_move_index}{Fore.RESET}")
                elif event.key == pygame.K_LEFT:
                    if not self.paused:
                         self._revert_to_previous_move()
                    else:
                        # Allow step-backward even if paused
                        self._revert_to_previous_move()
                        print(f"{Fore.CYAN}Stepped backward to move {self.current_move_index}{Fore.RESET}")
                elif event.key == pygame.K_r:
                    print(f"{Fore.CYAN}Restarting replay of game {self.game_number_display}{Fore.RESET}")
                    self.game.reset()
                    self.current_move_index = 0
                    # Potentially re-load or just reset state if that's preferred for 'r'
                elif event.key == pygame.K_q:
                    print(f"{Fore.CYAN}Quitting replay.{Fore.RESET}")
                    return False
        return True

    def run(self):
        """Run the replay engine's main loop."""
        if not self.all_moves_for_game: # Don't run if no moves loaded
            print(f"{Fore.YELLOW}No game data to replay. Exiting replay engine.{Fore.RESET}")
            return
            
        print(f"{Fore.GREEN}Starting replay of game {self.game_number_display} with {len(self.all_moves_for_game)} moves. Speed: {self.speed}x{Fore.RESET}")
        self.game.reset() # Ensure clean game state at start
        self.current_move_index = 0

        running = True
        while running:
            running = self._process_events()
            if not running: # Exit if _process_events returns False (e.g. Quit event)
                break

            if not self.paused:
                if self.current_move_index < len(self.all_moves_for_game):
                    self._apply_next_move()
                # else: game has finished replaying all moves
            
            if self.gui: # Check if GUI is provided
                # The ReplayGUI.draw expects: game, game_number, round_number, current_move
                self.gui.draw(self.game, self.game_number_display, self.round_number_display, self.current_move_index)
                pygame.display.flip()
            
            pygame.time.delay(int(1000 / self.speed)) 

        print(f"{Fore.GREEN}Replay finished.{Fore.RESET}") 