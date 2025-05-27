"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.
"""

import os
import json
import time
import pygame
from pygame.locals import *
from core.game_controller import GameController
from utils.replay_utils import extract_apple_positions
from config import TIME_DELAY, TIME_TICK

class ReplayEngine(GameController):
    """Engine for replaying recorded Snake games."""
    
    def __init__(self, log_dir, move_pause=1.0, auto_advance=False, use_gui=True):
        """Initialize the replay engine.
        
        Args:
            log_dir: Directory containing game logs
            move_pause: Time in seconds to pause between moves
            auto_advance: Whether to automatically advance through games
            use_gui: Whether to use GUI for display
        """
        super().__init__(use_gui=use_gui)
        
        # Initialize replay parameters
        self.log_dir = log_dir
        self.pause_between_moves = move_pause
        self.auto_advance = auto_advance
        
        # Game state specific to replay
        self.game_number = 1
        self.apple_positions = []
        self.apple_index = 0
        self.moves = []
        self.move_index = 0
        self.moves_made = []
        self.game_stats = {}
        self.last_move_time = time.time()
        self.running = True
        self.paused = False
        
        # Game statistics from the log file
        self.game_end_reason = None
        self.primary_llm = None
        self.secondary_llm = None
        self.game_timestamp = None
    
    def set_gui(self, gui_instance):
        """Set the GUI instance to use for display.
        
        Args:
            gui_instance: Instance of a GUI class for replay
        """
        super().set_gui(gui_instance)
        # Sync the GUI paused state with the replay engine
        if hasattr(gui_instance, 'set_paused'):
            gui_instance.set_paused(self.paused)
    
    def draw(self):
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            self.gui.draw(
                snake_positions=self.snake_positions,
                apple_position=self.apple_position if self.apple_index < len(self.apple_positions) else None,
                game_number=self.game_number,
                score=self.score,
                steps=self.steps,
                move_index=self.move_index,
                total_moves=len(self.moves),
                current_direction=self.current_direction,
                game_end_reason=self.game_end_reason,
                primary_llm=self.primary_llm,
                secondary_llm=self.secondary_llm,
                game_timestamp=self.game_timestamp
            )
    
    def load_game_data(self, game_number):
        """Load game data for a specific game number.
        
        Args:
            game_number: The game number to load
            
        Returns:
            Game data dictionary or None if loading failed
        """
        # Support both old and new filename formats
        new_format_file = os.path.join(self.log_dir, f"game_{game_number}.json")
        old_format_file = os.path.join(self.log_dir, f"game{game_number}.json")
        
        # Check which file exists (prefer new format)
        if os.path.exists(new_format_file):
            summary_file = new_format_file
        elif os.path.exists(old_format_file):
            summary_file = old_format_file
        else:
            print(f"Game {game_number} data not found")
            return None
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # Get apple positions
            if 'detailed_history' in game_data and 'apple_positions' in game_data['detailed_history']:
                self.apple_positions = game_data['detailed_history']['apple_positions']
            elif 'apple_positions' in game_data:
                self.apple_positions = game_data['apple_positions']
            else:
                print("No apple positions found in game data")
                return None
            
            # Get moves
            if 'detailed_history' in game_data and 'moves' in game_data['detailed_history']:
                self.moves = game_data['detailed_history']['moves']
            elif 'moves' in game_data:
                self.moves = game_data['moves']
            else:
                print("No moves found in game data")
                return None
            
            # Reset move index
            self.move_index = 0
            self.apple_index = 0
            self.moves_made = []
            self.score = 0
            self.steps = 0
            
            # Extract additional information for display
            self.game_end_reason = game_data.get('game_end_reason', None)
            
            # Get LLM information if available
            if 'llm_info' in game_data:
                llm_info = game_data['llm_info']
                self.primary_llm = f"{llm_info.get('primary_provider', 'Unknown')}/{llm_info.get('primary_model', 'Unknown')}"
                if llm_info.get('parser_provider') and llm_info.get('parser_provider').lower() != 'none':
                    self.secondary_llm = f"{llm_info.get('parser_provider', 'None')}/{llm_info.get('parser_model', 'None')}"
                else:
                    self.secondary_llm = "None/None"
            else:
                self.primary_llm = "Unknown/Unknown"
                self.secondary_llm = "Unknown/Unknown"
            
            # Get timestamp if available
            if 'metadata' in game_data and 'timestamp' in game_data['metadata']:
                self.game_timestamp = game_data['metadata']['timestamp']
            else:
                self.game_timestamp = "Unknown"
            
            # Save other game stats
            self.game_stats = game_data
            
            print(f"Loaded {len(self.apple_positions)} apple positions and {len(self.moves)} moves")
            print(f"Game {game_number}: End reason: {self.game_end_reason}, LLM: {self.primary_llm}")
            
            # Reset the game state
            self.reset()
            
            # Set the initial apple position
            if self.apple_positions:
                first_apple = self.apple_positions[0]
                if 'x' in first_apple and 'y' in first_apple:
                    self.apple_position = [first_apple['x'], first_apple['y']]
                
            # Update the board
            self._update_board()
            
            # Reset GUI move history if available
            if self.use_gui and self.gui and hasattr(self.gui, 'move_history'):
                self.gui.move_history = []
            
            return game_data
            
        except Exception as e:
            print(f"Error loading game data: {e}")
            return None
    
    def update(self):
        """Update game state for each frame."""
        if self.paused:
            return
            
        current_time = time.time()
        
        # Check if it's time for the next move
        if current_time - self.last_move_time >= self.pause_between_moves and self.move_index < len(self.moves):
            # Make the next move
            next_move = self.moves[self.move_index]
            self.move_index += 1
            self.moves_made.append(next_move)
            
            # Update game state
            game_continues, apple_eaten = self.make_move(next_move)
            
            # Update last move time
            self.last_move_time = current_time
            
            # Check if game is over
            if not game_continues:
                print(f"Game {self.game_number} over. Score: {self.score}, Steps: {self.steps}, End reason: {self.game_end_reason}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(1000)  # 1 second pause
                    self.game_number += 1
                    if not self.load_game_data(self.game_number):
                        print(f"No more games to load. Replay complete.")
                        self.running = False
            
            # Check if we're out of moves
            elif self.move_index >= len(self.moves):
                print(f"Replay complete for game {self.game_number}. Score: {self.score}, Steps: {self.steps}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(1000)  # 1 second pause
                    self.game_number += 1
                    if not self.load_game_data(self.game_number):
                        print(f"No more games to load. Replay complete.")
                        self.running = False
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
    
    def run(self):
        """Run the replay loop."""
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
            
        clock = pygame.time.Clock()
        
        # Load first game
        if not self.load_game_data(self.game_number):
            print(f"Could not load game {self.game_number}. Trying next game.")
            self.game_number += 1
            if not self.load_game_data(self.game_number):
                print("No valid games found in log directory.")
                return
        
        # Main game loop
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw game state
            if self.use_gui and self.gui:
                self.draw()
            
            # Control game speed
            pygame.time.delay(TIME_DELAY)
            clock.tick(TIME_TICK)
        
        # Clean up
        pygame.quit() 