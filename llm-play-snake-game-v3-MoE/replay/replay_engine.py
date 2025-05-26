"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.
"""

import os
import json
import time
import pygame
from pygame.locals import *
from core.game_engine import GameEngine
from utils.snake_utils import filter_invalid_reversals, extract_apple_positions

class ReplayEngine(GameEngine):
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
        self.move_pause = move_pause
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
    
    def set_gui(self, gui_instance):
        """Set the GUI instance to use for display.
        
        Args:
            gui_instance: Instance of a GUI class for replay
        """
        super().set_gui(gui_instance)
    
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
                current_direction=self.current_direction
            )
    
    def load_game_data(self, game_number):
        """Load data for a specific game number.
        
        Args:
            game_number: The game number to load
            
        Returns:
            Boolean indicating if game data was loaded successfully
        """
        # Reset game state
        self.snake_positions = [[self.grid_size // 2, self.grid_size // 2]]
        self.head_position = self.snake_positions[0]
        self.apple_positions = []
        self.apple_index = 0
        self.moves = []
        self.move_index = 0
        self.moves_made = []
        self.score = 0
        self.steps = 0
        self.current_direction = None
        
        # Load game data from summary file
        summary_file = os.path.join(self.log_dir, f"game{game_number}_summary.json")
        
        if not os.path.exists(summary_file):
            print(f"Game {game_number} summary file not found.")
            return False
        
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract game statistics
            self.game_stats = {
                'score': summary_data.get('score', 0),
                'steps': summary_data.get('steps', 0),
                'game_end_reason': summary_data.get('game_end_reason', 'Unknown'),
                'snake_length': summary_data.get('snake_length', 1)
            }
            
            # Load apple positions from JSON summary
            self.apple_positions = extract_apple_positions(self.log_dir, game_number)
            
            # Set initial apple position
            if self.apple_positions:
                self.apple_position = self.apple_positions[0]
                
            # Load moves from JSON summary
            if 'moves' in summary_data and summary_data['moves']:
                raw_moves = summary_data['moves']
                
                # Ensure moves are valid and no invalid reversals
                self.moves = filter_invalid_reversals(raw_moves)
                
                print(f"Loaded {len(self.moves)} moves for game {game_number}")
                
                if len(self.moves) != len(raw_moves):
                    print(f"Warning: Filtered out {len(raw_moves) - len(self.moves)} invalid moves")
                
                return len(self.moves) > 0
            else:
                print(f"No moves found for game {game_number}")
                return False
            
        except Exception as e:
            print(f"Error loading game data: {e}")
            return False
    
    def update(self):
        """Update game state for each frame."""
        if self.paused:
            return
            
        current_time = time.time()
        
        # Check if it's time for the next move
        if current_time - self.last_move_time >= self.move_pause and self.move_index < len(self.moves):
            # Make the next move
            next_move = self.moves[self.move_index]
            self.move_index += 1
            
            # Update game state
            game_continues, apple_eaten = self.make_move(next_move)
            
            # Update last move time
            self.last_move_time = current_time
            
            # Check if game is over
            if not game_continues:
                print(f"Game {self.game_number} over. Score: {self.score}, Steps: {self.steps}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(1000)  # 1 second pause
                    self.game_number += 1
                    self.load_game_data(self.game_number)
            
            # Check if we're out of moves
            elif self.move_index >= len(self.moves):
                print(f"Replay complete for game {self.game_number}. Score: {self.score}, Steps: {self.steps}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(1000)  # 1 second pause
                    self.game_number += 1
                    self.load_game_data(self.game_number)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                    print("Replay " + ("paused" if self.paused else "resumed"))
                elif event.key == K_n:
                    # Next game
                    self.game_number += 1
                    if not self.load_game_data(self.game_number):
                        print(f"Could not load game {self.game_number}. Staying with current game.")
                        self.game_number -= 1
                elif event.key == K_r:
                    # Restart current game
                    self.load_game_data(self.game_number)
    
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
            clock.tick(60)
        
        # Clean up
        pygame.quit() 