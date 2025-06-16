"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.
"""

import os
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pygame
from pygame.locals import *  # noqa: F403 – Pygame constants
import numpy as np

from core.game_controller import GameController
from config import TIME_DELAY, TIME_TICK
from utils.file_utils import get_game_json_filename, join_log_path

class ReplayEngine(GameController):
    """Engine for replaying recorded Snake games.

    This class consumes the *game_N.json* artefacts produced by the main
    Snake-LLM run and replays them either with a PyGame GUI or in headless
    mode, faithfully reproducing the original sequence of moves, apple
    spawns, timing, and statistics.
    """
    
    # -------------------------------
    # Construction / initialisation
    # -------------------------------

    def __init__(
        self,
        log_dir: str,
        move_pause: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:
        """Create a `ReplayEngine`.

        Parameters
        ----------
        log_dir
            Directory that contains *summary.json* and *game_*.json files.
        move_pause
            Seconds to pause between two moves (≃ playback speed).
        auto_advance
            When *True*, automatically jump to the next game once the current
            replay finishes.
        use_gui
            Whether to use the graphical interface (`pygame`).  Setting it to
            *False* keeps all computations but skips rendering – useful for
            batch validations.
        """
        super().__init__(use_gui=use_gui)
        
        # Initialize replay parameters
        self.log_dir: str = log_dir
        self.pause_between_moves: float = move_pause
        self.auto_advance: bool = auto_advance
        
        # Game state specific to replay
        self.game_number: int = 1
        self.apple_positions: List[List[int]] = []
        self.apple_index: int = 0
        self.moves: List[str] = []  # usually str, but keep Any for safety
        self.move_index: int = 0
        self.moves_made: List[str] = []
        self.game_stats: Dict[str, Any] = {}
        self.last_move_time: float = time.time()
        self.running: bool = True
        self.paused: bool = False
        
        # Game statistics from the log file
        self.game_end_reason: Optional[str] = None
        self.primary_llm: Optional[str] = None
        self.secondary_llm: Optional[str] = None
        self.game_timestamp: Optional[str] = None
        self.llm_response: Optional[str] = None
        self.planned_moves: List[str] = []
    
    def set_gui(self, gui_instance: Any) -> None:
        """Set the GUI instance to use for display.
        
        Args:
            gui_instance: Instance of a GUI class for replay
        """
        super().set_gui(gui_instance)
        # Sync the GUI paused state with the replay engine
        if hasattr(gui_instance, 'set_paused'):
            gui_instance.set_paused(self.paused)
    
    def draw(self) -> None:
        """Draw the current game state."""
        if self.use_gui and self.gui:
            # Create replay data dictionary
            replay_data = {
                'snake_positions': self.snake_positions,
                'apple_position': self.apple_position,
                'game_number': self.game_number,
                'score': self.score,
                'steps': self.steps,
                'move_index': self.move_index,
                'total_moves': len(self.moves),
                'planned_moves': self.planned_moves,
                'llm_response': self.llm_response,
                'primary_llm': self.primary_llm,
                'secondary_llm': self.secondary_llm,
                'paused': self.paused,
                'speed': 1.0 / self.pause_between_moves if self.pause_between_moves > 0 else 1.0,
                'timestamp': self.game_timestamp,
                'game_end_reason': self.game_end_reason
            }
            
            # Draw the replay view
            self.gui.draw(replay_data)
            
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """Load game data for a specific game number.
        
        Args:
            game_number: The game number to load
            
        Returns:
            Game data dictionary or None if loading failed
        """
        # Build the path to the game data file using the utility functions
        game_filename = get_game_json_filename(game_number)
        game_file = join_log_path(self.log_dir, game_filename)
        
        # Check if the file exists
        if not os.path.exists(game_file):
            print(f"Game {game_number} data not found")
            return None
        
        try:
            print(f"Loading game data from {game_file}")
            with open(game_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # Get basic game information
            loaded_score = game_data.get('score', 0)
            self.game_end_reason = game_data.get('game_end_reason', None)
            
            # Extract game data from detailed_history
            if 'detailed_history' not in game_data:
                print("Error: No detailed_history in game data")
                return None
                
            detailed_history = game_data['detailed_history']
            
            # ----- Simplified Data Loading Strategy -----
            # Instead of complex loading and fallback mechanisms, we'll use a simple strategy:
            # 1. Always use apple_positions from detailed_history for apples
            # 2. Always use moves from detailed_history for moves
            # This respects the fixed schema while being simple and reliable
            
            # Get apple positions - always use the top-level array
            self.apple_positions = []
            raw_apple_positions = detailed_history.get('apple_positions', [])
            
            for pos in raw_apple_positions:
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    self.apple_positions.append([pos['x'], pos['y']])
                elif isinstance(pos, (list, np.ndarray)) and len(pos) == 2:
                    self.apple_positions.append(pos)
            
            # Get moves - always use the top-level array
            self.moves = detailed_history.get('moves', [])
            
            # Simple validation check
            if not self.moves:
                print("Error: No moves found in game data")
                return None
                
            if not self.apple_positions:
                print("Warning: No apple positions found in game data")
            
            # Reset game state indices
            self.move_index = 0
            self.apple_index = 0
            self.moves_made = []
            
            # Get round information from metadata
            round_count = game_data.get('metadata', {}).get('round_count', 0)
            print(f"Game has {round_count} rounds")
            
            # Get LLM information
            if 'llm_info' in game_data:
                llm_info = game_data['llm_info']
                self.primary_llm = f"{llm_info.get('primary_provider', 'Unknown')}/{llm_info.get('primary_model', 'Unknown')}"
                
                if llm_info.get('parser_provider') and llm_info.get('parser_provider').lower() != 'none':
                    self.secondary_llm = f"{llm_info.get('parser_provider', 'None')}/{llm_info.get('parser_model', 'None')}"
                else:
                    self.secondary_llm = "None/None"
            else:
                self.primary_llm = "Unknown/Unknown"
                self.secondary_llm = "None/None"
            
            # Get timestamp
            self.game_timestamp = game_data.get('metadata', {}).get('timestamp', "Unknown")
            
            # Store game data
            self.game_stats = game_data
            
            print(f"Game {game_number}: Score: {loaded_score}, Steps: {len(self.moves)}, End reason: {self.game_end_reason}, LLM: {self.primary_llm}")
            
            # Initialize game state
            print("Initializing game state...")
            self.reset()
            
            # Set initial snake position (middle of grid)
            self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
            self.head_position = self.snake_positions[-1]
            
            # Set initial apple position
            if self.apple_positions:
                first_apple = self.apple_positions[0]
                
                if isinstance(first_apple, (list, np.ndarray)) and len(first_apple) == 2:
                    # Set initial apple position
                    success = self.set_apple_position(first_apple)
                    if not success:
                        # Use default position if invalid
                        self.apple_position = np.array([self.grid_size // 2, self.grid_size // 2])
                else:
                    # Default position
                    self.apple_position = np.array([self.grid_size // 2, self.grid_size // 2])
                    
                print(f"Set initial apple position: {self.apple_position}")
            
            # Update game board
            self._update_board()
            
            # Reset GUI move history if available
            if self.use_gui and self.gui and hasattr(self.gui, 'move_history'):
                self.gui.move_history = []
            
            # Get LLM response if available
            self.llm_response = detailed_history.get('llm_response', "No LLM response data available for this game.")
            
            # Get planned moves
            # If rounds_data is available, try to extract planned moves from the first round
            self.planned_moves = []
            if 'rounds_data' in detailed_history and detailed_history['rounds_data']:
                try:
                    # Get the first round's data
                    first_round_key = sorted(detailed_history['rounds_data'].keys(), 
                                           key=lambda k: int(k.split('_')[1]))[0]
                    first_round = detailed_history['rounds_data'][first_round_key]
                    
                    # Extract planned moves if available
                    if 'moves' in first_round and isinstance(first_round['moves'], list) and len(first_round['moves']) > 1:
                        # The first move is already used, so get the rest as planned moves
                        self.planned_moves = first_round['moves'][1:] if len(first_round['moves']) > 1 else []
                except Exception:
                    # If anything goes wrong, just leave planned_moves empty
                    pass
            
            print(f"Game {game_number} loaded successfully")
            return game_data
            
        except Exception as e:
            print(f"Error loading game data: {e}")
            traceback.print_exc()
            return None
    
    def update(self) -> None:
        """Update game state for each frame."""
        if self.paused:
            return
            
        current_time = time.time()
        
        # Check if it's time for the next move
        if current_time - self.last_move_time >= self.pause_between_moves and self.move_index < len(self.moves):
            try:
                # Get next move
                next_move = self.moves[self.move_index]
                print(f"Move {self.move_index+1}/{len(self.moves)}: {next_move}")
                
                # Update move tracking
                self.move_index += 1
                self.moves_made.append(next_move)
                
                # Update planned moves display
                if self.planned_moves and len(self.planned_moves) > 0:
                    self.planned_moves = self.planned_moves[1:] if len(self.planned_moves) > 1 else []
                
                # Execute the move
                game_continues = self.execute_replay_move(next_move)
                
                # Update last move time
                self.last_move_time = current_time
                
                # Handle game completion
                if not game_continues:
                    print(f"Game {self.game_number} over. Score: {self.score}, Steps: {self.steps}, End reason: {self.game_end_reason}")
                    
                    # Set move_index to the end to prevent further moves
                    self.move_index = len(self.moves)
                    
                    # Advance to next game if auto-advance is enabled
                    if self.auto_advance:
                        pygame.time.delay(1000)  # Pause before next game
                        self.load_next_game()
                
                # Check if we've finished all moves
                elif self.move_index >= len(self.moves):
                    print(f"Replay complete for game {self.game_number}. Score: {self.score}, Steps: {self.steps}")
                    
                    # Advance to next game if auto-advance is enabled
                    if self.auto_advance:
                        pygame.time.delay(1000)  # Pause before next game
                        self.load_next_game()
                
                # Update the display
                if self.use_gui and self.gui:
                    self.draw()
                    
            except Exception as e:
                print(f"Error during replay: {e}")
                traceback.print_exc()
                
                # Try to continue with next game if auto-advance is enabled
                if self.auto_advance:
                    self.load_next_game()
                    
    def load_next_game(self) -> None:
        """Load the next game in sequence."""
        self.game_number += 1
        if not self.load_game_data(self.game_number):
            print("No more games to load. Replay complete.")
            self.running = False
    
    def execute_replay_move(self, direction_key: str) -> bool:
        """Execute a move in the specified direction for replay.
        
        Args:
            direction_key: String key of the direction to move in
            
        Returns:
            Boolean indicating if the game is still active
        """
        # Standardize direction key to uppercase to handle case insensitivity
        if isinstance(direction_key, str):
            direction_key = direction_key.upper()
            
        # -------------------------------
        # Sentinel moves that represent a time-tick without actual movement
        # (e.g. blocked reversals or intentionally empty moves).  We simply
        # advance the replay pointer and keep the game alive without calling
        # make_move(), so the snake stays in place exactly as it did in the
        # original run.
        # -------------------------------
        if direction_key in ("INVALID_REVERSAL", "EMPTY", "SOMETHING_IS_WRONG"):
            # Mirror step accounting from the original run so that stats align.
            if direction_key == "INVALID_REVERSAL":
                # Use current direction context where possible
                self.game_state.record_invalid_reversal(direction_key, self._get_current_direction_key())
            elif direction_key == "EMPTY":
                self.game_state.record_empty_move()
            elif direction_key == "SOMETHING_IS_WRONG":
                self.game_state.record_something_is_wrong_move()
            return True  # Game continues, snake doesn't move
        
        # Use the parent class's make_move method to ensure consistent behavior
        # This will handle direction validation, reversal prevention, and game state updates
        game_active, apple_eaten = super().make_move(direction_key)
        
        # If apple was eaten, we need to manually advance to the next apple position from our log
        # since replay uses predefined apple positions from the game history
        if apple_eaten and self.apple_index + 1 < len(self.apple_positions):
            self.apple_index += 1
            next_apple = self.apple_positions[self.apple_index]
            
            if isinstance(next_apple, dict) and 'x' in next_apple and 'y' in next_apple:
                # Set apple position from dictionary format
                self.set_apple_position([next_apple['x'], next_apple['y']])
            elif isinstance(next_apple, (list, np.ndarray)) and len(next_apple) == 2:
                # Set apple position from array format
                self.set_apple_position(next_apple)
        
        return game_active
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        redraw_needed = False
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # Handle key presses
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle pause
                    self.paused = not self.paused
                    if self.gui and hasattr(self.gui, 'set_paused'):
                        self.gui.set_paused(self.paused)
                    print(f"Replay {'paused' if self.paused else 'resumed'}")
                    redraw_needed = True
                elif event.key in (pygame.K_UP, pygame.K_s):
                    # Speed up
                    self.pause_between_moves = max(0.1, self.pause_between_moves * 0.75)
                    print(f"Speed increased: {1/self.pause_between_moves:.1f}x")
                    redraw_needed = True
                elif event.key in (pygame.K_DOWN, pygame.K_d):
                    # Slow down
                    self.pause_between_moves = min(2.0, self.pause_between_moves * 1.25)
                    print(f"Speed decreased: {1/self.pause_between_moves:.1f}x")
                    redraw_needed = True
                elif event.key == pygame.K_r:
                    # Restart current game
                    self.load_game_data(self.game_number)
                    print(f"Restarting game {self.game_number}")
                    redraw_needed = True
                elif event.key in (pygame.K_RIGHT, pygame.K_n):
                    # Next game
                    self.game_number += 1
                    if not self.load_game_data(self.game_number):
                        print("No more games to load. Staying on current game.")
                        self.game_number -= 1
                    redraw_needed = True
                elif event.key in (pygame.K_LEFT, pygame.K_p):
                    # Previous game
                    if self.game_number > 1:
                        self.game_number -= 1
                        self.load_game_data(self.game_number)
                        print(f"Going to previous game {self.game_number}")
                    else:
                        print("Already at the first game")
                    redraw_needed = True
        
        # Redraw the UI if needed after processing events
        if redraw_needed and self.use_gui and self.gui:
            self.draw()
    
    def run(self) -> None:
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
  