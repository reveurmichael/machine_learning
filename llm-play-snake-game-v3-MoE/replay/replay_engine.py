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
from config import TIME_DELAY, TIME_TICK, DIRECTIONS
import numpy as np
import traceback
from utils.game_manager_utils import check_collision

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
        self.llm_response = None
        self.planned_moves = []
    
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
            
    def load_game_data(self, game_number):
        """Load game data for a specific game number.
        
        Args:
            game_number: The game number to load
            
        Returns:
            Game data dictionary or None if loading failed
        """
        # Build the path to the game data file
        game_file = os.path.join(self.log_dir, f"game_{game_number}.json")
        
        # Check if the file exists
        if not os.path.exists(game_file):
            print(f"Game {game_number} data not found")
            return None
        
        try:
            print(f"Loading game data from {game_file}")
            with open(game_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # Get basic game information
            self.score = game_data.get('score', 0)
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
            self.steps = 0
            
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
            
            print(f"Game {game_number}: Score: {self.score}, Steps: {len(self.moves)}, End reason: {self.game_end_reason}, LLM: {self.primary_llm}")
            
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
    
    def update(self):
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
                    
    def load_next_game(self):
        """Load the next game in sequence."""
        self.game_number += 1
        if not self.load_game_data(self.game_number):
            print("No more games to load. Replay complete.")
            self.running = False
    
    def execute_replay_move(self, direction_key):
        """Execute a move in the specified direction for replay.
        
        Args:
            direction_key: String key of the direction to move in
            
        Returns:
            Boolean indicating if the game is still active
        """
        # Get direction vector
        if direction_key not in DIRECTIONS:
            print(f"Invalid direction: {direction_key}, using RIGHT")
            direction_key = "RIGHT"
        
        direction = DIRECTIONS[direction_key]
        self.current_direction = direction
        
        # Calculate new head position
        head_x, head_y = self.head_position
        new_head = np.array([head_x + direction[0], head_y + direction[1]])
        
        # Debug information
        print(f"Moving {direction_key}: Head from ({head_x}, {head_y}) to ({new_head[0]}, {new_head[1]})")
        
        # Check if the new head position will eat an apple
        is_eating_apple = np.array_equal(new_head, self.apple_position)
        
        # Use the shared collision detection function from utils.game_manager_utils
        wall_collision, body_collision = check_collision(new_head, self.snake_positions, self.grid_size, is_eating_apple)
        
        if wall_collision:
            print(f"Game over: Snake hit wall at position {new_head}")
            return False
        
        if body_collision:
            print(f"Game over: Snake hit itself at position {new_head}")
            return False
        
        # Prepare new snake positions
        new_snake_positions = np.copy(self.snake_positions)
        new_snake_positions = np.vstack((new_snake_positions, new_head))
        
        # Check for apple
        apple_eaten = is_eating_apple
        
        if not apple_eaten:
            # Remove tail if no apple eaten
            new_snake_positions = new_snake_positions[1:]
        else:
            # Apple eaten - update score and apple position
            self.score += 1
            print(f"Apple eaten! Score: {self.score}")
            
            # Move to next apple position if available
            if self.apple_index + 1 < len(self.apple_positions):
                self.apple_index += 1
                next_apple = self.apple_positions[self.apple_index]
                
                if isinstance(next_apple, dict) and 'x' in next_apple and 'y' in next_apple:
                    # Set apple position
                    success = self.set_apple_position([next_apple['x'], next_apple['y']])
                    if not success:
                        # Use predetermined alternative position
                        alt_pos = self._place_apple_away_from_snake()
                        self.apple_position = alt_pos
                elif isinstance(next_apple, (list, np.ndarray)) and len(next_apple) == 2:
                    # Set apple position
                    success = self.set_apple_position(next_apple)
                    if not success:
                        # Use predetermined alternative position
                        alt_pos = self._place_apple_away_from_snake()
                        self.apple_position = alt_pos
                else:
                    # Use predetermined position
                    alt_pos = self._place_apple_away_from_snake()
                    self.apple_position = alt_pos
            else:
                # No more predefined apple positions
                alt_pos = self._place_apple_away_from_snake()
                self.apple_position = alt_pos
        
        # Update snake state
        self.snake_positions = new_snake_positions
        self.head_position = self.snake_positions[-1]
        
        # Update game board
        self._update_board()
        
        # Update step counter
        self.steps += 1
        
        return True
    
    def handle_events(self):
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
    
    def set_apple_position(self, position):
        """Set the apple position, avoiding snake body.
        
        Args:
            position: The desired position as [x, y]
            
        Returns:
            Boolean indicating success
        """
        if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 2:
            return False
            
        # Convert to numpy array if needed
        if not isinstance(position, np.ndarray):
            position = np.array(position)
            
        # Check that position is within bounds
        if (position[0] < 0 or position[0] >= self.grid_size or 
            position[1] < 0 or position[1] >= self.grid_size):
            return False
        
        # Check if position is on snake
        for segment in self.snake_positions:
            if np.array_equal(position, segment):
                # Position is invalid
                return False
                
        # Position is valid
        self.apple_position = position
        return True
    
    def _place_apple_away_from_snake(self):
        """Place the apple at a fixed distance from the snake's head.
        
        Returns:
            Array [x, y] with apple position
        """
        # Place apple at a fixed offset from head position
        return np.array([
            (self.head_position[0] + 5) % self.grid_size,
            (self.head_position[1] + 5) % self.grid_size
        ])

    @classmethod
    def find_log_directories(cls, root_dir="logs", max_depth=4):
        """Find valid log directories for replay.
        
        Args:
            root_dir: Root directory to start search from
            max_depth: Maximum directory depth to search
            
        Returns:
            List of valid log directory paths
        """
        from replay.replay_utils import find_valid_log_folders
        return find_valid_log_folders(root_dir, max_depth)
        
    @classmethod
    def is_valid_log_directory(cls, directory):
        """Check if a directory is a valid log directory.
        
        A valid log directory must have:
        - A summary.json file
        - At least one game_*.json file
        - A prompts directory
        - A responses directory
        
        Args:
            directory: Directory path to check
            
        Returns:
            Boolean indicating if the directory is a valid log directory
        """
        if not os.path.isdir(directory):
            return False
            
        # Check for required files and directories
        has_summary = os.path.exists(os.path.join(directory, "summary.json"))
        
        # Check for at least one game_*.json file
        has_game_files = False
        for filename in os.listdir(directory):
            if filename.startswith("game_") and filename.endswith(".json"):
                has_game_files = True
                break
                
        # Check for required directories
        has_prompts_dir = os.path.isdir(os.path.join(directory, "prompts"))
        has_responses_dir = os.path.isdir(os.path.join(directory, "responses"))
        
        return has_summary and has_game_files and has_prompts_dir and has_responses_dir 