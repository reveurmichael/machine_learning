"""
Replay engine for the Snake game.
Handles replaying of previously recorded games.
"""

import time
import traceback
from typing import Any, Dict, List, Optional

import pygame
from pygame.locals import *  # noqa: F403 – Pygame constants
import numpy as np

from core.game_controller import GameController
from config import TIME_DELAY, TIME_TICK
from replay.replay_utils import load_game_json, parse_game_data

class ReplayEngine(GameController):
    """Engine for replaying recorded Snake games.

    This class consumes the *game_N.json* artefacts produced by the main
    Snake-LLM run and replays them either with a PyGame GUI or in headless
    mode, faithfully reproducing the original sequence of moves, apple
    spawns, timing, and statistics.
    """

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
        self.moves: List[str] = []  
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
        """
        1. Load game data for a specific game number.
        2. Extract apple_positions, moves, LLM info, timestamp, etc.
        3. Reset a bunch of engine-specific indices.
        4. Validate the presence of detailed_history, fallbacks, prints, etc.

        Args:
            game_number: The game number to load

        Returns:
            Game data dictionary or None if loading failed
        """
        # Retrieve JSON dict with helper (keeps I/O concerns out of this file)
        game_file, game_data = load_game_json(self.log_dir, game_number)

        if game_data is None:
            return None

        try:
            print(f"Loading game data from {game_file}")
            parsed = parse_game_data(game_data)
            if parsed is None:
                return None

            # Unpack parsed fields 
            self.apple_positions = parsed["apple_positions"]
            self.moves = parsed["moves"]
            self.planned_moves = parsed["planned_moves"]
            self.game_end_reason = parsed["game_end_reason"]
            self.primary_llm = parsed["primary_llm"]
            self.secondary_llm = parsed["secondary_llm"]
            self.game_timestamp = parsed["timestamp"]

            loaded_score = game_data.get('score', 0)

            # Reset counters 
            self.move_index = 0
            self.apple_index = 0
            self.moves_made = []

            # Store raw game dict for reference
            self.game_stats = parsed["raw"]

            print(f"Game {game_number}: Score: {loaded_score}, Steps: {len(self.moves)}, End reason: {self.game_end_reason}, LLM: {self.primary_llm}")

            # Initialize game state
            print("Initializing game state...")
            self.reset()

            # Set initial snake position (middle of grid)
            self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
            self.head_position = self.snake_positions[-1]

            # Set initial apple position
            self.set_apple_position(self.apple_positions[0])

            # Update game board
            self._update_board()

            # Reset GUI move history if available
            if self.use_gui and self.gui and hasattr(self.gui, 'move_history'):
                self.gui.move_history = []

            # Get LLM response if available
            self.llm_response = parsed.get('llm_response', "No LLM response data available for this game.")

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

                # Update the display
                if self.use_gui and self.gui:
                    self.draw()

            except Exception as e:
                print(f"Error during replay: {e}")
                traceback.print_exc()

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
                self.game_state.record_invalid_reversal()
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
