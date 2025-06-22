"""
Snake game implementation with LLM integration.
Extends the base game controller with LLM-specific functionality.

The class BaseGameLogic is NOT Task0 specific.
The class GameLogic is Task0 specific.
"""

import traceback
import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray

from config.ui_constants import GRID_SIZE
from config.game_constants import DIRECTIONS
from core.game_data import BaseGameData, GameData
from utils.collision_utils import check_collision
from utils.moves_utils import normalize_direction, is_reverse
from utils.board_utils import generate_random_apple, update_board_array

# Avoid circular imports
if TYPE_CHECKING:
    from gui.base_gui import BaseGUI

# ------------------
# BaseGameLogic – generic game logic independent of controllers
# ------------------

# This class is NOT Task0 specific.
class BaseGameLogic:
    """Generic game logic class that handles core Snake game mechanics.

    This class provides the fundamental game logic without coupling to
    specific controller patterns. It can be used by different controller
    types (CLI, Web, GUI) through composition.
    
    Design Pattern: Strategy Pattern
    - Game logic is separated from UI concerns
    - Different controllers can use the same game logic
    - Easier to test and maintain
    """

    # Subclasses may override to inject their specialised data container.
    GAME_DATA_CLS = BaseGameData

    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        """Initialize the game logic.
        
        Args:
            grid_size: Size of the game grid
            use_gui: Whether to use GUI for display
        """

        # Game state variables
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]

        # Game state tracker for statistics
        self.game_state = self.GAME_DATA_CLS()
        self.game_state.reset()

        # Generate the very first apple
        self.apple_position = self._generate_apple()

        # Runtime trackers
        self.current_direction = None
        self.last_collision_type = None

        # Track the apple history starting with the first apple
        self.apple_positions_history = [self.apple_position.copy()]

        # Board entity codes
        self.board_info = {
            "empty": 0,
            "snake": 1,
            "apple": 2
        }

        # GUI settings
        self.use_gui = use_gui
        self.gui = None

        # Initialize the board
        self._update_board()

        # Sync initial snake body into GameData so snake_length starts correct
        self.game_state.snake_positions = self.snake_positions.tolist()

        # ----- Generic planning state (shared by all agent types) -----
        self.planned_moves: List[str] = []

    def set_gui(self, gui_instance: "BaseGUI") -> None:
        """Attach a GUI implementation (pygame, web-proxy, etc.)."""
        self.gui = gui_instance
        self.use_gui = gui_instance is not None

    def reset(self) -> None:
        """Reset the game to the initial state."""
        # Reset game state
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]

        # Reset game state tracker
        self.game_state.reset()

        # Generate the initial apple AFTER the game state has been reset
        self.apple_position = self._generate_apple()

        # Update the board
        self._update_board()

        # Draw if GUI is available
        if self.use_gui and self.gui:
            self.draw()

        # Clear runtime direction/collision trackers for the new game
        self.current_direction = None
        self.last_collision_type = None

        # Reset apple history and seed with the initial apple
        self.apple_positions_history = [self.apple_position.copy()]

        # Sync initial snake body into GameData so snake_length starts correct
        self.game_state.snake_positions = self.snake_positions.tolist()
        
        # Clear planned moves
        self.planned_moves = []

    def draw(self) -> None:
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            # Specific drawing handled by the GUI implementation
            pass

    def _update_board(self) -> None:
        """Update the game board with current snake and apple positions."""
        update_board_array(
            self.board,
            self.snake_positions,
            self.apple_position,
            self.board_info,
        )

    def _generate_apple(self) -> NDArray[np.int_]:
        """Generate a new apple at a random empty position."""
        position = generate_random_apple(self.snake_positions, self.grid_size)
        self.game_state.record_apple_position(position)
        return position

    def make_move(self, direction_key: str) -> Tuple[bool, bool]:
        """Execute a move in the specified direction."""
        # Standardize direction key to uppercase
        if isinstance(direction_key, str):
            direction_key = direction_key.upper()

        # Get direction vector
        if direction_key not in DIRECTIONS:
            print(f"Error: Invalid direction: {direction_key}")
            return False, False

        direction = DIRECTIONS[direction_key]

        # Don't allow reversing direction directly
        if (
            self.current_direction is not None and
            is_reverse(direction_key, self._get_current_direction_key())
        ):
            print(f"Tried to reverse direction: {direction_key}. No move will be made.")
            self.game_state.record_invalid_reversal()
            return True, False

        # Update current direction
        self.current_direction = direction

        # Calculate new head position
        head_x, head_y = self.head_position
        new_head = np.array([
            head_x + direction[0],
            head_y + direction[1]
        ])

        # Check if the new head position is where the apple is
        is_eating_apple_at_new_head = np.array_equal(new_head, self.apple_position)

        # Check for collisions
        wall_collision, body_collision = self._check_collision(new_head, is_eating_apple_at_new_head)

        if wall_collision:
            print(f"Game over! Snake hit wall moving {direction_key}")
            self.last_collision_type = 'WALL'
            self.game_state.record_move(direction_key, apple_eaten=False)
            self.game_state.record_game_end("WALL")
            self._on_game_over("WALL")
            return False, False

        if body_collision:
            print(f"Game over! Snake hit itself moving {direction_key}")
            self.last_collision_type = 'SELF'
            self.game_state.record_move(direction_key, apple_eaten=False)
            self.game_state.record_game_end("SELF")
            self._on_game_over("SELF")
            return False, False

        # Check if the snake eats an apple
        apple_eaten = is_eating_apple_at_new_head

        # No collision, proceed with move
        new_snake_positions = np.copy(self.snake_positions)
        new_snake_positions = np.vstack((new_snake_positions, new_head))

        if not apple_eaten:
            # Remove tail if no apple eaten
            new_snake_positions = new_snake_positions[1:]
        else:
            # Generate new apple and add to history when an apple is eaten
            self.apple_position = self._generate_apple()
            self.apple_positions_history.append(self.apple_position.copy())

        # Update snake positions and head
        self.snake_positions = new_snake_positions
        self.head_position = self.snake_positions[-1]

        # Update the board
        self._update_board()

        # Record move in game state
        self.game_state.record_move(direction_key, apple_eaten)

        # Display message if apple was eaten
        if apple_eaten:
            apples_emoji = "🍎" * self.score
            print(f"🚀 Apple eaten! Score: {self.score} {apples_emoji}")

        # Draw if GUI is available
        if self.use_gui and self.gui:
            self.draw()

        # Keep the GameData replica in sync
        self.game_state.snake_positions = self.snake_positions.tolist()

        # Allow subclasses to post-process the step
        self._post_move(apple_eaten)

        return True, apple_eaten

    def _check_collision(
        self, head_position: NDArray[np.int_], is_eating_apple_flag: bool
    ) -> Tuple[bool, bool]:
        """Check for wall or body collision."""
        # Pass only the body segments (excluding the current head which is the last element)
        snake_body_without_head = self.snake_positions[:-1] if len(self.snake_positions) > 1 else []
        return check_collision(
            head_position=head_position,
            snake_body=snake_body_without_head,
            grid_size=self.grid_size,
            is_apple_eaten=is_eating_apple_flag,
        )

    def _get_current_direction_key(self) -> str:
        """Return the current direction as a string key (e.g., 'UP')."""
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
        return "RIGHT"  # Fallback

    def get_current_direction_key(self) -> str:
        """Public accessor for the snake's current direction key."""
        return self._get_current_direction_key()

    @property
    def score(self) -> int:
        """Get the current score from the game state."""
        return self.game_state.score

    @property
    def steps(self) -> int:
        """Get the current steps from the game state."""
        return self.game_state.steps

    @property
    def snake_length(self) -> int:
        """Current snake length."""
        return self.game_state.snake_length

    # ------------------
    # Generic helpers for planned-move agents
    # ------------------

    def get_next_planned_move(self):
        """Pop and return the next move from the current *planned_moves* list."""
        if self.planned_moves:
            return self.planned_moves.pop(0)
        return None

    # ------------------
    # Extension hooks (NOP by default) – subclasses override as needed.
    # ------------------

    def _post_move(self, apple_eaten: bool) -> None:
        """Hook called after every successful move."""
        return None

    def _on_game_over(self, reason: str) -> None:
        """Hook invoked right before returning from a terminal collision."""
        return None

    def get_state_snapshot(self):
        """Return a plain-Python snapshot of the current board."""
        return {
            "board": [row.copy() for row in self.board],
            "direction": self._get_current_direction_key() or "NONE",
            "apple": tuple(self.apple_position),
            "score": self.score,
            "steps": self.steps,
        }

# ------------------
# Task-0 concrete implementation – plugs in GameData for LLM metrics.
# ---------------------

# This class is Task0 specific.
class GameLogic(BaseGameLogic):
    """Snake game with LLM agent integration."""
    
    GAME_DATA_CLS = GameData  # type: ignore  # injected into BaseGameController
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True):
        """Initialize the snake game.
        
        Args:
            grid_size: Size of the game grid (default from config)
            use_gui: Whether to use GUI for display
        """
        super().__init__(grid_size, use_gui)
        
        # LLM-specific state
        self.processed_response = ""
    
    @property
    def head(self) -> Tuple[int, int]:
        """Get the current head position.
        
        Returns:
            Tuple of (x, y) coordinates of the snake's head
        """
        return tuple(self.head_position)
    
    @property
    def apple(self) -> Tuple[int, int]:
        """Get the current apple position.
        
        Returns:
            Tuple of (x, y) coordinates of the apple
        """
        return tuple(self.apple_position)
    
    @property
    def body(self) -> List[Tuple[int, int]]:
        """Get the snake body positions (excluding head).
        
        Returns:
            List of (x, y) tuples for body segments
        """
        # Convert each position to a tuple and exclude the head (last element)
        return [tuple(pos) for pos in self.snake_positions[:-1]][::-1]
    
    def draw(self) -> None:
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            self.gui.draw_board(self.board, self.board_info, self.head_position)
            
            # Create game info dictionary
            game_info = {
                'score': self.score,
                'steps': self.steps,
                'planned_moves': self.planned_moves,
                'llm_response': self.processed_response
            }
            
            self.gui.draw_game_info(game_info)
    
    def reset(self):  # type: ignore[override]
        """Reset the game and clear Task-0-specific fields."""
        super().reset()
        self.processed_response = ""
        return self.get_state_representation()
    
    def get_state_representation(self) -> str:
        """Generate a representation of the game state for the LLM prompt.
        
        Returns:
            String representation of the game state using the template from config.py
        """
        # Get current direction as a string
        current_direction = self._get_current_direction_key() if self.current_direction is not None else "NONE"
        
        # Import locally to avoid module-level LLM dependency
        from llm.prompt_utils import prepare_snake_prompt  # local import

        return prepare_snake_prompt(
            head_position=self.head_position,
            body_positions=self.body,
            apple_position=self.apple_position,
            current_direction=current_direction
        )
    
    def parse_llm_response(self, response: str):
        """Parse the LLM's response to extract multiple sequential moves.
        
        Args:
            response: Text response from the LLM in JSON format
            
        Returns:
            The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
            or None if no valid moves were found
        """
        try:
            # Import locally to avoid module-level LLM dependency
            from llm.parsing_utils import parse_llm_response  # local import
            from utils.text_utils import process_response_for_display  # local import

            return parse_llm_response(response, process_response_for_display, self)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            traceback.print_exc()
            
            # Store the raw response for display (first 200 chars)
            self.processed_response = (
                f"ERROR: Failed to parse LLM response\n\n{response[:200]}..."
            )
            
            # Clear previous planned moves
            self.planned_moves = []
            
            # Update game state to record error
            self.game_state.record_something_is_wrong_move()
            
            return None
    