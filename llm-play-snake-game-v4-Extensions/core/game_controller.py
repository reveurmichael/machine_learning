"""
Game controller for the Snake game.
Provides core game logic that can run with or without a GUI.

The class BaseGameController is NOT Task0 specific.
The class GameController is Task0 specific.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from config.ui_constants import GRID_SIZE
from config.game_constants import DIRECTIONS
from core.game_data import BaseGameData, GameData
from utils.collision_utils import check_collision
from utils.moves_utils import normalize_direction, is_reverse
from utils.board_utils import generate_random_apple, update_board_array
from colorama import Fore

# ---------------------
# Typing helpers – avoid heavy GUI imports at runtime
# ---------------------

if TYPE_CHECKING:
    from gui.base_gui import BaseGUI

# ---------------------
# Generic controller – agnostic to the concrete GameData subclass.
# ---------------------

class BaseGameController:
    """Base class for the Snake game controller."""

    # Subclasses may override to inject their specialised data container.
    GAME_DATA_CLS = BaseGameData

    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        """Initialize the game controller.
        
        Args:
            grid_size: Size of the game grid
            use_gui: Whether to use GUI for display
        """

        # Game state variables
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]

        # Game state tracker for statistics – deferred to class attribute so
        # specialisations (LLM, RL, etc.) can plug in their own rich data
        # container without having to re-implement the whole constructor.
        self.game_state = self.GAME_DATA_CLS()

        # Ensure *reset()* on the freshly created tracker so round_manager and
        # other internal structures are ready before the first apple is
        # generated.
        self.game_state.reset()

        # Generate the very first apple now that the game state is ready.  This
        # ensures that `self.apple_position` exists before the first render.
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

    def set_gui(self, gui_instance: "BaseGUI") -> None:
        """Attach a GUI implementation (pygame, web-proxy, etc.).

        The controller itself remains *UI-agnostic* – all drawing is
        delegated to the injected object which must expose the expected
        ``draw_*`` methods.
        
        Args:
            gui_instance: Any object implementing the game-GUI interface.
        """
        self.gui = gui_instance
        self.use_gui = gui_instance is not None

    def reset(self) -> None:
        """Reset the game to the initial state."""
        # Reset game state
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]

        # Reset game state tracker
        self.game_state.reset()

        # Generate the initial apple AFTER the game state has been reset so that
        # record_apple_position() works correctly (round_buffer is not None).
        self.apple_position = self._generate_apple()

        # Note: _generate_apple() already records the apple position in the
        # game_state, so we avoid double-recording here.

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

    def filter_invalid_reversals(
        self,
        moves: list[str],
        current_direction: str | None = None,
    ) -> list[str]:
        """Filter out invalid reversal moves from a sequence.
        
        Args:
            moves: List of move directions
            current_direction: Current direction of the snake (defaults to self.current_direction if None)
            
        Returns:
            Filtered list of moves with invalid reversals removed
        """
        if not moves or len(moves) <= 1:
            return moves

        filtered_moves: list[str] = []
        last_direction = current_direction or self._get_current_direction_key() or moves[0]

        for move in moves:
            move = normalize_direction(move)

            if is_reverse(move, last_direction):
                print(f"Filtering out invalid reversal move: {move} after {last_direction}")
                # Record invalid reversal in game state when available
                if hasattr(self, "game_state"):
                    self.game_state.record_invalid_reversal()
                # Skip this move – continue with next
                continue

            # Valid move → keep and update last_direction reference
            filtered_moves.append(move)
            last_direction = move

        # If all moves were filtered out, return empty list
        if not filtered_moves:
            print("All moves were invalid reversals. Not moving.")

        return filtered_moves

    def _generate_apple(self) -> NDArray[np.int_]:
        """Generate a new apple at a random empty position.
        
        Returns:
            Array of [x, y] coordinates for the new apple
        """
        position = generate_random_apple(self.snake_positions, self.grid_size)

        # Record the apple position in game state
        self.game_state.record_apple_position(position)

        # We do NOT start a new round when an apple is generated
        # Rounds are ONLY tied to LLM communications
        return position

    def set_apple_position(self, position: List[int]) -> bool:
        """Set the apple position manually.
        
        Args:
            position: Position to place the apple as [x, y]
            
        Returns:
            Boolean indicating if the position was valid and set successfully
        """
        try:
            x, y = position

            # Validate position
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                print(f"Invalid apple position: {position}")
                return False

            # Check if position is empty
            if any(np.array_equal([x, y], pos) for pos in self.snake_positions):
                print(f"Cannot place apple on snake: {position}")
                return False

            # Set the position
            self.apple_position = np.array([x, y])

            # Update game state with the new apple position
            self.game_state.record_apple_position(self.apple_position)

            # We do NOT start a new round when an apple is set
            # Rounds are ONLY tied to LLM communications

            # Refresh board to reflect the new apple
            self._update_board()

            if self.use_gui and self.gui:
                self.draw()

            return True
        except Exception as e:
            print(f"Error setting apple position: {e}")
            return False

    def make_move(self, direction_key: str) -> Tuple[bool, bool]:
        """Execute a move in the specified direction.
        
        Args:
            direction_key: String key of the direction to move in ("UP", "DOWN", etc.)
            
        Returns:
            Tuple of (game_active, apple_eaten) where:
                game_active: Boolean indicating if the game is still active
                apple_eaten: Boolean indicating if an apple was eaten on this move
        """
        # Standardize direction key to uppercase to handle case insensitivity
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

            # Record this as an invalid reversal
            self.game_state.record_invalid_reversal()

            # Return immediately, effectively making no move
            return True, False

        # Update current direction
        self.current_direction = direction

        # Calculate new head position according to our coordinate system
        head_x, head_y = self.head_position

        # Apply direction vector to head position
        new_head = np.array([
            head_x + direction[0],  # Apply dx to x-coordinate
            head_y + direction[1]   # Apply dy to y-coordinate
        ])

        # Debug log
        print(f"Moving {direction_key}: Head from ({head_x}, {head_y}) to ({new_head[0]}, {new_head[1]})")

        # Check if the new head position is where the apple is
        is_eating_apple_at_new_head = np.array_equal(new_head, self.apple_position)

        # Check for collisions - pass the apple flag to handle collisions correctly
        wall_collision, body_collision = self._check_collision(new_head, is_eating_apple_flag=is_eating_apple_at_new_head)

        if wall_collision:
            print(f"Game over! Snake hit wall moving {direction_key}")
            self.last_collision_type = 'WALL'

            # Record the fatal move so it is visible in logs, stats and replay
            # We treat it as a normal (non–apple-eating) step that immediately
            # causes the game end.
            self.game_state.record_move(direction_key, apple_eaten=False)

            # Note: We do NOT increment round_count when a collision occurs
            # This ensures round numbers in game_N.json match the prompts/responses folders
            self.game_state.record_game_end("WALL")

            # Extension hook so future controllers (RL, curriculum, etc.) can
            # react to a terminal event without duplicating collision logic.
            self._on_game_over("WALL")

            return False, False  # Game over, no apple eaten

        if body_collision:
            print(f"Game over! Snake hit itself moving {direction_key}")
            self.last_collision_type = 'SELF'

            # Record the fatal move for visibility and analytics
            self.game_state.record_move(direction_key, apple_eaten=False)

            # Note: We do NOT increment round_count when a collision occurs
            # This ensures round numbers in game_N.json match the prompts/responses folders
            self.game_state.record_game_end("SELF")

            # Extension hook for subclasses.
            self._on_game_over("SELF")

            return False, False  # Game over, no apple eaten

        # Check if the snake eats an apple
        apple_eaten = is_eating_apple_at_new_head

        # No collision, proceed with move
        # Create a copy of current snake positions to modify
        new_snake_positions = np.copy(self.snake_positions)

        # Add new head to snake positions (at the end)
        new_snake_positions = np.vstack((new_snake_positions, new_head))

        if not apple_eaten:
            # Remove tail (first element) if no apple eaten
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

        # Record move in game state - this handles incrementing the score if an apple was eaten
        self.game_state.record_move(direction_key, apple_eaten)

        # Display message if apple was eaten (after score has been updated)
        if apple_eaten:
            apples_emoji = "🍎" * self.score
            print(f"🚀 Apple eaten! Score: {self.score} {apples_emoji}")

        # Draw if GUI is available
        if self.use_gui and self.gui:
            self.draw()

        # Keep the GameData replica in sync for accurate snake_length in
        # summaries and replays.
        self.game_state.snake_positions = self.snake_positions.tolist()

        # Allow subclasses to post-process the step (e.g. reward shaping).
        self._post_move(apple_eaten)

        return True, apple_eaten  # Game continues, with or without apple eaten

    def _check_collision(
        self, head_position: NDArray[np.int_], is_eating_apple_flag: bool
    ) -> Tuple[bool, bool]:
        """Check for wall or body collision.

        This method is now a wrapper around the stateless `check_collision`
        utility function, passing the relevant parts of the game state.

        Args:
            head_position: The prospective new position of the snake's head.
            is_eating_apple_flag: Flag indicating if the snake just ate.

        Returns:
            A tuple of (wall_collision, body_collision).
        """
        return check_collision(
            head_position=head_position,
            snake_body=self.snake_positions,
            grid_size=self.grid_size,
            is_apple_eaten=is_eating_apple_flag,
        )

    def _get_current_direction_key(self) -> str:
        """Return the current direction as a string key (e.g., 'UP')."""
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
        return "RIGHT"  # Fallback

    # Public wrapper for external modules – avoids accessing the protected
    # helper directly and silences static-analysis warnings.
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
        """Current snake length.

        Single-source-of-truth: defer to the `GameData` tracker so that **all**
        components calculate the length the same way.  This avoids the subtle
        risk of future discrepancies if one piece updates the body array and
        the other is forgotten.
        """
        return self.game_state.snake_length 

    # ------------------
    # RL-friendly helpers (no deps on external libraries).
    # ------------------

    def get_state_snapshot(self):  # noqa: D401 – simple accessor
        """Return an immutable snapshot of the current board as a plain dict.

        Previously returned a ``GameState`` dataclass; that dependency has been
        removed to slim down Task-0.
        """
        return {
            "board": [row.copy() for row in self.board],
            "direction": self._get_current_direction_key() or "NONE",
            "apple": tuple(self.apple_position),
            "score": self.score,
            "steps": self.steps,
        }

    def reset_env(self):  # noqa: D401 – gym-like naming
        """Reset the environment and return the initial state snapshot.

        Added as a convenience for RL tasks (Task-3) and has *no* effect on
        Task-0 because nothing calls it yet.
        """

        self.reset()
        return self.get_state_snapshot()

    def step(self, action: str):  # gym-style signature
        """Perform *action* and return (next_state, reward, done, info).

        Reward = change in score (apples eaten) so it stays compatible with
        future dense/shape functions.
        """

        prev_score = self.score
        active, apple_eaten = self.make_move(action)
        reward = self.score - prev_score
        done = not active
        info = {"apple_eaten": apple_eaten}

        return self.get_state_snapshot(), reward, done, info

    # ---------------------
    # Extension hooks (NOP by default) – subclasses override as needed.
    # ---------------------

    def _post_move(self, apple_eaten: bool) -> None:  # pragma: no cover
        """Hook called after every successful move.

        Parameters
        ----------
        apple_eaten
            *True* when the move consumed an apple.
        """

        # Default implementation does nothing – override in subclasses.
        return None

    def _on_game_over(self, reason: str) -> None:  # pragma: no cover
        """Hook invoked right before returning from a terminal collision.

        The *reason* string follows the values passed to
        :py:meth:`GameData.record_game_end` ("WALL", "SELF", etc.).
        """

        return None

# ------------------
# Task-0 concrete controller (LLM-aware) – plugs in GameData.
# ------------------

class GameController(BaseGameController):
    """Task-0 controller uses the richer :class:`GameData` tracker."""

    GAME_DATA_CLS = GameData

    # No further overrides needed; the base class honours the custom data
    # container type and all behaviour remains unchanged.