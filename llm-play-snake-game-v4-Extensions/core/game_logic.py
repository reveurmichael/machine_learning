"""
Snake game implementation with LLM integration.
Extends the base game controller with LLM-specific functionality.
"""

import traceback
from typing import List, Tuple, Dict

from core.game_controller import GameController
from llm.prompt_utils import prepare_snake_prompt
from llm.parsing_utils import parse_llm_response
from utils.text_utils import process_response_for_display
from config.ui_constants import GRID_SIZE

# ------------------
# BaseGameLogic – generic, LLM-agnostic subclass of GameController
# ------------------

class BaseGameLogic(GameController):
    """Generic game-loop convenience layer reused by all tasks.

    It extends :class:`GameController` by adding the *planning* helpers that
    are useful for both classical algorithms (BFS, Hamiltonian) and LLM-based
    agents.  Anything LLM-specific (prompt construction, token stats, etc.) is
    left to the concrete Task-0 subclass.
    """

    # pylint: disable=too-many-arguments – matches GameController signature
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True):
        super().__init__(grid_size, use_gui)

        # ----- Generic planning state (shared by all agent types) -----
        self.planned_moves: List[str] = []

    # ------------------
    # Generic helpers for planned-move agents
    # ------------------

    def get_next_planned_move(self):
        """Pop and return the next move from the current *planned_moves* list."""
        if self.planned_moves:
            return self.planned_moves.pop(0)
        return None

    # ------------------
    # Reset & state snapshots (LLM-agnostic)
    # ------------------

    def reset(self):  # type: ignore[override]
        """Reset core state and clear *planned_moves*.

        Returns a serialisable snapshot that higher-level tasks may use to
        feed their policy (e.g. heuristic path finder, RL observation).
        """
        super().reset()
        self.planned_moves = []
        return self.get_state_snapshot()

    def get_state_snapshot(self):  # noqa: D401 – helper method
        """Return a plain-Python snapshot of the current board.

        This neutral structure is intentionally NumPy-free so that second-
        citizen tasks (heuristics, RL, etc.) can depend on it without pulling
        the heavyweight scientific stack.
        """

        return {
            "board": [row.copy() for row in self.board],
            "direction": self._get_current_direction_key() or "NONE",
            "apple": tuple(self.apple_position),
            "score": self.score,
            "steps": self.steps,
        }

# ------------------
# Task-0 concrete implementation (LLM agent)
# ------------------

class GameLogic(BaseGameLogic):
    """Snake game with LLM agent integration."""
    
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
        
        # Use the utility function from llm_utils
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
            return parse_llm_response(response, process_response_for_display, self)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            traceback.print_exc()
            
            # Store the raw response for display
            self.processed_response = f"ERROR: Failed to parse LLM response\n\n{response[:200]}..."
            
            # Clear previous planned moves
            self.planned_moves = []
            
            # Update game state to record error
            self.game_state.record_something_is_wrong_move()
            
            return None
    