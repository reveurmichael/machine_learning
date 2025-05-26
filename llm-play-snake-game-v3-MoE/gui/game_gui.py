"""
Game GUI module for the Snake game.
Handles the main game interface.
"""

import pygame
from gui.base import BaseGUI
from config import GRID_SIZE, CELL_SIZE, BLACK, GREY3


class GameGUI(BaseGUI):
    """Handles the main game interface."""

    def __init__(self, grid_size=10, cell_size=40):
        """Initialize the game GUI.
        
        Args:
            grid_size: Size of the game grid
            cell_size: Size of each cell in pixels
        """
        super().__init__(grid_size, cell_size)
        self.initialize()

    def draw(self, game):
        """Draw the current game state.
        
        Args:
            game: SnakeGame instance
        """
        # Clear the screen
        self.clear_screen()
        
        # Draw the grid
        self.draw_grid()
        
        # Draw the snake
        self.draw_snake(game.body)
        
        # Draw the apple
        self.draw_apple(game.apple)
        
        # Draw the score and steps
        self.draw_score(game.score)
        self.draw_steps(game.steps)
        
        # Draw game over message if game is over
        if game.game_over:
            self.draw_game_over(game.collision_type)
        
        # Update the display
        self.update_display()

    def draw_llm_response(self, response_text):
        """Draw the LLM response.
        
        Args:
            response_text: The LLM response text to display
        """
        super().draw_llm_response(response_text)
        self.update_display() 