"""
Replay GUI module for the Snake game.
Handles the replay interface.
"""

import pygame
from gui.base import BaseGUI


class ReplayGUI(BaseGUI):
    """Handles the replay interface."""

    def __init__(self, grid_size=10, cell_size=40):
        """Initialize the replay GUI.
        
        Args:
            grid_size: Size of the game grid
            cell_size: Size of each cell in pixels
        """
        super().__init__(grid_size, cell_size)
        self.initialize()

    def draw(self, game, game_number, round_number, current_move):
        """Draw the current replay state.
        
        Args:
            game: SnakeGame instance
            game_number: Current game number
            round_number: Current round number
            current_move: Current move number
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
        
        # Draw replay information
        self.draw_replay_info(game_number, round_number, current_move)
        
        # Draw game over message if game is over
        if game.game_over:
            self.draw_game_over(game.collision_type)
        
        # Update the display
        self.update_display()

    def draw_replay_info(self, game_number, round_number, current_move):
        """Draw replay-specific information.
        
        Args:
            game_number: Current game number
            round_number: Current round number
            current_move: Current move number
        """
        # Create the info text
        info_text = [
            f"Game: {game_number}",
            f"Round: {round_number}",
            f"Move: {current_move}",
            "",
            "Controls:",
            "Space: Pause/Resume",
            "Right: Next Move",
            "Left: Previous Move",
            "R: Restart",
            "Q: Quit"
        ]
        
        # Draw each line of text
        for i, line in enumerate(info_text):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20)) 