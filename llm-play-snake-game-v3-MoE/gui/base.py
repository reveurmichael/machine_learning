"""
Base GUI module for the Snake game.
Provides common GUI elements and functionality.
"""

import pygame
import numpy as np
from config import GRID_SIZE, CELL_SIZE, COLORS


class BaseGUI:
    """Base class for all GUI implementations."""

    def __init__(self, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
        """Initialize the base GUI.
        
        Args:
            grid_size: Size of the game grid
            cell_size: Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        self.screen = None
        self.font = None

    def initialize(self):
        """Initialize pygame and create the window."""
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Snake Game")
        self.font = pygame.font.Font(None, 36)

    def draw_grid(self):
        """Draw the game grid."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, COLORS['GRID'], rect, 1)

    def draw_snake(self, snake_positions):
        """Draw the snake.
        
        Args:
            snake_positions: List of [x, y] positions for each snake segment
        """
        for i, pos in enumerate(snake_positions):
            x, y = pos
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            # Head is a different color
            color = COLORS['SNAKE_HEAD'] if i == 0 else COLORS['SNAKE_BODY']
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, COLORS['GRID'], rect, 1)

    def draw_apple(self, apple_position):
        """Draw the apple.
        
        Args:
            apple_position: [x, y] position of the apple
        """
        if apple_position is not None:
            x, y = apple_position
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, COLORS['APPLE'], rect)
            pygame.draw.rect(self.screen, COLORS['GRID'], rect, 1)

    def draw_score(self, score):
        """Draw the current score.
        
        Args:
            score: Current game score
        """
        score_text = self.font.render(f"Score: {score}", True, COLORS['TEXT'])
        self.screen.blit(score_text, (10, 10))

    def draw_steps(self, steps):
        """Draw the current step count.
        
        Args:
            steps: Current number of steps
        """
        steps_text = self.font.render(f"Steps: {steps}", True, COLORS['TEXT'])
        self.screen.blit(steps_text, (10, 50))

    def draw_game_over(self, collision_type):
        """Draw the game over screen.
        
        Args:
            collision_type: Type of collision that ended the game
        """
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(128)
        overlay.fill(COLORS['BLACK'])
        self.screen.blit(overlay, (0, 0))

        # Draw game over text
        game_over_text = self.font.render("Game Over!", True, COLORS['TEXT'])
        text_rect = game_over_text.get_rect(center=(self.window_size/2, self.window_size/2 - 50))
        self.screen.blit(game_over_text, text_rect)

        # Draw collision type
        collision_text = self.font.render(f"Collision: {collision_type}", True, COLORS['TEXT'])
        collision_rect = collision_text.get_rect(center=(self.window_size/2, self.window_size/2))
        self.screen.blit(collision_text, collision_rect)

        # Draw restart instructions
        restart_text = self.font.render("Press R to restart", True, COLORS['TEXT'])
        restart_rect = restart_text.get_rect(center=(self.window_size/2, self.window_size/2 + 50))
        self.screen.blit(restart_text, restart_rect)

    def draw_error(self, error_message):
        """Draw an error message.
        
        Args:
            error_message: The error message to display
        """
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(128)
        overlay.fill(COLORS['BLACK'])
        self.screen.blit(overlay, (0, 0))

        # Draw error text
        error_text = self.font.render("Error!", True, COLORS['ERROR'])
        text_rect = error_text.get_rect(center=(self.window_size/2, self.window_size/2 - 50))
        self.screen.blit(error_text, text_rect)

        # Draw error message
        message_text = self.font.render(error_message, True, COLORS['ERROR'])
        message_rect = message_text.get_rect(center=(self.window_size/2, self.window_size/2))
        self.screen.blit(message_text, message_rect)

    def draw_llm_response(self, response_text):
        """Draw the LLM response.
        
        Args:
            response_text: The LLM response text to display
        """
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(128)
        overlay.fill(COLORS['BLACK'])
        self.screen.blit(overlay, (0, 0))

        # Draw response text
        response_lines = response_text.split('\n')
        for i, line in enumerate(response_lines):
            line_text = self.font.render(line, True, COLORS['TEXT'])
            line_rect = line_text.get_rect(center=(self.window_size/2, self.window_size/2 - 100 + i*40))
            self.screen.blit(line_text, line_rect)

    def clear_screen(self):
        """Clear the screen."""
        self.screen.fill(COLORS['BACKGROUND'])

    def update_display(self):
        """Update the display."""
        pygame.display.flip()

    def quit(self):
        """Clean up pygame resources."""
        pygame.quit() 