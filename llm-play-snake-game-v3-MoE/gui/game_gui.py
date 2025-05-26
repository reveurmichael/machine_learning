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

    def draw_board(self, board, board_info, head_position=None):
        """Draw the game board with snake and apple.
        
        Args:
            board: 2D array representing the game board
            board_info: Dictionary with board entity information
            head_position: Position of the snake's head as [x, y]
        """
        # Clear the game area
        self.clear_game_area()

        # Draw board entities
        for y, grid_line in enumerate(board):
            for x, value in enumerate(grid_line):
                # Calculate the actual display position with flipped y-coordinate
                # Flip y-coordinate since (0,0) is now bottom-left but pygame draws from top-left
                display_y = self.grid_size - 1 - y
                
                if value == board_info["snake"]:  # snake
                    # Use a different color for the head
                    is_head = (head_position is not None and head_position[0] == x and head_position[1] == y)
                    self.draw_snake_segment(x, display_y, is_head)
                elif value == board_info["apple"]:  # apple
                    self.draw_apple([x, y], flip_y=True)
                    
        # Draw walls/borders
        self.draw_walls()
        
        # Update display for this region
        pygame.display.update((0, 0, self.height+1, self.height+1))
    
    def draw_snake_segment(self, x, y, is_head=False):
        """Draw a single snake segment.
        
        Args:
            x: X coordinate
            y: Y coordinate
            is_head: Whether this segment is the head
        """
        from config import SNAKE_C, SNAKE_HEAD_C
        
        rect = pygame.Rect(
            x * self.pixel,
            y * self.pixel,
            self.pixel - 5,
            self.pixel - 5
        )
        
        color = SNAKE_HEAD_C if is_head else SNAKE_C
        pygame.draw.rect(self.screen, color, rect)
    
    def draw_game_info(self, score, steps, planned_moves=None, llm_response=None):
        """Draw game information and LLM response.
        
        Args:
            score: Current game score
            steps: Current step count
            planned_moves: List of planned moves from LLM
            llm_response: Processed LLM response text ready for display
        """
        # Clear info panel
        self.clear_info_panel()
        
        # Draw score and steps
        score_text = self.font.render(f"Score: {score}", True, BLACK)
        steps_text = self.font.render(f"Steps: {steps}", True, BLACK)
        
        self.screen.blit(score_text, (self.height + 20, 20))
        self.screen.blit(steps_text, (self.height + 20, 60))
        
        # Draw planned moves if available
        if planned_moves:
            moves_text = self.font.render("Planned moves:", True, BLACK)
            self.screen.blit(moves_text, (self.height + 20, 100))
            
            # Display each planned move
            moves_str = ", ".join(planned_moves)
            moves_display = self.font.render(moves_str, True, GREY3)
            self.screen.blit(moves_display, (self.height + 20, 130))
        
        # Draw LLM response if available
        if llm_response:
            response_title = self.font.render("LLM Response:", True, BLACK)
            self.screen.blit(response_title, (self.height + 20, 170))
            
            # Draw the response text area
            self.render_text_area(
                llm_response,
                self.height + 20,  # x
                200,               # y
                self.text_panel_width,  # width
                350                # height
            )
        
        # Update display
        pygame.display.flip() 