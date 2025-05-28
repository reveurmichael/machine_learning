"""
Game GUI components for the Snake game.
Provides specialized GUI functionality for the main game.
"""

import pygame
from gui.base import BaseGUI
from config import COLORS

class GameGUI(BaseGUI):
    """GUI class for the main game display."""
    
    def __init__(self):
        """Initialize the game GUI."""
        super().__init__()
        self.init_display("LLM Snake Agent")
    
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
        
        # Update display for this region
        pygame.display.update((0, 0, self.height, self.height))
    
    def draw_snake_segment(self, x, y, is_head=False):
        """Draw a single snake segment.
        
        Args:
            x: X coordinate
            y: Y coordinate
            is_head: Whether this segment is the head
        """
        rect = pygame.Rect(
            x * self.pixel,
            y * self.pixel,
            self.pixel - 5,
            self.pixel - 5
        )
        
        color = COLORS['SNAKE_HEAD'] if is_head else COLORS['SNAKE_BODY']
        pygame.draw.rect(self.screen, color, rect)
    
    def draw_game_info(self, game_info):
        """Draw game information and LLM response.
        
        Args:
            game_info: Dictionary containing game information:
                - score: Current game score
                - steps: Current step count
                - planned_moves: List of planned moves from LLM
                - llm_response: Processed LLM response text ready for display
        """
        # Clear info panel
        self.clear_info_panel()
        
        # Extract values from game_info dictionary
        score = game_info.get('score', 0)
        steps = game_info.get('steps', 0)
        planned_moves = game_info.get('planned_moves')
        llm_response = game_info.get('llm_response')
        
        # Draw score and steps
        score_text = self.font.render(f"Score: {score}", True, COLORS['BLACK'])
        steps_text = self.font.render(f"Steps: {steps}", True, COLORS['BLACK'])
        
        self.screen.blit(score_text, (self.height + 20, 20))
        self.screen.blit(steps_text, (self.height + 20, 60))
        
        # Always display "Planned moves:" header
        moves_text = self.font.render("Planned moves:", True, COLORS['BLACK'])
        self.screen.blit(moves_text, (self.height + 20, 100))
        
        # Display planned moves with proper formatting:
        # - When planned_moves is a non-empty list: show the moves
        # - When planned_moves is an empty list: show "[]" (LLM returned zero moves)
        # - When planned_moves is None: show empty string (all moves completed)
        if planned_moves is not None:
            if len(planned_moves) > 0:
                moves_str = ", ".join(planned_moves)
            else:
                moves_str = ""  # Empty list from LLM
        else:
            moves_str = ""  # No more planned moves to execute
            
        moves_display = self.font.render(moves_str, True, COLORS['BLACK'])
        self.screen.blit(moves_display, (self.height + 20, 130))
        
        # Draw LLM response if available
        if llm_response:
            response_title = self.font.render("LLM Response:", True, COLORS['BLACK'])
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