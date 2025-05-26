"""
Replay GUI components for the Snake game.
Provides specialized GUI functionality for replay mode.
"""

import pygame
from gui.base import BaseGUI
from config import WHITE, APP_BG, GRID_BG, SNAKE_HEAD_C, SNAKE_C, APPLE_C

class ReplayGUI(BaseGUI):
    """GUI class for replay display."""
    
    def __init__(self):
        """Initialize the replay GUI."""
        super().__init__()
        self.init_display("Snake Game Replay")
    
    def draw(self, snake_positions, apple_position, game_number, score, steps, 
             move_index, total_moves, current_direction):
        """Draw the complete replay state.
        
        Args:
            snake_positions: List of snake segment positions
            apple_position: Position of the apple
            game_number: Current game number
            score: Current score
            steps: Current step count
            move_index: Index of current move
            total_moves: Total number of moves
            current_direction: Current direction of movement
        """
        # Fill background
        self.screen.fill(APP_BG)
        
        # Draw grid
        self.draw_grid()
        
        # Draw snake
        self.draw_snake(snake_positions)
        
        # Draw apple if available
        if apple_position is not None:
            self.draw_apple(apple_position)
        
        # Draw game info
        self.draw_replay_info(
            game_number=game_number,
            score=score,
            steps=steps,
            move_index=move_index,
            total_moves=total_moves,
            current_direction=current_direction
        )
        
        # Update display
        pygame.display.flip()
    
    def draw_replay_info(self, game_number, score, steps, move_index, total_moves, current_direction):
        """Draw replay information panel.
        
        Args:
            game_number: Current game number
            score: Current score
            steps: Current step count
            move_index: Index of current move
            total_moves: Total number of moves
            current_direction: Current direction of movement
        """
        # Set up font
        font = pygame.font.SysFont('arial', 20)
        
        # Right panel info
        info_text = [
            f"Game: {game_number}",
            f"Score: {score}",
            f"Steps: {steps}",
            f"Moves: {move_index}/{total_moves}",
            f"Current Direction: {current_direction or 'None'}",
            f"Press Space to pause/resume",
            f"Press N for next game",
            f"Press R to restart game",
            f"Press Esc to quit"
        ]
        
        # Display each line
        y_offset = 20
        for text in info_text:
            text_surface = font.render(text, True, WHITE)
            self.screen.blit(text_surface, (self.height + 20, y_offset))
            y_offset += 30 