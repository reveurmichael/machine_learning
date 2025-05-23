import pygame
import numpy as np

class DrawWindow:
    def __init__(self):
        self.cell_size = 20
        self.grid_size = 20
        self.width = self.cell_size * self.grid_size
        self.height = self.cell_size * self.grid_size + 100  # Extra space for info
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 24)
        
    def draw_board(self, board, board_info, head_position):
        """Draw the game board with snake and apple."""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Draw cell based on content
                if board[y][x] == board_info["snake"]:
                    # Make head a different color
                    if np.array_equal([x, y], head_position):
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Green head
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), rect)  # Darker green body
                elif board[y][x] == board_info["apple"]:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red apple
                else:
                    pygame.draw.rect(self.screen, (50, 50, 50), rect)  # Dark gray empty
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
        
        pygame.display.flip()
    
    def draw_game_info(self, score, steps, planned_moves=None, llm_response=None):
        """Draw game information below the board."""
        # Draw score and steps
        score_text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        steps_text = self.font.render(f"Steps: {steps}", True, (255, 255, 255))
        
        self.screen.blit(score_text, (10, self.height - 90))
        self.screen.blit(steps_text, (10, self.height - 60))
        
        # Draw planned moves if available
        if planned_moves:
            moves_text = self.font.render(f"Planned: {', '.join(planned_moves)}", True, (255, 255, 255))
            self.screen.blit(moves_text, (10, self.height - 30))
        
        # Draw LLM response if available
        if llm_response:
            response_text = self.font.render(f"LLM: {llm_response[:50]}...", True, (255, 255, 255))
            self.screen.blit(response_text, (10, self.height - 120))
        
        pygame.display.flip() 