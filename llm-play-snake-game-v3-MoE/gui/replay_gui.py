"""
Replay GUI components for the Snake game.
Provides specialized GUI functionality for replay mode.
"""

import pygame
from gui.base import BaseGUI
from config import COLORS

class ReplayGUI(BaseGUI):
    """GUI class for replay display."""
    
    def __init__(self):
        """Initialize the replay GUI."""
        super().__init__()
        self.init_display("Snake Game - Replay Mode")
        self.move_history = []
        self.current_move = None
        self.paused = False
        
    def draw(self, snake_positions, apple_position, game_number, score, steps, 
             move_index, total_moves, current_direction, 
             game_end_reason=None, primary_llm=None, secondary_llm=None, game_timestamp=None):
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
            game_end_reason: Reason the game ended (if applicable)
            primary_llm: Primary LLM provider/model
            secondary_llm: Secondary LLM provider/model (parser)
            game_timestamp: Timestamp when the game was played
        """
        # Fill background
        self.screen.fill(COLORS['BACKGROUND'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw snake
        self.draw_snake(snake_positions)
        
        # Draw apple if available
        if apple_position is not None:
            self.draw_apple(apple_position)
            
        # Update move history if there's a new move
        if move_index > 0 and (len(self.move_history) < move_index):
            self.current_move = current_direction
            self.move_history.append(current_direction)
        
        # Draw game info
        self.draw_replay_info(
            game_number=game_number,
            score=score,
            steps=steps,
            move_index=move_index,
            total_moves=total_moves,
            current_direction=current_direction,
            game_end_reason=game_end_reason,
            primary_llm=primary_llm,
            secondary_llm=secondary_llm,
            game_timestamp=game_timestamp,
            paused=self.paused
        )
        
        # Update display
        pygame.display.flip()
        
    def set_paused(self, paused):
        """Set the paused state of the replay.
        
        Args:
            paused: Boolean indicating if replay is paused
        """
        self.paused = paused
    
    def draw_replay_info(self, game_number, score, steps, move_index, total_moves, current_direction,
                        game_end_reason=None, primary_llm=None, secondary_llm=None, game_timestamp=None,
                        paused=False):
        """Draw replay information panel.
        
        Args:
            game_number: Current game number
            score: Current score
            steps: Current step count
            move_index: Index of current move
            total_moves: Total number of moves
            current_direction: Current direction of movement
            game_end_reason: Reason the game ended (if applicable)
            primary_llm: Primary LLM provider/model
            secondary_llm: Secondary LLM provider/model (parser)
            game_timestamp: Timestamp when the game was played
            paused: Whether the replay is paused
        """
        # Clear info panel with the same background color as other modes
        self.clear_info_panel()
        
        # Set up fonts
        font = pygame.font.SysFont('arial', 20)
        title_font = pygame.font.SysFont('arial', 22, bold=True)
        highlight_font = pygame.font.SysFont('arial', 20, bold=True)
        
        # Right panel info
        # Game title with status
        title_text = "Game Replay"
        if paused:
            title_text += " (PAUSED)"
        title = title_font.render(title_text, True, COLORS['ERROR'] if paused else COLORS['BLACK'])
        self.screen.blit(title, (self.height + 20, 10))
        
        # Game statistics section
        stats_title = title_font.render("Game Statistics", True, COLORS['BLACK'])
        self.screen.blit(stats_title, (self.height + 20, 50))
        
        stats_text = [
            f"Game: {game_number}",
            f"Score: {score}",
            f"Steps: {steps}",
            f"Progress: {move_index}/{total_moves} ({int(move_index/max(1, total_moves)*100)}%)",
            f"Direction: {current_direction or 'None'}"
        ]
        
        # Display each statistic
        y_offset = 80
        for text in stats_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30
        
        # Recent moves history
        moves_title = title_font.render("Recent Moves", True, COLORS['BLACK'])
        self.screen.blit(moves_title, (self.height + 20, y_offset + 10))
        y_offset += 40
        
        # Show last 5 moves
        recent_moves = self.move_history[-5:] if len(self.move_history) > 0 else ["None"]
        for i, move in enumerate(recent_moves):
            # Use highlight font for current move
            if i == len(recent_moves) - 1:
                move_text = highlight_font.render(f"➤ {move}", True, COLORS['SNAKE_HEAD'])
            else:
                move_text = font.render(f"   {move}", True, COLORS['BLACK'])
            self.screen.blit(move_text, (self.height + 30, y_offset))
            y_offset += 25
        
        # LLM information section
        llm_title = title_font.render("LLM Information", True, COLORS['BLACK'])
        self.screen.blit(llm_title, (self.height + 20, y_offset + 10))
        y_offset += 40
        
        llm_text = [
            f"Primary: {primary_llm or 'Unknown'}",
            f"Parser: {secondary_llm or 'None'}"
        ]
        
        for text in llm_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30
        
        # Game metadata section
        meta_title = title_font.render("Game Metadata", True, COLORS['BLACK'])
        self.screen.blit(meta_title, (self.height + 20, y_offset + 10))
        y_offset += 40
        
        # Format game end reason if available
        end_reason_text = "Unknown"
        if game_end_reason:
            end_reason_map = {
                "WALL": "Hit Wall",
                "SELF": "Hit Self",
                "MAX_STEPS": "Max Steps",
                "EMPTY_MOVES": "Empty Moves",
                "ERROR": "LLM Error"
            }
            end_reason_text = end_reason_map.get(game_end_reason, game_end_reason)
        
        meta_text = [
            f"End Reason: {end_reason_text}",
            f"Timestamp: {game_timestamp or 'Unknown'}"
        ]
        
        for text in meta_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30
        
        # Progress bar for replay
        self.draw_progress_bar(move_index, total_moves, self.height + 20, y_offset + 20, 
                              self.text_panel_width - 40, 20)
        y_offset += 50
        
        # Controls section
        controls_title = title_font.render("Controls", True, COLORS['BLACK'])
        self.screen.blit(controls_title, (self.height + 20, y_offset + 10))
        y_offset += 40
        
        controls_text = [
            "Space: Pause/Resume",
            "N: Next Game",
            "R: Restart Game",
            "S: Speed Up",
            "D: Slow Down",
            "Esc: Quit"
        ]
        
        for text in controls_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30
            
    def draw_progress_bar(self, current, total, x, y, width, height):
        """Draw a progress bar for the replay progress.
        
        Args:
            current: Current value
            total: Total value
            x: X position
            y: Y position
            width: Width of the progress bar
            height: Height of the progress bar
        """
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS['GREY2'], bg_rect)
        
        # Progress
        if total > 0:
            progress_percent = min(1.0, current / total)
            progress_width = int(width * progress_percent)
            progress_rect = pygame.Rect(x, y, progress_width, height)
            pygame.draw.rect(self.screen, COLORS['SNAKE_HEAD'], progress_rect)
        
        # Border
        pygame.draw.rect(self.screen, COLORS['BLACK'], bg_rect, 1) 