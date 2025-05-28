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
             game_end_reason=None, primary_llm=None, secondary_llm=None, game_timestamp=None, llm_response=None):
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
            llm_response: The LLM response text to display
        """
        # Fill background
        self.screen.fill(COLORS['BACKGROUND'])

        # Draw snake with custom method to ensure head is properly identified
        self.draw_snake_for_replay(snake_positions)

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
            paused=self.paused,
            llm_response=llm_response
        )

        # Update display
        pygame.display.flip()

    def draw_snake_for_replay(self, snake_positions):
        """Draw the snake for replay mode, ensuring the head is correctly identified.
        
        Args:
            snake_positions: List of [x,y] positions for snake segments
        """
        # Safely check if snake_positions exists and has elements
        # Use numpy's size or shape attribute instead of direct boolean evaluation
        if snake_positions is None or len(snake_positions) == 0:
            return

        # In replay mode, the head is always the last segment in the list
        head_index = len(snake_positions) - 1

        for i, position in enumerate(snake_positions):
            x, y = position

            # Draw rectangle for snake segment
            rect = pygame.Rect(
                x * self.pixel,
                y * self.pixel,
                self.pixel - 5,
                self.pixel - 5
            )

            # Use different color for head (which is the last segment in the positions list)
            if i == head_index:
                pygame.draw.rect(self.screen, COLORS['SNAKE_HEAD'], rect)
            else:
                pygame.draw.rect(self.screen, COLORS['SNAKE_BODY'], rect)

    def set_paused(self, paused):
        """Set the paused state of the replay.
        
        Args:
            paused: Boolean indicating if replay is paused
        """
        self.paused = paused

    def draw_replay_info(self, game_number, score, steps, move_index, total_moves, current_direction,
                        game_end_reason=None, primary_llm=None, secondary_llm=None, game_timestamp=None,
                        paused=False, llm_response=None):
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
            llm_response: The LLM response to display
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

        # Stats without direction
        stats_text = [
            f"Game: {game_number}",
            f"Score: {score}",
            f"Steps: {steps}",
            f"Progress: {move_index}/{total_moves} ({int(move_index/max(1, total_moves)*100)}%)"
        ]

        # Display each statistic
        y_offset = 80
        for text in stats_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30

        # Add game end reason if available
        if game_end_reason:
            end_reason_map = {
                "WALL": "Hit Wall",
                "SELF": "Hit Self",
                "MAX_STEPS": "Max Steps",
                "EMPTY_MOVES": "Empty Moves",
                "ERROR": "LLM Error"
            }
            end_reason_text = end_reason_map.get(game_end_reason, game_end_reason)
            reason_text = font.render(f"End Reason: {end_reason_text}", True, COLORS['BLACK'])
            self.screen.blit(reason_text, (self.height + 30, y_offset))
            y_offset += 30

        # LLM information section
        y_offset += 10
        llm_title = title_font.render("LLM Information", True, COLORS['BLACK'])
        self.screen.blit(llm_title, (self.height + 20, y_offset))
        y_offset += 40

        llm_text = [
            f"Primary: {primary_llm or 'Unknown'}",
            f"Parser: {secondary_llm or 'None'}"
        ]

        for text in llm_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30

        progress_title = title_font.render("Progress", True, COLORS["BLACK"])
        self.screen.blit(progress_title, (self.height + 20, y_offset))
        y_offset += 40

        # Progress bar for replay
        y_offset += 10
        self.draw_progress_bar(move_index, total_moves, self.height + 20, y_offset, 
                              self.text_panel_width - 40, 20)
        y_offset += 40

        # Controls section with updated instructions
        controls_title = title_font.render("Controls", True, COLORS['BLACK'])
        self.screen.blit(controls_title, (self.height + 20, y_offset))
        y_offset += 40

        controls_text = [
            "Space: Pause/Resume",
            "Left/Right Arrows: Prev/Next Game",
            "Up/Down Arrows: Speed Up/Down",
            "R: Restart Current Game",
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
        score_text = self.font.render(f"Score: {score}", True, COLORS['BLACK'])
        steps_text = self.font.render(f"Steps: {steps}", True, COLORS['BLACK'])

        self.screen.blit(score_text, (self.height + 20, 20))
        self.screen.blit(steps_text, (self.height + 20, 60))

        # Draw planned moves if available
        if planned_moves:
            moves_text = self.font.render("Planned moves:", True, COLORS['BLACK'])
            self.screen.blit(moves_text, (self.height + 20, 100))

            # Display each planned move
            moves_str = ", ".join(planned_moves)
            moves_display = self.font.render(moves_str, True, COLORS['GREY3'])
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
