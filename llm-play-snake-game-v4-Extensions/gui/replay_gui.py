"""PyGame GUI for the *replay* viewer of recorded Snake games."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from config.ui_constants import COLORS
from gui.base_gui import BaseGUI


# Note: This replay GUI displays LLM response details and is therefore
# Task-0 specific.  Future tasks can implement their own viewer subclassing
# :class:`BaseGUI`.

class ReplayGUI(BaseGUI):
    """PyGame-based overlay used by the offline *replay* mode."""

    def __init__(self) -> None:
        """Initialize the replay GUI. """
        super().__init__()
        
        # Initialize replay-specific attributes
        self.move_history = []
        self.planned_moves = []
        self.paused = False
        self.primary_llm = "Unknown/Unknown"
        self.secondary_llm = "Unknown/Unknown"
        self.llm_response = ""
        self.game_number = 1
        self.game_stats = None
        self.timestamp = "Unknown"
        
        # Initialize display with custom caption
        self.init_display("Snake Game Replay")

    def draw(self, replay_data: Mapping[str, Any]) -> None:
        """Draw the replay view.
        
        Args:
            replay_data: Dictionary containing replay information:
                - snake_positions: Array of snake positions
                - apple_position: Array of apple position
                - game_number: Current game number
                - score: Current score
                - steps: Current step count
                - move_index: Current move index
                - total_moves: Total number of moves
                - planned_moves: List of planned moves
                - llm_response: LLM response text
                - primary_llm: Name of primary LLM
                - secondary_llm: Name of secondary LLM
                - paused: Whether the replay is paused
                - timestamp: Timestamp of the game
                - game_end_reason: Reason the game ended (optional)
        """
        if not self.screen:
            return
        
        # Extract values from dictionary
        snake_positions = replay_data.get('snake_positions', [])
        apple_position = replay_data.get('apple_position', [0, 0])
        game_number = replay_data.get('game_number', 0)
        score = replay_data.get('score', 0)
        steps = replay_data.get('steps', 0)
        move_index = replay_data.get('move_index', 0)
        total_moves = replay_data.get('total_moves', 0)
        llm_response = replay_data.get('llm_response', '')
        primary_llm = replay_data.get('primary_llm', 'Unknown')
        secondary_llm = replay_data.get('secondary_llm', 'Unknown')
        paused = replay_data.get('paused', False)
        timestamp = replay_data.get('timestamp', 'Unknown')
        game_end_reason = replay_data.get('game_end_reason', None)
        final_score = replay_data.get('final_score', None)

        # Fill background
        self.screen.fill(COLORS['BACKGROUND'])

        # Draw snake with custom method to ensure head is properly identified
        self.draw_snake_for_replay(snake_positions)

        # Draw apple if available
        if apple_position is not None:
            self.draw_apple(apple_position)

        # Get current direction safely
        current_direction = "NONE"
        if (isinstance(snake_positions, list) and len(snake_positions) > 0 and 
            isinstance(snake_positions[-1], (list, tuple, np.ndarray))):
            # If snake_positions[-1] has at least 3 elements and the third is a string direction
            if len(snake_positions[-1]) > 2 and isinstance(snake_positions[-1][2], str):
                current_direction = snake_positions[-1][2]
        
        # Update move history if there's a new move
        if move_index > 0 and (len(self.move_history) < move_index):
            # Only add to history if we have a valid direction
            if isinstance(current_direction, str) and current_direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                self.move_history.append(current_direction)
            else:
                self.move_history.append("UNKNOWN")

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
            game_timestamp=timestamp,
            paused=paused,
            llm_response=llm_response,
            final_score=final_score
        )

        # Update display
        pygame.display.flip()

    def draw_snake_for_replay(self, snake_positions: Sequence[Sequence[int]]) -> None:
        """Draw the snake for replay mode, ensuring the head is correctly identified.
        
        Args:
            snake_positions: List of [x,y] positions for snake segments
        """
        if snake_positions is None or len(snake_positions) == 0:
            return

        # In replay mode, the head is always the last segment in the list
        head_index = len(snake_positions) - 1

        for i, position in enumerate(snake_positions):
            x, y = position

            # Head is the last element in the list during replay
            is_head = i == head_index
            self.draw_snake_segment(x, y, is_head)

    def set_paused(self, paused: bool) -> None:
        """Set the paused state of the replay.
        
        Args:
            paused: Boolean indicating if replay is paused
        """
        self.paused = paused

    def draw_replay_info(
        self,
        game_number: int,
        score: int,
        steps: int,
        move_index: int,
        total_moves: int,
        current_direction: str,
        game_end_reason: str | None = None,
        primary_llm: str | None = None,
        secondary_llm: str | None = None,
        game_timestamp: str | None = None,
        paused: bool = False,
        llm_response: str | None = None,
        final_score: int | None = None,
    ) -> None:
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
        score_display = f"- Score: {score}"
        if final_score is not None:
            score_display = f"- Score: {score}/{final_score}"
        
        stats_text = [
            f"- Game: {game_number}",
            score_display,
            f"- Progress: {move_index}/{total_moves} ({int(move_index/max(1, total_moves)*100)}%)"
        ]

        # Display each statistic
        y_offset = 80
        for text in stats_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30

        # Add game end reason if available
        if game_end_reason:
            end_reason_text = game_end_reason
            reason_text = font.render(f"- End Reason: {end_reason_text}", True, COLORS['BLACK'])
            self.screen.blit(reason_text, (self.height + 30, y_offset))
            y_offset += 30

        # LLM information section
        y_offset += 10
        llm_title = title_font.render("LLM Information", True, COLORS['BLACK'])
        self.screen.blit(llm_title, (self.height + 20, y_offset))
        y_offset += 40

        llm_text = [
            f"- Primary: {primary_llm or 'Unknown'}",
            f"- Parser: {secondary_llm or 'None'}"
        ]

        for text in llm_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 33

        progress_title = title_font.render("Progress", True, COLORS["BLACK"])
        self.screen.blit(progress_title, (self.height + 20, y_offset))
        y_offset += 30

        # Progress bar for replay
        y_offset += 10
        self.draw_progress_bar(move_index, total_moves, self.height + 20, y_offset, 
                              self.text_panel_width - 40, 20)
        y_offset += 30

        # Controls section with updated instructions
        controls_title = title_font.render("Controls", True, COLORS['BLACK'])
        self.screen.blit(controls_title, (self.height + 20, y_offset))
        y_offset += 40

        controls_text = [
            "- Space: Pause/Resume",
            "- Left/Right Arrows: Prev/Next Game",
            "- Up/Down Arrows: Speed Up/Down",
            "- R: Restart Current Game",
            "- Esc: Quit"
        ]

        for text in controls_text:
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (self.height + 30, y_offset))
            y_offset += 30
