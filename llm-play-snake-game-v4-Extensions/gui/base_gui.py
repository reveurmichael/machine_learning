"""
Base GUI components for the Snake game.
Provides common setup and drawing functionality.

This whole module is NOT Task0 specific.
"""

import pygame
from typing import Protocol, List

from config.ui_constants import (
    COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, GRID_SIZE
)

# Global registry so second-citizen tasks can register HUD plug-ins once and
# have them automatically attached to all GUI instances without subclassing.
GLOBAL_PANELS: List["InfoPanel"] = []

def register_panel(panel: "InfoPanel") -> None:  # noqa: D401 – simple registry
    """Register *panel* (implements :class:`InfoPanel`) for all future GUIs."""

    if panel not in GLOBAL_PANELS:
        GLOBAL_PANELS.append(panel)

# ---------------------
# Optional plug-in interface so second-citizen tasks can inject HUD elements
# without modifying the first-citizen GUI code.
# ---------------------

class InfoPanel(Protocol):
    """Small widget that draws additional info next to the board."""

    def draw(self, surface: pygame.Surface, game: "core.GameLogic") -> None:  # type: ignore[name-defined]
        ...

class BaseGUI:
    """Base class for UI setup."""
    
    def __init__(self):
        """Initialize the UI dimensions."""
        self.width = WINDOW_WIDTH
        self.width_plus = 200
        self.height = WINDOW_HEIGHT
        self.grid_size = GRID_SIZE
        # Fallback to at least 1 pixel to avoid ZeroDivision and very large grids
        self.pixel = max(1, self.height // max(self.grid_size, 1))
        
        # Optional grid overlay (useful for RL visualisation, off by default)
        self.show_grid = False
        
    def init_display(self, title: str = "Snake Game"):
        """Initialize the pygame display.
        
        This method sets up the pygame display, screen surface, fonts, and other
        GUI components needed for rendering. It should be called after __init__
        but before any drawing operations.
        
        Args:
            title: Window title to display
        """
        import pygame  # deferred import to avoid hard dependency
        
        pygame.init()
        
        # Create the main display surface
        self.screen = pygame.display.set_mode((self.width + self.width_plus, self.height))
        pygame.display.set_caption(title)
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Initialize clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Calculate text panel width for info display
        self.text_panel_width = self.width_plus - 40  # Leave some margin
        
        # Initialize extra panels list for plugins
        self.extra_panels = []

    def draw_apple(self, apple_position, flip_y=False):
        """Draw the apple at the given position."""
        x, y = apple_position
        y_display = y if flip_y else (self.grid_size - 1 - y)
        padding = 2
        apple_size = self.pixel - (2 * padding)
        rect = pygame.Rect(
            x * self.pixel + padding,
            y_display * self.pixel + padding,
            apple_size,
            apple_size
        )
        pygame.draw.rect(self.screen, COLORS["APPLE"], rect)

    def clear_game_area(self):
        """Clear the game board area with properly aligned grid."""
        pygame.draw.rect(
            self.screen, 
            COLORS["BACKGROUND"], 
            (0, 0, self.height, self.height)
        )
        if self.show_grid and self.pixel >= 4:
            for i in range(self.grid_size + 1):
                x = i * self.pixel
                pygame.draw.line(self.screen, COLORS["GRID"], (x, 0), (x, self.height), 1)
            for i in range(self.grid_size + 1):
                y = i * self.pixel
                pygame.draw.line(self.screen, COLORS["GRID"], (0, y), (self.height, y), 1)
    def clear_info_panel(self):
        """Clear the information panel area."""
        # Draw background rectangle for info panel without separation line
        pygame.draw.rect(
            self.screen,
            COLORS['APP_BG'],
            (self.height, 0, self.width+self.width_plus-self.height, self.height)
        )
    
    def render_text_area(self, text, x, y, width, height, max_lines=20):
        """Render text inside a scrollable text area.
        
        Args:
            text: Text to render
            x: X position of text area
            y: Y position of text area
            width: Width of text area
            height: Height of text area
            max_lines: Maximum number of lines to render
        """
        # Setup text area with border
        pygame.draw.rect(
            self.screen,
            COLORS['GREY'],  # Light gray background for text area
            (x, y, width, height),
            0,  # Filled rectangle
            3   # Rounded corners
        )
        
        pygame.draw.rect(
            self.screen,
            COLORS['GREY'],  # Darker border
            (x, y, width, height),
            2,  # Border only (not filled)
            3   # Rounded corners
        )
        
        # Calculate text parameters
        avg_char_width = self.small_font.size("m")[0]
        max_chars_per_line = int(width / avg_char_width * 1.7)
        line_height = 20
        padding_left = 10
        
        # Split into lines and render
        y_offset = y + 10  # Starting y position inside the text area
        lines = text.split('\n')
        
        for line in lines[:max_lines]:
            # Skip empty lines
            if not line.strip():
                y_offset += line_height
                continue
                
            # Further split lines that are too long
            for i in range(0, len(line), max_chars_per_line):
                segment = line[i:i+max_chars_per_line]
                if segment:
                    text_surface = self.small_font.render(segment, True, COLORS['BLACK'])
                    self.screen.blit(text_surface, (x + padding_left, y_offset))
                    y_offset += line_height
                    
                    # Stop if we've reached the bottom of the text area
                    if y_offset > y + height - line_height:
                        return 
    
    def draw_game_info(self, game_info):
        # Hook for subclasses; default implementation also iterates plug-ins.
        for panel in self.extra_panels:
            panel.draw(self.screen, game_info.get("game"))

    # ------------------
    # Small utility so RL tasks that dynamically change grid size can
    # reconfigure the pixel scaling without reinstantiating the GUI.
    # ------------------

    def resize(self, grid_size: int):
        """Update internal scaling when the board size changes."""

        self.grid_size = grid_size
        self.pixel = max(1, self.height // max(self.grid_size, 1))
        
        # Optional grid overlay (useful for RL visualisation, off by default)

    # ------------------
    # Utility: return RGB numpy array of the current screen (for videos)
    # ------------------

    def get_rgb_array(self):
        """Return an RGB array of the current screen or ``None`` in headless mode."""

        if self.screen is None:
            return None

        import numpy as np

        surface = pygame.display.get_surface()
        if surface is None:
            return None

        return np.transpose(
            pygame.surfarray.array3d(surface), (1, 0, 2)
        ).copy()  # H×W×C

    # Simple API for tasks to toggle the grid overlay.
    def toggle_grid(self, show: bool | None = None):
        """Enable or disable the grid overlay.

        Called by second-citizen GUIs to show training heat-maps; noop for Task-0.
        """

        if show is None:
            self.show_grid = not self.show_grid
        else:
            self.show_grid = bool(show)

    def draw_progress_bar(self, current: int, total: int, x: int, y: int, width: int, height: int):
        """Generic progress‐bar widget used by multiple GUIs.

        Parameters
        ----------
        current
            Current value of the metric.
        total
            Total (max) value.  When 0 the bar is drawn empty.
        x, y, width, height
            Geometry of the bar in pixels.
        """

        import pygame  # Local to avoid hard dependency at import-time

        # Background bar (grey)
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS["GREY"], bg_rect)

        # Filled part representing progress
        if total > 0:
            progress_percent = max(0.0, min(1.0, current / total))
            progress_width = int(width * progress_percent)
            progress_rect = pygame.Rect(x, y, progress_width, height)
            pygame.draw.rect(self.screen, COLORS["SNAKE_HEAD"], progress_rect)

        # Border
        pygame.draw.rect(self.screen, COLORS["BLACK"], bg_rect, 1)

    # ------------------
    # Shared snake-segment renderer so GUI subclasses need not duplicate code.
    # ------------------

    def draw_snake_segment(
        self,
        x: int,
        y: int,
        is_head: bool = False,
        *,
        flip_y: bool = False,
    ) -> None:
        """Draw one snake segment.

        Parameters
        ----------
        x, y
            Grid coordinates of the segment.
        is_head
            When *True*, the segment is coloured as the head.
        flip_y
            If *True*, ``y`` is already in display space – skip the Cartesian
            flip applied by :py:meth:`draw_square`.
        """

        color = COLORS['SNAKE_HEAD'] if is_head else COLORS['SNAKE_BODY']
        self.draw_square(x, y, color, flip_y=flip_y)

    # ------------------
    # Low-level helper shared by GameGUI and ReplayGUI
    # ------------------

    def draw_square(
        self,
        x: int,
        y: int,
        color,
        *,
        flip_y: bool = False,
    ) -> None:
        """Draw a single snake/body/apple square at board coords ``(x, y)``.

        If ``flip_y`` is *False* the Y coordinate is converted from Cartesian
        (origin bottom-left) to PyGame screen space (origin top-left).  Pass
        ``flip_y=True`` when *y* is already in screen space to prevent double
        flipping – this is the case inside `GameGUI.draw_board`.
        """

        import pygame  # deferred import

        y_display = y if flip_y else (self.grid_size - 1 - y)

        # Calculate padding to center the square in the grid cell
        padding = 2  # Small padding for visual separation
        square_size = self.pixel - (2 * padding)

        rect = pygame.Rect(
            x * self.pixel + padding,
            y_display * self.pixel + padding,
            square_size,
            square_size,
        )

        pygame.draw.rect(self.screen, color, rect)
