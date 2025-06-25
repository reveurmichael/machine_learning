"""
Human playable version of the Snake game.
Allows a human player to control the snake with arrow keys.
"""

import sys
import pygame
from pygame.locals import (  # pylint: disable=no-name-in-module
    QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_r
)
from gui.game_gui import GameGUI
from core.game_controller import GameController
from config.ui_constants import TIME_TICK, COLORS


class HumanGameGUI(GameGUI):
    """GUI class for the human-played Snake game."""

    def __init__(self):
        """Initialize the human game GUI."""
        super().__init__()
        self.init_display("Snake Game - Human Player")

    def draw_game_info(self, game_info: dict):
        """Draw basic score/step information for human-controlled games.

        Args:
            game_info: Dictionary containing game statistics. Expected keys:
                - score: Current game score
                - steps: Number of steps taken so far
        """
        assert self.screen is not None, "Screen must be initialized before drawing"

        # Clear info panel
        self.clear_info_panel()

        # Extract statistics
        score = game_info.get("score", 0)
        steps = game_info.get("steps", 0)

        # Draw score and steps
        title_font = pygame.font.SysFont('arial', 22, bold=True)
        regular_font = pygame.font.SysFont('arial', 20)

        # Title for game stats
        stats_title = title_font.render("Game Statistics", True, COLORS['BLACK'])
        self.screen.blit(stats_title, (self.height + 20, 20))

        # Game stats
        score_text = regular_font.render(f"Score: {score}", True, COLORS['BLACK'])
        steps_text = regular_font.render(f"Steps: {steps}", True, COLORS['BLACK'])

        self.screen.blit(score_text, (self.height + 30, 50))
        self.screen.blit(steps_text, (self.height + 30, 80))

        # Controls section
        controls_title = title_font.render("Controls", True, COLORS['BLACK'])
        self.screen.blit(controls_title, (self.height + 20, 130))

        controls = [
            "- Arrow Keys: Move Snake",
            "- R: Reset Game",
            "- Esc: Quit Game"
        ]

        y_offset = 160
        for control in controls:
            control_text = regular_font.render(control, True, COLORS['BLACK'])
            self.screen.blit(control_text, (self.height + 30, y_offset))
            y_offset += 30

        # Instructions section
        instructions_title = title_font.render("Instructions", True, COLORS['BLACK'])
        self.screen.blit(instructions_title, (self.height + 20, 270))

        instructions = [
            "- Eat apples to grow longer",
            "- Avoid hitting walls",
            "- Don't collide with yourself",
            "- Try to get the highest score!"
        ]

        y_offset = 300
        for instruction in instructions:
            instruction_text = regular_font.render(instruction, True, COLORS['BLACK'])
            self.screen.blit(instruction_text, (self.height + 30, y_offset))
            y_offset += 30

        # Game status section for game over
        if hasattr(self, 'game_over') and self.game_over:
            status_title = title_font.render("Game Status", True, COLORS['ERROR'])
            self.screen.blit(status_title, (self.height + 20, 430))

            status_text = regular_font.render("GAME OVER!", True, COLORS['ERROR'])
            self.screen.blit(status_text, (self.height + 30, 460))

            restart_text = regular_font.render("Press R to restart", True, COLORS['BLACK'])
            self.screen.blit(restart_text, (self.height + 30, 490))

        # Update display
        pygame.display.flip()

    def set_game_over(self, is_game_over):
        """Set the game over status for display.

        Args:
            is_game_over: Boolean indicating if game is over
        """
        self.game_over = is_game_over


def handle_input(game, gui):
    """Handle keyboard input for snake control.

    Args:
        game: The GameController instance
        gui: The GUI instance

    Returns:
        Boolean indicating if the game should continue
    """
    # Default to continuing the game
    running = True

    # Process all events
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_UP:
                game_active, _ = game.make_move("UP")
                gui.set_game_over(not game_active)
            elif event.key == K_DOWN:
                game_active, _ = game.make_move("DOWN")
                gui.set_game_over(not game_active)
            elif event.key == K_LEFT:
                game_active, _ = game.make_move("LEFT")
                gui.set_game_over(not game_active)
            elif event.key == K_RIGHT:
                game_active, _ = game.make_move("RIGHT")
                gui.set_game_over(not game_active)
            elif event.key == K_r:
                game.reset()
                gui.set_game_over(False)

    return running


def main():
    """Run the human playable snake game."""
    # Initialize pygame
    pygame.init()  # pylint: disable=no-member

    # Set up game components with custom GUI
    gui = HumanGameGUI()
    gui.game_over = False

    # Create game controller and connect it to the GUI
    game = GameController(use_gui=True)
    game.set_gui(gui)

    # Set window title
    pygame.display.set_caption("Snake Game - Human Player")

    # Main game loop
    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle user input
        running = handle_input(game, gui)

        # Draw the game state - convert board to proper type and head position
        board_list = [[int(cell) for cell in row] for row in game.board]
        head_pos = game.head_position.tolist() if game.head_position is not None else None
        gui.draw_board(board_list, game.board_info, head_pos)
        gui.draw_game_info({"score": game.score, "steps": game.steps})

        # Control game speed
        clock.tick(TIME_TICK)

    # Clean up
    pygame.quit()  # pylint: disable=no-member
    sys.exit()


if __name__ == "__main__":
    main()
