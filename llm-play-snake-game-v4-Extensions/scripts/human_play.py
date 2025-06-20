"""Human playable Snake game (moved to scripts/).

Launch with:
    python scripts/human_play.py

The script guarantees it runs from the repository root so that relative paths
(e.g. logs/) behave consistently.

This whole module is NOT Task0 specific. But no need to make it generic anyway. 
"""

from __future__ import annotations

import pygame
from pygame.locals import (
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_r,
)

from gui.game_gui import GameGUI
from core.game_controller import GameController
from config.ui_constants import TIME_TICK, COLORS
from utils.path_utils import ensure_repo_root


class HumanGameGUI(GameGUI):
    """GUI class for human-controlled Snake."""

    def __init__(self):
        super().__init__()
        self.init_display("Snake Game - Human Player")
        self.game_over = False

    def draw_game_info(self, game_info: dict):
        self.clear_info_panel()
        score = game_info.get("score", 0)
        steps = game_info.get("steps", 0)

        title_font = pygame.font.SysFont("arial", 22, bold=True)
        regular_font = pygame.font.SysFont("arial", 20)

        self.screen.blit(title_font.render("Game Statistics", True, COLORS["BLACK"]), (self.height + 20, 20))
        self.screen.blit(regular_font.render(f"Score: {score}", True, COLORS["BLACK"]), (self.height + 30, 50))
        self.screen.blit(regular_font.render(f"Steps: {steps}", True, COLORS["BLACK"]), (self.height + 30, 80))

        self.screen.blit(title_font.render("Controls", True, COLORS["BLACK"]), (self.height + 20, 130))
        ctrls = [
            "- Arrow Keys: Move Snake",
            "- R: Reset Game",
            "- Esc: Quit Game",
        ]
        y_off = 160
        for c in ctrls:
            self.screen.blit(regular_font.render(c, True, COLORS["BLACK"]), (self.height + 30, y_off))
            y_off += 30

        self.screen.blit(title_font.render("Instructions", True, COLORS["BLACK"]), (self.height + 20, 270))
        instr = [
            "- Eat apples to grow longer",
            "- Avoid hitting walls",
            "- Don't collide with yourself",
            "- Try to get the highest score!",
        ]
        y_off = 300
        for ins in instr:
            self.screen.blit(regular_font.render(ins, True, COLORS["BLACK"]), (self.height + 30, y_off))
            y_off += 30

        if self.game_over:
            self.screen.blit(title_font.render("Game Status", True, COLORS["ERROR"]), (self.height + 20, 430))
            self.screen.blit(regular_font.render("GAME OVER!", True, COLORS["ERROR"]), (self.height + 30, 460))
            self.screen.blit(regular_font.render("Press R to restart", True, COLORS["BLACK"]), (self.height + 30, 490))

        pygame.display.flip()

    def set_game_over(self, is_game_over: bool):
        self.game_over = is_game_over


def _handle_input(game: GameController, gui: HumanGameGUI) -> bool:
    running = True
    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
            elif event.key == K_UP:
                active, _ = game.make_move("UP"); gui.set_game_over(not active)
            elif event.key == K_DOWN:
                active, _ = game.make_move("DOWN"); gui.set_game_over(not active)
            elif event.key == K_LEFT:
                active, _ = game.make_move("LEFT"); gui.set_game_over(not active)
            elif event.key == K_RIGHT:
                active, _ = game.make_move("RIGHT"); gui.set_game_over(not active)
            elif event.key == K_r:
                game.reset(); gui.set_game_over(False)
    return running


def main():
    pygame.init()
    gui = HumanGameGUI()
    game = GameController(use_gui=True)
    game.set_gui(gui)
    pygame.display.set_caption("Snake Game - Human Player")
    clock = pygame.time.Clock()

    running = True
    while running:
        running = _handle_input(game, gui)
        gui.draw_board(game.board, game.board_info, game.head_position)
        gui.draw_game_info({"score": game.score, "steps": game.steps})
        clock.tick(TIME_TICK)

    pygame.quit()


if __name__ == "__main__":
    _repo_root = ensure_repo_root()
    main() 
