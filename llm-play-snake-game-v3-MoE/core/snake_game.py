"""
Snake game module for the Snake game.
Handles the core game logic and state.
"""

import random
import pygame
from gui.game_gui import GameGUI


class SnakeGame:
    """Handles the core game logic and state."""

    def __init__(self, grid_size=10, cell_size=40):
        """Initialize the snake game.
        
        Args:
            grid_size: Size of the game grid
            cell_size: Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Game state
        self.head = (grid_size // 2, grid_size // 2)
        self.body = [self.head]
        self.apple = self._generate_apple()
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.collision_type = None
        
        # GUI
        self.gui = None

    def set_gui(self, gui):
        """Set the GUI instance.
        
        Args:
            gui: GameGUI instance
        """
        self.gui = gui

    def reset(self):
        """Reset the game state."""
        self.head = (self.grid_size // 2, self.grid_size // 2)
        self.body = [self.head]
        self.apple = self._generate_apple()
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.collision_type = None

    def move(self, direction):
        """Move the snake in the specified direction.
        
        Args:
            direction: Direction to move ("UP", "DOWN", "LEFT", "RIGHT")
        """
        if self.game_over:
            return
            
        # Update direction
        self.direction = direction
        
        # Calculate new head position
        x, y = self.head
        if direction == "UP":
            y -= 1
        elif direction == "DOWN":
            y += 1
        elif direction == "LEFT":
            x -= 1
        elif direction == "RIGHT":
            x += 1
            
        # Check for collisions
        if (x < 0 or x >= self.grid_size or
            y < 0 or y >= self.grid_size or
            (x, y) in self.body):
            self.game_over = True
            self.collision_type = "wall" if (x < 0 or x >= self.grid_size or
                                           y < 0 or y >= self.grid_size) else "self"
            return
            
        # Update head and body
        self.head = (x, y)
        self.body.insert(0, self.head)
        
        # Check if apple was eaten
        if self.head == self.apple:
            self.score += 1
            self.apple = self._generate_apple()
        else:
            self.body.pop()
            
        # Increment steps
        self.steps += 1

    def _generate_apple(self):
        """Generate a new apple position.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            pos = (x, y)
            if pos not in self.body:
                return pos 