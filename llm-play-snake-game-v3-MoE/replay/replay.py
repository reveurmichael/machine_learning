"""
Replay module for the Snake game.
Provides a simple interface to replay previously recorded games.
"""

import pygame
from pygame.locals import *
import os
import sys
import json
import time
import numpy as np

# Import from the replay engine module
from replay.replay_engine import ReplayEngine

def handle_key_events(event):
    """Handle key events for replay navigation.
    
    Args:
        event: Pygame event to handle
        
    Returns:
        Action to take based on the key press
    """
    # Quit events
    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        return "quit"
        
    # Navigation keys
    if event.type == KEYDOWN:
        # Speed control
        if event.key in (K_UP, K_s):
            return "speed_up"
        
        if event.key in (K_DOWN, K_d):
            return "speed_down"
            
        # Game navigation
        if event.key == K_r:
            return "restart"
            
        if event.key in (K_RIGHT, K_n):
            return "next_game"
            
        if event.key in (K_LEFT, K_p):
            return "prev_game"
            
        # Pause control
        if event.key == K_SPACE:
            return "toggle_pause"
            
    return None

def replay_game(log_dir, game_number=1, move_pause=0.5, auto_advance=False):
    """Replay a recorded game.
    
    Args:
        log_dir: Directory containing game logs
        game_number: Game number to replay
        move_pause: Pause duration between moves
        auto_advance: Whether to automatically advance to the next game
        
    Returns:
        Final game state
    """
    engine = ReplayEngine(log_dir, move_pause, auto_advance)
    engine.load_game_data(game_number)
    engine.run()
    return engine 