# Core game and UI constants (extracted from former config.py)

# -------------------------------- Colors ---------------------------------
COLORS = {
    'SNAKE_HEAD': (255, 140, 0),    # Bright orange for snake head
    'SNAKE_BODY': (209, 204, 192),  # Light gray for snake body
    'APPLE': (192, 57, 43),         # Red for apple
    'BACKGROUND': (44, 44, 84),     # Dark blue background
    'GRID': (87, 96, 111),          # Grid lines
    'TEXT': (255, 255, 255),        # White text
    'ERROR': (231, 76, 60),         # Red for error messages
    'BLACK': (0, 0, 0),             # Black
    'WHITE': (255, 255, 255),       # White
    'GREY': (189, 195, 199),        # Light grey
    'APP_BG': (240, 240, 240)       # App background
}

# ------------------------------ GUI tuning -------------------------------
WINDOW_WIDTH = 800      # Width of the application window
WINDOW_HEIGHT = 600     # Height of the application window
TIME_DELAY = 40         # General delay time for the game loop
TIME_TICK = 280         # Tick rate for the game

# Pause times (in seconds)
PAUSE_BETWEEN_MOVES_SECONDS = 1.0   # Pause time between moves

# ----------------------------- Game rules --------------------------------
GRID_SIZE = 10
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 20
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 20

DIRECTIONS = {
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
}

__all__ = [
    'COLORS',
    'WINDOW_WIDTH', 'WINDOW_HEIGHT', 'TIME_DELAY', 'TIME_TICK',
    'PAUSE_BETWEEN_MOVES_SECONDS',
    'GRID_SIZE',
    'MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED',
    'MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED',
    'DIRECTIONS',
] 