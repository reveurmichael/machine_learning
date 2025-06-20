# This is NOT Task0 specific.
COLORS = {
    "SNAKE_HEAD": (255, 140, 0),  # Bright orange for snake head
    "SNAKE_BODY": (209, 204, 192),  # Light gray for snake body
    "APPLE": (192, 57, 43),  # Red for apple
    "BACKGROUND": (44, 44, 84),  # Dark blue background
    "GRID": (87, 96, 111),  # Grid lines
    "TEXT": (255, 255, 255),  # White text
    "ERROR": (231, 76, 60),  # Red for error messages
    "BLACK": (0, 0, 0),  # Black
    "WHITE": (255, 255, 255),  # White
    "GREY": (189, 195, 199),  # Light grey
    "APP_BG": (240, 240, 240),  # App background
}

# This is NOT Task0 specific.
GRID_SIZE = 10

WINDOW_WIDTH = 800  # Width of the application window. This one is NOT Task0 specific.
WINDOW_HEIGHT = 600  # Height of the application window. This one is NOT Task0 specific.
TIME_DELAY = 40  # General delay time for the game loop. This one is NOT Task0 specific.
TIME_TICK = 280  # Tick rate for the game. This one is NOT Task0 specific.


# ---------------------
# Default values - for Streamlit page only,
# *NOT* for main.py args default
# Those values are Task0 specific.
# ---------------------

DEFAULT_PROVIDER = "ollama"  # for Streamlit page only
DEFAULT_MODEL = "deepseek-r1:14b"  # for Streamlit page only args default
DEFAULT_PARSER_PROVIDER = "ollama"  # for Streamlit page only
DEFAULT_PARSER_MODEL = "gemma3:12b-it-qat"  # for Streamlit page only
