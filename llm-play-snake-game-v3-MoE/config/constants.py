import pathlib


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

WINDOW_WIDTH = 800      # Width of the application window
WINDOW_HEIGHT = 600     # Height of the application window
TIME_DELAY = 40         # General delay time for the game loop
TIME_TICK = 280         # Tick rate for the game

PAUSE_BETWEEN_MOVES_SECONDS = 1.0   # Pause time between moves

GRID_SIZE = 10
MAX_GAMES_ALLOWED = 2
MAX_STEPS_ALLOWED = 400
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 20
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 20
MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED = 20


AVAILABLE_PROVIDERS = sorted(
    {
        p.stem.replace("_provider", "")
        for p in pathlib.Path("llm/providers").glob("*_provider.py")
        if p.stem != "base_provider"
    }
)

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_PARSER_PROVIDER = "ollama"
DEFAULT_PARSER_MODEL = "gemma3:12b-it-qat"

DIRECTIONS = {
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
}

# -------------------------- End-reason mapping ---------------------------
# Single source of truth for user-facing explanations of why a game ended
# (kept in sync with GameData.record_game_end() and front-end displays).

END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS_REACHED": "Max Steps Reached",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Consecutive Empty Moves Reached",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Consecutive Something Is Wrong Reached",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Consecutive Invalid Reversals Reached",
}

__all__ = [
    'COLORS',
    'WINDOW_WIDTH', 'WINDOW_HEIGHT', 'TIME_DELAY', 'TIME_TICK',
    'PAUSE_BETWEEN_MOVES_SECONDS',
    'GRID_SIZE',
    'MAX_GAMES_ALLOWED',
    'MAX_STEPS_ALLOWED',
    'MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED',
    'MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED',
    'MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED',
    'DIRECTIONS',
    'AVAILABLE_PROVIDERS',
    'DEFAULT_PROVIDER', 'DEFAULT_MODEL', 'DEFAULT_PARSER_PROVIDER', 'DEFAULT_PARSER_MODEL',
    'END_REASON_MAP',
] 
