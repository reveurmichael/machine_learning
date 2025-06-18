from llm.providers import list_providers


AVAILABLE_PROVIDERS = list_providers()


PAUSE_BETWEEN_MOVES_SECONDS = 1.0  # Pause time between moves

MAX_GAMES_ALLOWED = 2
MAX_STEPS_ALLOWED = 400
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 5
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 5
MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED = 10


VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]

DIRECTIONS = {
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
}


# -------------------------------
# End-reason mapping
# Single source of truth for user-facing explanations of why a game ended.
# Kept in sync with GameData.record_game_end() and front-end displays.
# -------------------------------

END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS_REACHED": "Max Steps Reached",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Consecutive Empty Moves Reached",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Consecutive Something Is Wrong Reached",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Consecutive Invalid Reversals Reached",
}
