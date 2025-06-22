"""Core game configuration constants for the Snake game."""

from llm.providers import list_providers


AVAILABLE_PROVIDERS = list_providers()


PAUSE_BETWEEN_MOVES_SECONDS = 1.0  # Pause time between moves

MAX_GAMES_ALLOWED = 2
MAX_STEPS_ALLOWED = 400
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 3
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 3
MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED = 10
MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED = 1  # consecutive NO_PATH_FOUND responses before game over

# Optional pause after *any* EMPTY tick (irrespective of reason).  Separate
# control previously tied to NO_PATH_FOUND.
SLEEP_AFTER_EMPTY_STEP = 3.0  # minutes


# --------------------------------
# Sentinel move names recorded in game logs.  Keep this single source of truth
# so replay, analytics, and any future tooling can reference the same list.
# --------------------------------

SENTINEL_MOVES = (
    "INVALID_REVERSAL",  # blocked reversal attempt
    "EMPTY",             # LLM produced no move
    "SOMETHING_IS_WRONG",  # parsing / LLM error
    "NO_PATH_FOUND",     # LLM explicitly stated no safe path
)

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
    "MAX_STEPS_REACHED": "Max Steps",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Empty Moves",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Something Is Wrong",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Invalid Reversals",
    "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED": "Max No Path Found",
}
