from llm.providers import list_providers


AVAILABLE_PROVIDERS = list_providers()


PAUSE_BETWEEN_MOVES_SECONDS = 1.0  # Pause time between moves

MAX_GAMES_ALLOWED = 2
MAX_STEPS_ALLOWED = 400
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 3   # Only relevant for Task-0 and distillation3fine-tune tracks
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 3  # Parsing/LLM errors – Task-0 & distillation
MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED = 10   # Applies to ALL tasks (heuristics/RL/SL/LLM)
MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED = 1  # Task-0: LLM admitted no path; heuristics/RL treat as done

# Optional pause after *any* EMPTY tick (irrespective of reason).  Separate
# control previously tied to NO_PATH_FOUND.
SLEEP_AFTER_EMPTY_STEP = 3.0  # minutes


# --------------------------
# Sentinel move names recorded in game logs.  Keep this single source of truth
# so replay, analytics, and any future tooling can reference the same list.
# --------------------------

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


# --------------------------
# End-reason mapping
# Single source of truth for user-facing explanations of why a game ended.
# Kept in sync with GameData.record_game_end() and front-end displays.
# --------------------------

END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS_REACHED": "Max Steps",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Empty Moves",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Something Is Wrong",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Invalid Reversals",
    "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED": "Max No Path Found",
}

"""Shared constants for **Task-0 (LLM Snake)** and future tracks.

The values below serve as *sane defaults* for the production LLM game.  They
are intentionally conservative so a runaway model does not soft-lock a session
(e.g. by emitting thousands of EMPTY moves).  Second-citizen tasks will adopt
a subset of these limits:

• Heuristics / RL / Supervised (Task-1 → 3)
  – operate with *deterministic* policies or value functions and never expect
    EMPTY or SOMETHING_IS_WRONG sentinels – only path-planning failures or
    invalid reversals.
  – therefore they respect **INVALID_REVERSAL** and **NO_PATH_FOUND** caps but
    ignore EMPTY / SOMETHING_IS_WRONG.

• Distillation / Fine-tuning (Task-4 → 5)
  – still interact with an LLM and thus may hit any sentinel; they inherit *all*
    limits unchanged.

These comments are documentation-only and do not affect Task-0 behaviour.
"""
