"""Game-wide constants shared across tasks.

The list of AVAILABLE_PROVIDERS depends on importing :pymod:`llm.providers`
which in turn triggers the initialisation of the *llm* package.  Importing it
eagerly here caused a circular-import chain when other packages pulled in
`config.game_constants` early during the boot-process (``llm.__init__`` →
``llm.agent_llm`` → ``core.*`` → this module).

To avoid that, we defer the import to **runtime** via a helper function so
that modules that only need *static* constants (e.g. `DIRECTIONS`) do not pay
the cost or risk of initialising the LLM sub-package.
"""


def list_available_providers() -> list[str]:  # noqa: D401 – tiny helper
    """Return the provider list, importing lazily to dodge circular deps."""

    try:
        from llm.providers import list_providers  # local import

        return list_providers()
    except Exception:  # pragma: no cover – safe fallback
        return []


# Expose the list at module import *after* the helper so CLI scripts can still
# treat it as a constant – but the lazy function avoids circular-import crash
# because the *value* is only needed in CLI context (after all packages have
# loaded).

# The *dynamic* provider list is only required by CLI helpers and dashboards.
# To avoid triggering the heavy LLM-provider import graph during early module
# loading (and thus risking circular imports), we expose an **empty list** as
# default.  User-facing code should call ``list_available_providers()`` when
# it actually needs the data.

AVAILABLE_PROVIDERS: list[str] = []  # Task-0 specific – populated lazily


PAUSE_BETWEEN_MOVES_SECONDS = 1.0  # Pause time between moves. This one is NOT Task0 specific.

PAUSE_PREVIEW_BEFORE_MAKING_FIRST_MOVE_SECONDS = 3.0  # This one is NOT Task0 specific.

MAX_GAMES_ALLOWED = 2 # This one is NOT Task0 specific.
MAX_STEPS_ALLOWED = 400 # This one is NOT Task0 specific.
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 3   # Only relevant for Task-0 and distillation3fine-tune tracks. So this one is Task0 specific.
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 3  # Parsing/LLM errors – Task-0 & distillation. So this one is Task0 specific.
MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED = 10  # Applies to ALL tasks (heuristics/RL/SL/LLM). So this one is NOT Task0 specific.
MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED = 1  # Task-0: LLM admitted no path; heuristics/RL treat as done. So this one is NOT Task0 specific.

# Optional pause after *any* EMPTY tick (irrespective of reason).  Separate
# control previously tied to NO_PATH_FOUND. So it's Task0 specific.
SLEEP_AFTER_EMPTY_STEP = 3.0  # minutes


# ---------------------
# Sentinel move names recorded in game logs.  Keep this single source of truth
# so replay, analytics, and any future tooling can reference the same list.
# ---------------------

# This one is NOT Task0 specific. Indeed, we know that "EMPTY" and "SOMETHING_IS_WRONG" are ONLY used in Task0. And "INVALID_REVERSAL" and "NO_PATH_FOUND" are used in Task0, Task1, Task2, Task3, Task4, Task5. But this won't have any effect on the other tasks. So we just regard it as NOT Task0 specific, hence we regard it as generic.

SENTINEL_MOVES = (
    "INVALID_REVERSAL",  # blocked reversal attempt
    "EMPTY",             # LLM produced no move
    "SOMETHING_IS_WRONG",  # parsing / LLM error
    "NO_PATH_FOUND",     # LLM explicitly stated no safe path
)

# This one is NOT Task0 specific.
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]

# This one is NOT Task0 specific.s
DIRECTIONS = {
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
}


# ---------------------
# End-reason mapping
# Single source of truth for user-facing explanations of why a game ended.
# Kept in sync with GameData.record_game_end() and front-end displays.
# This one is NOT Task0 specific.
# Indeed, we know that "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED" and "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED" are ONLY used in Task0. And "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED" and "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED" are used in all tasks (Task0, Task1, Task2, Task3, Task4, Task5). But this won't have any effect on the other tasks. So we just regard it as NOT Task0 specific, hence we regard it as generic.
# ---------------------

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
