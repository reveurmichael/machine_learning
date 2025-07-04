We have, in ROOT/config/game_constants.py:



END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS_REACHED": "Max Steps",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Empty Moves",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Something Is Wrong",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Invalid Reversals",
    "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED": "Max No Path Found",
}


Therefore, all extensions should use this map to define the end reason. Particularly, you want to make sure that each game_N.json file has a valid end reason, from one of the keys in the END_REASON_MAP.

