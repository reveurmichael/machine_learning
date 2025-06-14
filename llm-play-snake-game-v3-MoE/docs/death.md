## Question

For  all human_play.py mode,  human_play_web.py mode, replay.py mode, replay_web.py mode, main.py mode, do they have the same death (self collision, wall, max step reached, max consecutive empty/wrong responses, max consecutive invalid reversals, etc.) detection machanism? Are things coherent?


## Answer
END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS_REACHED": "Max Steps Reached",
    "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED": "Max Consecutive Empty Moves Reached",
    "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED": "Max Consecutive Something Is Wrong Reached",
    "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED": "Max Consecutive Invalid Reversals Reached",
}