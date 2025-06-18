## Question

Will planned_moves record those NO_PATH_FOUND EMPTY SOMETHING_IS_WRONG sentitals?


## Answer


No.  
The list `game.planned_moves` is strictly meant for “real” moves that the snake could still execute (UP / DOWN / LEFT / RIGHT). Sentinel placeholders are handled elsewhere:

• When a NO_PATH_FOUND, EMPTY or SOMETHING_IS_WRONG condition is detected the game loop calls  
  `record_no_path_found_move()`, `record_empty_move()` or `record_something_is_wrong_move()` on `GameData`.  
  Those helpers append the sentinel to the permanent `moves` history (and update the step counters) but never touch `planned_moves`.

• In `llm.parsing_utils.parse_llm_response()` the very first thing we do is  
  `game_instance.planned_moves = []` – clearing any leftovers before we *refill* the list with the directions extracted from the LLM JSON response. If parsing fails and a SOMETHING_IS_WRONG sentinel is logged, the planned-move buffer therefore remains empty.

• The NO_PATH_FOUND branch in `core/game_loop._handle_no_path_found()` (and the EMPTY branch in `_handle_no_move()`) likewise updates statistics and `current_game_moves` without touching the plan queue.

So the queue the GUI shows under “Planned moves” will never contain sentinel strings; only legal movement directions appear there.