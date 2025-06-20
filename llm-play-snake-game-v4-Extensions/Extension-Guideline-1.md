If we were to have a BaseClassBlabla, which will be used by task0, 1, 2, 3, 4, 5, and then Task0 extends BaseClassBlabla, and then Task1, Task2, Task3, Task4, Task5 extends, either Task0, or BaseClassBlabla, then we can have a very flexible and easy to extend codebase.

## Attributes and functions that should NOT be in BaseClassBlabla
So one good principle to follow is that, in BaseClassBlabla, we should NOT have attributes and functions specifically related to LLM or LLM playing snake game and not to other algorithms playing snake game. For example, such attributes and functions should NOT be in BaseClassBlabla:
self.llm_response
self.primary_llm
self.secondary_llm
self.llm_response_time
self.llm_response_time_primary
self.llm_response_time_secondary
self.llm_response_time_primary_avg
self.llm_response_time_secondary_avg
self.max_consecutive_empty_moves_allowed 
self.max_consecutive_something_is_wrong_allowed
self.empty_steps
self.something_is_wrong_steps
self.consecutive_empty_steps
self.consecutive_something_is_wrong
self.time_stats
self.token_stats
self.awaiting_plan
All thing related to Continuation mode should not be in BaseClassBlabla. Continuation mode is a feature that is only used for task0.
def report_final_statistics(self) (this one is for every task different implementation)
def continue_from_session
def continue_from_directory
llm_response


## Base Class that is already OK (mostly OK, but maybe not perfect)

class BaseReplayEngine(GameController)
class BaseGameManager
class BaseGameController
class BaseGameLogic(GameController)
class BaseGameData (except one minor thing: it s self.stats = GameStatistics should later on become self.stats = BaseGameStatistics)

## Attributes and functions that should be in BaseClassBlabla
On the contrary, in BaseClassBlabla, we should have attributes and functions that are generic and can be used by all tasks. For example, such attributes and functions (or properties, or methods) should be in BaseClassBlabla:
self.round_counts
self.total_rounds
self.round_counts
self.total_rounds
self.round_manager (BaseRoundManager)
self.grid_size
self.board
self.snake_positions
self.apple_position
self.score
self.steps
self.move_index
self.total_moves
self.need_new_plan
self.planned_moves
self.head_position
self.game_state
self.game_over
self.game_end_reason 
self.max_consecutive_invalid_reversals_allowed
self.max_consecutive_no_path_found_allowed
self.current_direction 
self.last_collision_type
self.apple_positions_history
self.use_gui
self.gui
self.game_count
self.total_score
self.total_steps
self.valid_steps
self.invalid_reversals
self.consecutive_invalid_reversals
self.consecutive_no_path_found
self.last_no_path_found
self.no_path_found_steps
self.game
self.current_game_moves
self.running
self.args.move_pause
def get_pause_between_moves(self)
self.game_active
def setup_game(self):
self.log_dir
self.pause_between_moves
self.auto_advance
def set_gui(self, gui_instance)
def _build_state_base(self) -> dict: # These keys may be absent on some subclasses but harmlessly for task1, task2, task3, task4, task5
def load_next_game(self)
def execute_replay_move(self
def handle_events(self)

Also, task 0 having a bit of extra is not really a problem. For example, we can still use SENTINEL_MOVES = ( "INVALID_REVERSAL", "EMPTY", "SOMETHING_IS_WRONG", "NO_PATH_FOUND", ), though, for task1, task2, task3, we will only be using "INVALID_REVERSAL" and "NO_PATH_FOUND".

## Recap what I asked you to do:

We are preparing really hard to be able to deliver as is said in project-structure-plan.md . Now we want to make the current codebase as prepared as possible for the new Task1, Task2, Task3, Task4, Task5. think first,Our code should follow the principle of SOLID, open to extension, closed to modification.  We do things only for task0  yet (change no functionality, but more prepared for future roadmap). No need to do anything for Task1, Task2, Task3, Task4, Task5. But, make sure your modification will always be used for task0. No need to prepare things that are not used for task0 (so no need for over-praparation,  but it's good be very future-proof, without introducing unncessary code that is not used for task0 yet). Future Task1, Task2, Task3, Task4, Task5 will most like inherit from the base class (BaseGameData, BaseGameManager, BaseController), because OOP make things more flexible and easy to extend. all python files in the folder "core" should be named in game_*.py pattern. all python files in the folder "utils" should be named in *_utils.py pattern. See the example of replay_data.py, it's a very good SOLID example for future preparation. One good idea is to have a base example BaseClassBlabla for future extension, and for  Task0 (LLM playing snake game) , it extends BaseClassBlabla as well. Important: Move as many will-be-used-by-task-0-1-2-3-4-5 attributes and functions into BaseClassBlabla as possible (but not all, for attributes that will not be useful for task1, task2, task3, task4, task5, we should not make them into the BaseClassBlabla). 

## Now what you should do:

Make sure things within the folder "core" are really well refactored, sometimes we have a lot of inter-class dependances. double check.



