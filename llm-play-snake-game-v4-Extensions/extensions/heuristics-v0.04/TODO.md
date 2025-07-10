VITAL: usually, we should never change code in Task0_ROOT/Core. But this time, I am dead serious to refactor. So you can change the code of Task0, for this time. I know Task0 is working well, and I know extensions/heuristics-v0.04 is working well. But I want to refactor the code to make it more elegant and clean and clear.

Problem 1: 
In heuristics-v0.04/game_data.py, we have this: def record_game_end. In a good design, most likely in the derived class we don't need to override the record_game_end method. For this time we have to do this because in the base class we don't have self.stats.time_stats.record_end_time() . But this one should be shared across all the derived classes. Task0 should have that as well (maybe already have that).?

Problem 2:
In heuristics-v0.04/game_data.py,  it's too long and error prone. In a good design, it should be much shorter and cleaner for the derived classes. I have leaved a lot of TODOs in the code and I want to refactor this.

Problem 3:
In heuristics-v0.04/game_logic.py, it's complex. I don't know if we should refactor this. I have leaved a lot of TODOs in the code and I want to refactor this (maybe we should refactor this, or maybe we should not refactor this).


Problem 4:
In heuristics-v0.04/game_manager.py, it's complex. I don't know if we should refactor this. I have leaved a lot of TODOs in the code and I want to refactor this (maybe we should refactor this, or maybe we should not refactor this).

Problem 5:
While summary.json should always be there, which is already good, I want the generation of game_N.json to be optional (maybe by argspaser? maybe just a params there for this moment.)


The attached md files are important. Please read them. 