VITAL: usually, we should never change code in Task0_ROOT/Core. But this time, I am dead serious to refactor. So you can change the code of Task0, for this time. I know Task0 is working well, and I know extensions/heuristics-v0.04 is working well. But I want to refactor the code to make it more elegant and clean and clear.

Problem 1: 
In heuristics-v0.04/game_data.py, we have this: @dataclass
class HeuristicTimeStats:
    """Time statistics for heuristic algorithms without LLM pollution.
    
    This class provides the same timing functionality as TimeStats but
    excludes LLM-specific fields like llm_communication_time.
    """
    start_time: float
    
    def record_end_time(self) -> None:
        """Record the end time."""
        self.end_time = time.time()
    
    def asdict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        end = getattr(self, 'end_time', None) or time.time()
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": end - self.start_time,
        }
. In a good design, most likely in the derived class we don't need to have this, because start and end time stats should be there for all tasks. But this one should be shared across all the derived classes. Task0 should have that as well (maybe already have that).?

Problem 2:
In heuristics-v0.04/game_data.py,  it's too long and error prone. In a good design, it should be much shorter and cleaner for the derived classes. I have leaved a lot of TODOs in the code and I want to refactor this.

Problem 3:
In heuristics-v0.04/game_logic.py, it's complex. I don't know if we should refactor this. I have leaved a lot of TODOs in the code and I want to refactor this (maybe we should refactor this, or maybe we should not refactor this).


Problem 4:
In heuristics-v0.04/game_manager.py, it's complex. I don't know if we should refactor this. I have leaved a lot of TODOs in the code and I want to refactor this (maybe we should refactor this, or maybe we should not refactor this).

Problem 5:
While summary.json should always be there, which is already good, I want the generation of game_N.json to be optional (maybe by argspaser? maybe just a params there for this moment.)


Problem 6:
Ideally, HOW THINGS ARE DONE is already there in the Base Classes. In the derived classes, we should be worried about WHAT THINGS should be there. This should be very true for game_data and game_stats and maybe game_rounds. Maybe for other core classes as well.

The attached md files are important. Please read them. 

Check attached md files: todo-core.md, task0-improvement-for-extensions.md, limits-manager-impact-on-extensions.md. Their opinion might be useful for you. 


