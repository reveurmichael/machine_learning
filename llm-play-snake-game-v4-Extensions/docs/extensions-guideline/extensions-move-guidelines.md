# Preventing Duplicate-Move and Round-Sync Bugs in Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`) and provides essential guidelines for extension developers.

> **Audience:** Developers working on Task 1-5 inside the `extensions/` folder.
>
> **Scope:** How to integrate alternative planners (heuristics, RL, fine-tuned LLMs) without breaking the core invariants that keep Task-0 logs clean.

## 🎯 **Core System Invariants**

These invariants maintain data integrity and must never be violated:

### **I-1: One Planner Request ≙ One Round**
Every LLM/Planner request corresponds to exactly one round, keeping `round_count`, prompt/response filenames, and `RoundManager.rounds_data` in perfect synchronization.

### **I-2: `planned_moves` as Single Source of Truth**
The `planned_moves` queue is the only authoritative source for future moves, ensuring UI, replays, and analytics all observe the same data.

### **I-3: Centralized Move Execution**
Every executed move must pass through `GameLoop._execute_next_planned_move` to guarantee exactly-once recording in `RoundBuffer.moves`.

### **I-4: Proper Round Completion**
When a plan finishes, `GameManager.finish_round()` must be called to flush buffers and increment counters. This is never treated as an "EMPTY" sentinel.

### **I-5: Clear Separation of Concerns**
No physics class (controller/logic) may advance rounds. Only the game loop owns gameplay flow management.

## 🚫 **Common Implementation Pitfalls**

### **Duplicate First Move Bug**
**Problem:** Leaving the first element of a new plan in `planned_moves` after execution
**Symptoms:** Duplicate first move in JSON logs (`["RIGHT","RIGHT",...]`)
**Solution:** Pop the move immediately or route all moves through `_execute_next_planned_move`

### **Excessive Round Completion**
**Problem:** Calling `finish_round()` too frequently (e.g., after each apple)
**Symptoms:** Mismatch between prompt/response filenames and JSON rounds
**Solution:** Only call when the plan queue is completely empty

### **Misclassified Round Boundaries**
**Problem:** Treating normal round boundaries as `EMPTY` moves
**Symptoms:** False "Empty Moves occurred" warnings and premature termination
**Solution:** Use proper round-completion path (`finish_round`) instead of `_handle_no_move()`

### **Bypassed Plan Recording**
**Problem:** Not calling `RoundManager.record_planned_moves()`
**Symptoms:** Missing `planned_moves` in logs, broken replay functionality
**Solution:** Always feed the full plan through the recording system

## 🛠️ **Recommended Implementation Pattern**

### **Step 1: Subclass `BaseGameLoop`**
Override only the minimal set of required hooks:
- `_get_new_plan()` - Fetch/compute new plan and store in `game.planned_moves`
- Optionally: `_handle_no_move()` or `_handle_no_path_found()` for custom sentinels

### **Step 2: Avoid Overriding `run()`**
Unless absolutely necessary, inherit the existing `run()` implementation. If you must override, replicate the Task-0 logic that forces `_process_active_game`.

### **Step 3: Implement `_get_new_plan()` Correctly**
```python
def _get_new_plan(self) -> None:
    manager = self.manager

    # Handle round bookkeeping
    if getattr(manager, "_first_plan", False):
        manager._first_plan = False  # First round of new game
    else:
        manager.increment_round("heuristic new round")

    # Compute the plan
    plan = self.pathfinder.find_path(manager.game)  # Returns ["UP","LEFT",...]

    # Record for logging and UI
    manager.game.game_state.round_manager.record_planned_moves(plan)
    manager.game.planned_moves = plan

    # Signal readiness
    manager.need_new_plan = False
```

### **Step 4: Never Return First Move**
Let `_execute_next_planned_move()` handle move extraction to avoid duplication.

### **Step 5: Proper Round Completion**
When `_execute_next_planned_move()` returns `None`, call `manager.finish_round()` and set `manager.need_new_plan = True`.

### **Step 6: Respect Buffer Boundaries**
Never directly manipulate `RoundBuffer.moves`. Use `GameData.record_move()` or higher-level helpers.

## 📋 **Implementation Checklist**

### **Core Requirements**
- [ ] Subclass `BaseGameLoop` appropriately
- [ ] Implement `_get_new_plan()` with proper round management
- [ ] Use `RoundManager.record_planned_moves()` for all plans
- [ ] Let `_execute_next_planned_move()` handle move extraction
- [ ] Call `finish_round()` only when plan queue is empty
- [ ] Avoid direct manipulation of `RoundBuffer.moves`

### **Testing Requirements**
- [ ] Enable headless testing with `--max-games 1 --max-steps 50 --no-gui`
- [ ] Verify no duplicate moves in JSON logs
- [ ] Confirm round count matches prompt/response file count
- [ ] Test that replays work correctly
- [ ] Validate that all moves appear in round data

## 🔧 **Example Implementation**

### **Heuristic Planner Integration**
```python
from core.game_loop import BaseGameLoop

class HeuristicGameLoop(BaseGameLoop):
    """
    Game loop for heuristic-based Snake gameplay using pathfinding algorithms
    
    Design Pattern: Template Method Pattern
    - Inherits game loop structure from BaseGameLoop
    - Overrides planning method for heuristic-specific logic
    - Maintains all round management invariants
    """

    def _get_new_plan(self) -> None:
        manager = self.manager

        # Round bookkeeping (Invariant I-1)
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("heuristic planning")

        # Compute heuristic plan
        current_state = manager.game.get_state_snapshot()
        plan = self.pathfinder.find_optimal_path(current_state)

        # Record plan (Invariant I-2)
        manager.game.game_state.round_manager.record_planned_moves(plan)
        manager.game.planned_moves = plan

        # Signal completion
        manager.need_new_plan = False
```

### **Reinforcement Learning Integration**
```python
class RLGameLoop(BaseGameLoop):
    """
    Game loop for reinforcement learning agents
    
    Maintains invariants while enabling RL-specific training workflows
    """

    def _get_new_plan(self) -> None:
        manager = self.manager

        # Round management
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("rl action selection")

        # RL agent action selection
        state = manager.game.get_state_snapshot()
        action = self.rl_agent.select_action(state)
        
        # Convert single action to plan format
        plan = [action]

        # Record and store
        manager.game.game_state.round_manager.record_planned_moves(plan)
        manager.game.planned_moves = plan
        manager.need_new_plan = False
```

## 🧪 **Testing and Validation**

### **Automated Testing**
```python
def test_extension_log_integrity():
    """Test that extension maintains log integrity"""
    # Run extension in headless mode
    result = run_extension("--max-games 1 --max-steps 50 --no-gui")
    
    # Load generated logs
    game_data = json.load(open(result.latest_game_json))
    
    # Validate round integrity
    rounds_data = game_data["detailed_history"]["rounds_data"]
    
    for round_id, round_info in rounds_data.items():
        moves = round_info["moves"]
        
        # Check for duplicates
        assert moves == list(dict.fromkeys(moves)), f"Duplicates in round {round_id}"
        
        # Validate move sequence
        assert all(move in VALID_MOVES for move in moves), f"Invalid moves in round {round_id}"
    
    # Verify round count consistency
    expected_files = len([f for f in os.listdir(result.prompts_dir) if f.endswith('.txt')])
    actual_rounds = len(rounds_data)
    assert expected_files == actual_rounds, "Round count mismatch"
```

### **Manual Verification**
```bash
# Quick verification commands
python main.py --max-games 1 --max-steps 50 --no-gui
grep -o '"moves":\[.*\]' logs/latest/game_1.json | head -5
ls logs/latest/prompts/ | wc -l
ls logs/latest/responses/ | wc -l
```

## 🔗 **Integration Benefits**

Following these guidelines ensures:

### **Data Integrity**
- Consistent round numbering across all log components
- Clean, duplicate-free move sequences
- Reliable replay functionality

### **System Compatibility**
- Extensions work seamlessly with existing dashboards
- Analytics tools function correctly across all task types
- Cross-extension comparisons remain valid

### **Development Efficiency**
- Reduced debugging time for move-related issues
- Predictable behavior across different algorithm types
- Simplified testing and validation processes

---

**Following these implementation patterns keeps your extension's logs perfectly aligned with Task-0's robust data contracts, ensuring replays, dashboards, and analytics remain reliable across all future tasks.** 