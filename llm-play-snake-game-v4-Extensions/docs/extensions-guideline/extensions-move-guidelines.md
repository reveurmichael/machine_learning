# Extension Integration Guidelines: Preventing System Invariant Violations

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`) and provides essential guidelines for extension developers.

## 🎯 **Core Philosophy: Maintaining System Integrity**

When integrating alternative planners (heuristics, reinforcement learning, supervised learning) into the Snake Game AI architecture, developers must preserve critical system invariants that ensure data integrity, logging consistency, and replay functionality across all extensions.

### **Target Audience**
- Extension developers working on Task 1-5 implementations
- Developers integrating new algorithms into existing extensions
- Contributors maintaining system consistency across extension types

## 🔒 **Critical System Invariants**

These invariants are fundamental to system integrity and must never be violated:

### **I-1: Round-Request Synchronization**
**Requirement**: Every planner request corresponds to exactly one round
**Purpose**: Maintains perfect synchronization between `round_count`, prompt/response filenames, and `RoundManager.rounds_data`
**Impact**: Ensures consistent logging and replay functionality

### **I-2: Planned Moves Authority**
**Requirement**: The `planned_moves` queue serves as the single source of truth for future moves
**Purpose**: Ensures UI components, replay systems, and analytics observe identical data
**Impact**: Prevents data inconsistencies across system components

### **I-3: Centralized Move Execution**
**Requirement**: All executed moves must pass through `GameLoop._execute_next_planned_move`
**Purpose**: Guarantees exactly-once recording in `RoundBuffer.moves`
**Impact**: Ensures complete and accurate move tracking

### **I-4: Proper Round Lifecycle Management**
**Requirement**: `GameManager.finish_round()` must be called when plans complete
**Purpose**: Flushes buffers and increments counters properly
**Impact**: Maintains round boundary integrity and prevents EMPTY sentinel misclassification

### **I-5: Architectural Separation**
**Requirement**: Physics classes (controller/logic) cannot advance rounds
**Purpose**: Only the game loop controls gameplay flow
**Impact**: Maintains clear architectural boundaries and predictable behavior

## 🚫 **Common Implementation Anti-Patterns**

### **1. Duplicate Move Recording**
**Problem**: Leaving the first element of a new plan in `planned_moves` after execution
**Symptoms**: 
- Duplicate first moves in JSON logs (`["RIGHT","RIGHT",...]`)
- Inconsistent move sequences in replay systems
- Analytics reporting inflated move counts

**Solution**: Always pop moves immediately or route all execution through `_execute_next_planned_move`

```python
# ❌ INCORRECT: Potential duplicate recording
def execute_move(self):
    move = self.planned_moves[0]  # Gets move but doesn't remove
    self.game_controller.execute_move(move)  # Direct execution bypasses system

# ✅ CORRECT: Proper move execution
def execute_move(self):
    move = self._execute_next_planned_move()  # Gets and removes move properly
    # Move is automatically recorded by the system
```

### **2. Excessive Round Completion**
**Problem**: Calling `finish_round()` too frequently (e.g., after each apple collection)
**Symptoms**:
- Mismatch between prompt/response filenames and JSON round data
- Inflated round counts in analytics
- Broken replay synchronization

**Solution**: Only call `finish_round()` when the plan queue is completely empty

```python
# ❌ INCORRECT: Finishing rounds too often
def handle_apple_eaten(self):
    self.score += 1
    self.finish_round()  # Wrong: This is not a round boundary

# ✅ CORRECT: Proper round completion
def check_round_completion(self):
    if not self.planned_moves:  # Only when queue is empty
        self.finish_round()
        self.need_new_plan = True
```

### **3. Misclassified Round Boundaries**
**Problem**: Treating normal round boundaries as `EMPTY` moves
**Symptoms**:
- False "Empty Moves occurred" warnings
- Premature game termination
- Incorrect consecutive move counters

**Solution**: Use proper round-completion path instead of `_handle_no_move()`

```python
# ❌ INCORRECT: Misclassifying normal boundaries
def handle_plan_completion(self):
    self._handle_no_move()  # Wrong: This triggers EMPTY sentinel logic

# ✅ CORRECT: Proper round boundary handling
def handle_plan_completion(self):
    self.finish_round()  # Correct: Normal round completion
    self.need_new_plan = True
```

### **4. Bypassed Plan Recording**
**Problem**: Not calling `RoundManager.record_planned_moves()`
**Symptoms**:
- Missing `planned_moves` in JSON logs
- Broken replay functionality
- Incomplete analytics data

**Solution**: Always record the full plan through the system

```python
# ❌ INCORRECT: Bypassing recording system
def set_new_plan(self, moves):
    self.planned_moves = moves  # Direct assignment without recording

# ✅ CORRECT: Proper plan recording
def set_new_plan(self, moves):
    self.game_state.round_manager.record_planned_moves(moves)
    self.planned_moves = moves
```

## 🛠️ **Recommended Implementation Pattern**

### **Step 1: Extend BaseGameLoop**
Create extension-specific game loops by inheriting from `BaseGameLoop` and overriding minimal required methods:

```python
from core.game_loop import BaseGameLoop

class HeuristicGameLoop(BaseGameLoop):
    """
    Game loop for heuristic-based Snake gameplay using pathfinding algorithms
    
    Design Pattern: Template Method Pattern
    - Inherits proven game loop structure from BaseGameLoop
    - Overrides planning method for algorithm-specific logic
    - Maintains all system invariants automatically
    
    Educational Value:
    Demonstrates how inheritance enables code reuse while allowing
    customization of specific behaviors without breaking system contracts.
    """
    
    def __init__(self, manager, pathfinder):
        super().__init__(manager)
        self.pathfinder = pathfinder
    
    def _get_new_plan(self) -> None:
        """Override to implement heuristic planning logic"""
        # Implementation details below
        pass
```

### **Step 2: Implement _get_new_plan() Correctly**
This is the core method that must be implemented by all extensions:

```python
def _get_new_plan(self) -> None:
    """
    Generate new plan using heuristic algorithms
    
    This method implements the core planning logic while maintaining
    all system invariants for round management and move recording.
    """
    manager = self.manager

    # Handle round bookkeeping (Invariant I-1)
    if getattr(manager, "_first_plan", False):
        manager._first_plan = False  # First round of new game
    else:
        manager.increment_round("heuristic planning")

    # Compute the plan using algorithm-specific logic
    current_state = manager.game.get_state_snapshot()
    plan = self.pathfinder.find_optimal_path(current_state)  # Returns ["UP","LEFT",...]

    # Record plan for logging and UI (Invariant I-2)
    manager.game.game_state.round_manager.record_planned_moves(plan)
    manager.game.planned_moves = plan

    # Signal planning completion
    manager.need_new_plan = False
```

### **Step 3: Avoid Overriding run() Method**
Unless absolutely necessary, inherit the existing `run()` implementation to maintain consistency:

```python
class HeuristicGameLoop(BaseGameLoop):
    # ✅ RECOMMENDED: Use inherited run() method
    # The base implementation handles all invariants correctly
    
    # Only override if you need fundamentally different execution flow
    # If overriding, replicate Task-0 logic for _process_active_game
```

### **Step 4: Proper Move Execution**
Let the inherited system handle move extraction to avoid duplication:

```python
# ✅ CORRECT: Let system handle move execution
def process_game_step(self):
    if self.need_new_plan:
        self._get_new_plan()
    
    # System automatically calls _execute_next_planned_move()
    # No manual move extraction needed
```

### **Step 5: Handle Round Completion**
Implement proper round completion when the plan queue is empty:

```python
def check_plan_completion(self):
    """Check if current plan is complete and handle accordingly"""
    if not self.manager.game.planned_moves:  # Plan queue empty
        self.manager.finish_round()  # Proper round completion
        self.manager.need_new_plan = True  # Signal need for new plan
```

## 🧪 **Implementation Examples**

### **Heuristics Extension Integration**
```python
class HeuristicGameLoop(BaseGameLoop):
    """Heuristic pathfinding game loop with A* algorithm"""
    
    def __init__(self, manager, algorithm="ASTAR"):
        super().__init__(manager)
        self.pathfinder = self._create_pathfinder(algorithm)
    
    def _get_new_plan(self) -> None:
        manager = self.manager

        # Round management (Invariant I-1)
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("heuristic planning")

        # Algorithm-specific planning
        current_state = manager.game.get_state_snapshot()
        path = self.pathfinder.find_path(
            start=current_state["head_position"],
            goal=current_state["apple_position"],
            obstacles=current_state["snake_positions"][1:]  # Exclude head
        )

        # System integration (Invariant I-2)
        manager.game.game_state.round_manager.record_planned_moves(path)
        manager.game.planned_moves = path
        manager.need_new_plan = False
```

### **Reinforcement Learning Integration**
```python
class RLGameLoop(BaseGameLoop):
    """Reinforcement learning game loop with DQN agent"""
    
    def __init__(self, manager, rl_agent):
        super().__init__(manager)
        self.rl_agent = rl_agent
    
    def _get_new_plan(self) -> None:
        manager = self.manager

        # Round management
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("rl action selection")

        # RL-specific action selection
        state_tensor = self._convert_state_to_tensor(
            manager.game.get_state_snapshot()
        )
        action = self.rl_agent.select_action(state_tensor)
        
        # Convert single action to plan format
        plan = [self._action_to_direction(action)]

        # System integration
        manager.game.game_state.round_manager.record_planned_moves(plan)
        manager.game.planned_moves = plan
        manager.need_new_plan = False
```

### **Supervised Learning Integration**
```python
class SupervisedGameLoop(BaseGameLoop):
    """Supervised learning game loop with trained neural network"""
    
    def __init__(self, manager, model):
        super().__init__(manager)
        self.model = model
    
    def _get_new_plan(self) -> None:
        manager = self.manager

        # Round management
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False
        else:
            manager.increment_round("model prediction")

        # Model prediction
        features = self._extract_features(manager.game.get_state_snapshot())
        prediction = self.model.predict(features)
        move = self._prediction_to_direction(prediction)
        
        # Single move plan
        plan = [move]

        # System integration
        manager.game.game_state.round_manager.record_planned_moves(plan)
        manager.game.planned_moves = plan
        manager.need_new_plan = False
```

## 📋 **Implementation Validation Checklist**

### **Core Requirements**
- [ ] **Inheritance**: Extends `BaseGameLoop` appropriately
- [ ] **Planning Method**: Implements `_get_new_plan()` with proper round management
- [ ] **Plan Recording**: Uses `RoundManager.record_planned_moves()` for all plans
- [ ] **Move Execution**: Relies on `_execute_next_planned_move()` for move extraction
- [ ] **Round Completion**: Calls `finish_round()` only when plan queue is empty
- [ ] **Buffer Integrity**: Avoids direct manipulation of `RoundBuffer.moves`

### **Testing Requirements**
- [ ] **Headless Testing**: Works with `--max-games 1 --max-steps 50 --no-gui`
- [ ] **Log Integrity**: No duplicate moves in JSON logs
- [ ] **Round Consistency**: Round count matches prompt/response file count
- [ ] **Replay Functionality**: Generated logs work with replay systems
- [ ] **Move Completeness**: All moves appear in round data

### **Quality Assurance**
- [ ] **Error Handling**: Graceful handling of algorithm failures
- [ ] **Performance**: Reasonable execution time for planning operations
- [ ] **Memory Management**: No memory leaks in long-running games
- [ ] **Thread Safety**: Safe for concurrent execution if needed

## 🔍 **Automated Validation Tools**

### **Extension Testing Framework**
```python
def test_extension_system_integrity():
    """Comprehensive test for extension system invariant compliance"""
    
    # Initialize extension with test configuration
    extension = create_test_extension()
    
    # Run controlled test game
    result = extension.run_game(
        max_games=1,
        max_steps=50,
        headless=True
    )
    
    # Load and validate generated logs
    game_data = json.load(open(result.game_json_path))
    
    # Validate round integrity (Invariant I-1)
    rounds_data = game_data["detailed_history"]["rounds_data"]
    assert len(rounds_data) > 0, "No rounds recorded"
    
    for round_id, round_info in rounds_data.items():
        moves = round_info["moves"]
        
        # Check for duplicates (Invariant I-3)
        assert moves == list(dict.fromkeys(moves)), \
            f"Duplicate moves found in round {round_id}: {moves}"
        
        # Validate move format
        assert all(move in VALID_MOVES for move in moves), \
            f"Invalid moves in round {round_id}: {moves}"
        
        # Check planned moves recording (Invariant I-2)
        assert "planned_moves" in round_info, \
            f"Missing planned_moves in round {round_id}"
    
    # Verify round count consistency
    expected_rounds = result.round_count
    actual_rounds = len(rounds_data)
    assert expected_rounds == actual_rounds, \
        f"Round count mismatch: expected {expected_rounds}, got {actual_rounds}"
    
    print("✅ Extension passes all system integrity tests")
```

### **Performance Validation**
```python
def benchmark_extension_performance():
    """Benchmark extension performance against baseline"""
    
    baseline_time = benchmark_baseline_implementation()
    extension_time = benchmark_extension_implementation()
    
    # Extension should not be significantly slower
    performance_ratio = extension_time / baseline_time
    assert performance_ratio < 2.0, \
        f"Extension is too slow: {performance_ratio:.2f}x baseline"
    
    print(f"✅ Extension performance: {performance_ratio:.2f}x baseline")
```

## 🔗 **Integration with Extension Architecture**

### **Path Management**
Extensions must use standardized path utilities:
```python
from extensions.common.path_utils import ensure_project_root

# Ensure proper working directory
ensure_project_root()
```

### **Configuration Management**
Follow established configuration patterns:
```python
from config.game_constants import VALID_MOVES, DIRECTIONS
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
```

### **Logging Integration**
Use consistent logging patterns:
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Starting extension with algorithm: %s", algorithm_name)
```

---

**Following these guidelines ensures that extensions maintain system integrity while providing the flexibility needed for diverse algorithmic approaches. The invariants protect the core functionality while enabling innovation in planning and decision-making strategies.** 