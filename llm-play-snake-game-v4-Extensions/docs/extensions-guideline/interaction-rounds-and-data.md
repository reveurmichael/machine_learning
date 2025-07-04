# Interaction Between Rounds and Data Management in Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and explains the data flow architecture for extensions.

> **See also:** `core.md`, `heuristics-as-foundation.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Pipeline Architecture**

The Snake Game AI project uses a **pipeline architecture** where game execution flows through distinct phases: **Planning → Execution → Recording → Analysis**. This architecture ensures data integrity, enables debugging, and provides a consistent foundation for all AI approaches.

### **Pipeline Flow Overview**
```
Game State → Planning Phase → Execution Phase → Recording Phase → Analysis Phase
     ↓              ↓              ↓              ↓              ↓
  Snapshot    Planned Moves   Actual Moves   Round Data    Datasets
```

## 🏗️ **Heuristics-v0.04 Implementation**

### **1. Round Management Architecture**

#### **Round Structure**
```python
# Each round represents one decision-making cycle
Round {
    number: int,              # Sequential round counter
    apple_position: [x, y],   # Apple location for this round
    planned_moves: [str],     # What the agent planned to do
    moves: [str],             # What actually happened
    game_state: dict          # Snapshot of game state
}
```

#### **Round Manager Responsibilities**
```python
class HeuristicRoundManager(RoundManager):
    """
    Manages the round-by-round data flow for heuristic algorithms.
    
    Key Responsibilities:
    - Buffer management (temporary storage during round)
    - Data persistence (flushing buffer to permanent storage)
    - SSOT compliance (single source of truth for round data)
    """
    
    def sync_round_data(self) -> None:
        """Flush round buffer to permanent storage before starting new round."""
        # This ensures moves are recorded before buffer is replaced
        current_round_dict["moves"] = list(self.round_buffer.moves)
        self.round_buffer.moves.clear()  # Reset for next round
```

### **2. Data Flow Pipeline**

#### **Step 1: Planning Phase**
```python
# In HeuristicGameLogic.get_next_planned_move()
def get_next_planned_move(self) -> str:
    """Plan the next move using heuristic algorithm."""
    if not self.planned_moves:
        self.planned_moves = self.plan_next_moves()  # Agent decides
    
    move = self.planned_moves.pop(0)  # Get next planned move
    return move
```

#### **Step 2: Execution Phase**
```python
# In HeuristicGameManager._run_single_game()
move = self.game.get_next_planned_move()  # Get planned move
self.game.make_move(move)  # Execute the move
```

#### **Step 3: Recording Phase**
```python
# In BaseGameData.record_move()
def record_move(self, move: str, apple_eaten: bool = False) -> None:
    """Record the executed move in game state and round buffer."""
    self.moves.append(move)  # Add to game history
    self.steps += 1          # Increment step counter
    
    # Add to round buffer for round-by-round tracking
    if hasattr(self, "round_manager") and self.round_manager:
        self.round_manager.round_buffer.add_move(move)
```

#### **Step 4: Flushing Phase**
```python
# In HeuristicGameManager._run_single_game()
# Sync previous round's data before starting new round
if self.round_count > 0:
    self.game.game_state.round_manager.sync_round_data()

# Start new round
self.start_new_round(f"{self.algorithm_name} pathfinding")
```

### **3. Key Data Structures**

#### **Game State Snapshot**
```python
def get_state_snapshot(self) -> dict:
    """Capture current game state for planning and recording."""
    return {
        "head_position": self.head_position,
        "snake_positions": self.snake_positions.tolist(),
        "apple_position": self.apple_position,
        "grid_size": self.grid_size,
        "score": self.score,
        "steps": self.steps,
        "current_direction": self._get_current_direction_key(),
        "snake_length": len(self.snake_positions)
    }
```

#### **Round Data Structure**
```python
rounds_data = {
    "1": {
        "round": 1,
        "apple_position": [6, 3],
        "planned_moves": ["DOWN"],    # Planning phase
        "moves": ["DOWN"],            # Execution phase
        "game_state": {...}           # State snapshot
    },
    "2": {
        "round": 2,
        "apple_position": [6, 3],
        "planned_moves": ["RIGHT"],
        "moves": ["RIGHT"],
        "game_state": {...}
    }
}
```

## 🔄 **Pipeline Flow in Detail**

### **Complete Round Cycle**
```python
# 1. Start new round
self.start_new_round("BFS pathfinding")
# Creates new round buffer, increments round counter

# 2. Plan move
move = self.game.get_next_planned_move()
# Agent analyzes current state, decides on move

# 3. Execute move
self.game.make_move(move)
# Applies move to game state, calls record_move()

# 4. Record move
self.game_state.record_move(move, apple_eaten)
# Adds move to round buffer and game history

# 5. Record game state
self.round_manager.record_game_state(self.game.get_state_snapshot())
# Captures post-move state for this round

# 6. Sync round data (before next round)
self.round_manager.sync_round_data()
# Flushes buffer to permanent rounds_data storage
```

### **Data Integrity Guarantees**
- **SSOT Compliance**: Each move is recorded exactly once in the authoritative round buffer
- **Temporal Consistency**: Game state snapshots are captured immediately after move execution
- **Pipeline Separation**: Planning and execution phases are clearly separated in the data structure

## 🎯 **Similarities with Task-0**

### **Shared Architecture**
```python
# Both Task-0 and Heuristics use the same pipeline:
# 1. Planning Phase
planned_moves = agent.plan_next_moves()  # LLM or heuristic planning

# 2. Execution Phase  
move = get_next_planned_move()
make_move(move)

# 3. Recording Phase
record_move(move)

# 4. Round Management
sync_round_data()
```

### **Data Structure Compatibility**
```json
{
  "detailed_history": {
    "rounds_data": {
      "1": {
        "round": 1,
        "apple_position": [x, y],
        "planned_moves": ["UP"],    // Planning phase
        "moves": ["UP"]             // Execution phase
      }
    }
  }
}
```

### **Key Differences**
- **Task-0**: LLM may plan multiple moves, execution might differ from plan
- **Heuristics**: Planning and execution are identical (deterministic algorithms)

## 🚀 **Future Extensions: ML/DL/RL/Supervised**

### **Reinforcement Learning (RL)**
```python
# Planning Phase: Policy network decides action
action = policy_network.predict(state)
planned_moves = [action_to_move(action)]

# Execution Phase: Environment applies action
make_move(planned_moves[0])

# Recording Phase: Same as heuristics
record_move(move)
```

### **Supervised Learning**
```python
# Planning Phase: Neural network predicts move
prediction = neural_network.predict(state_features)
planned_moves = [prediction_to_move(prediction)]

# Execution Phase: Apply predicted move
make_move(planned_moves[0])

# Recording Phase: Same pipeline
record_move(move)
```

### **Deep Learning (DL)**
```python
# Planning Phase: Complex neural network analysis
features = extract_deep_features(game_state)
planned_moves = deep_network.predict(features)

# Execution Phase: Apply deep learning decision
make_move(planned_moves[0])

# Recording Phase: Consistent recording
record_move(move)
```

## 📊 **Limits Manager Integration**

### **Automatic Limit Tracking**
```python
# Limits manager automatically tracks consecutive events
class ConsecutiveLimitsManager:
    def record_move(self, move: str) -> bool:
        """Track move and check if limits exceeded."""
        if move == "NO_PATH_FOUND":
            return self._handle_limit_occurrence(LimitType.NO_PATH_FOUND)
        elif move == "INVALID_REVERSAL":
            return self._handle_limit_occurrence(LimitType.INVALID_REVERSALS)
        # ... other move types
```

### **Extension Benefits**
- **Automatic Management**: Extensions inherit sophisticated limit tracking
- **Configurable Behavior**: Custom limits for different AI approaches
- **Debugging Support**: Detailed limit status for troubleshooting

## 🎓 **Educational Value**

### **Pipeline Understanding**
- **Clear Separation**: Planning vs execution phases are distinct
- **Data Flow**: Understanding how information flows through the system
- **Debugging**: Easy to trace where decisions are made and executed

### **Extension Development**
- **Consistent Interface**: All extensions use the same round management
- **Modular Design**: Easy to swap different AI approaches
- **Data Integrity**: SSOT compliance ensures reliable datasets

## 🔧 **Implementation Guidelines for Future Extensions**

### **Required Components**
```python
class YourExtensionGameManager(BaseGameManager):
    def _run_single_game(self) -> float:
        while not self.game.game_over:
            # 1. Sync previous round
            if self.round_count > 0:
                self.game.game_state.round_manager.sync_round_data()
            
            # 2. Start new round
            self.start_new_round("Your AI planning")
            
            # 3. Plan move
            move = self.game.get_next_planned_move()
            
            # 4. Execute move
            self.game.make_move(move)
            
            # 5. Record state
            self.game.game_state.round_manager.record_game_state(
                self.game.get_state_snapshot()
            )
```

### **Data Structure Requirements**
- **Round Data**: Must include `planned_moves` and `moves` for pipeline clarity
- **Game State**: Must provide `get_state_snapshot()` for planning and recording
- **Move Recording**: Must call `record_move()` after each move execution
- **Round Syncing**: Must call `sync_round_data()` before starting new rounds

### **SSOT Compliance Checklist**
- [ ] Each move recorded exactly once in round buffer
- [ ] Round data flushed before starting new round
- [ ] Game state snapshots captured after move execution
- [ ] Planning and execution phases clearly separated in data
- [ ] No duplicate or missing moves in final output

## 🏆 **Conclusion**

The heuristics-v0.04 extension demonstrates the **canonical pipeline architecture** that all future extensions should follow:

1. **Planning Phase**: Agent analyzes state and decides on moves
2. **Execution Phase**: Moves are applied to game state
3. **Recording Phase**: Moves and state are captured in round data
4. **Analysis Phase**: Data is used for datasets and debugging

This architecture ensures **data integrity**, **debugging capability**, and **extensibility** for all AI approaches while maintaining **Task-0 compatibility** and **SSOT compliance**.

---

**This tutorial provides the foundation for implementing any AI extension while maintaining the robust, scalable architecture established by the Snake Game AI project.**
