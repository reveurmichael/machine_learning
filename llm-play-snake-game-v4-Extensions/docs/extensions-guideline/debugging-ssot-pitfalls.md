# Debugging SSOT Pitfalls: Common Issues in Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and provides debugging guidance for SSOT compliance issues.

> **See also:** `coordinate-system.md`, `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Overview: Why SSOT Debugging Matters**

Single Source of Truth (SSOT) compliance is **critical** for dataset quality in Snake Game AI extensions. When SSOT validation fails, the generated datasets become unsuitable for LLM fine-tuning, rendering hours of computation worthless.

This tutorial covers the most common SSOT pitfalls encountered during extension development, with real-world examples from our investigation of the heuristics-v0.04 post-apple indexing issue.

## 🚨 **Critical Pitfall #1: Coordinate System Mismatch**

### **The Problem**
The most insidious SSOT issue is **coordinate system inconsistency** between different parts of the codebase.

### **Root Cause**
```python
# ❌ WRONG: Inconsistent head position tracking
class BaseGameLogic:
    def __init__(self):
        self.snake_positions = np.array([[4, 4]])  # [0] is HEAD
        self.head_position = self.snake_positions[-1]  # [-1] is TAIL!
    
    def get_state_snapshot(self):
        return {
            "head_position": self.head_position.tolist(),  # Points to TAIL
            "snake_positions": self.snake_positions.tolist(),  # [0] is HEAD
        }
```

### **The Issue**
- **`snake_positions[0]`** = HEAD (first element)
- **`snake_positions[-1]`** = TAIL (last element)  
- **`self.head_position`** was pointing to `snake_positions[-1]` (TAIL!)
- **`_format_prompt`** uses `snake_positions[0]` (HEAD)

This created **inconsistent data** where the prompt showed one head position and the completion expected another.

### **The Fix**
```python
# ✅ CORRECT: Consistent head position tracking
def get_state_snapshot(self) -> dict:
    # SSOT Fix: Use snake_positions[0] as the head position
    # According to coordinate-system.md: snake_positions[0] is HEAD, [-1] is TAIL
    head_pos = self.snake_positions[0].tolist() if len(self.snake_positions) > 0 else [0, 0]
    
    return {
        "head_position": head_pos,  # Now consistent with snake_positions[0]
        "snake_positions": self.snake_positions.tolist(),
        # ... rest of state
    }
```

### **Debugging Strategy**
```python
# Add debug output to check consistency
def _format_prompt(self, game_state):
    head_pos = game_state.get('head_position', [0, 0])
    snake_positions = game_state.get('snake_positions', [])
    print(f"[DEBUG] head_pos={head_pos}, snake_positions[0]={snake_positions[0] if snake_positions else 'None'}")
    
    # If these don't match, you have a coordinate system issue!
    if snake_positions and head_pos != snake_positions[0]:
        print(f"[ERROR] Head position mismatch: {head_pos} vs {snake_positions[0]}")
```

## 🚨 **Critical Pitfall #2: Timing Issues in State Recording**

### **The Problem**
Recording game state at the wrong time creates **temporal inconsistencies**.

### **Common Mistakes**
```python
# ❌ WRONG: Recording state after move execution
def make_move(self, direction):
    # Apply move first
    self.apply_move(direction)
    
    # Record state AFTER move (wrong!)
    recorded_state = self.get_state_snapshot()
    self.round_manager.record_game_state(recorded_state)
```

### **The Issue**
- State recorded **after** move execution
- Prompt shows **post-move** state
- Completion expects **pre-move** state
- **SSOT violation**: Prompt and completion use different states

### **The Fix**
```python
# ✅ CORRECT: Record state BEFORE move execution
def make_move(self, direction):
    # Record PRE-MOVE state for SSOT compliance
    recorded_state = self.get_state_snapshot()
    self.round_manager.record_game_state(recorded_state)
    
    # Apply move AFTER recording
    self.apply_move(direction)
```

### **Debugging Strategy**
```python
# Add timing debug output
def _run_single_game(self):
    while not self.game.game_over:
        # Debug: Check state before recording
        pre_state = self.game.get_state_snapshot()
        print(f"[DEBUG] Pre-move state: head={pre_state['head_position']}")
        
        # Record state BEFORE move
        self.game.game_state.round_manager.record_game_state(pre_state)
        
        # Get and apply move
        move = self.game.get_next_planned_move()
        self.game.make_move(move)
        
        # Debug: Check state after move
        post_state = self.game.get_state_snapshot()
        print(f"[DEBUG] Post-move state: head={post_state['head_position']}")
```

## 🚨 **Critical Pitfall #3: Indexing Errors in Dataset Generation**

### **The Problem**
Incorrect indexing when mapping moves to game states creates **alignment issues**.

### **Common Mistakes**
```python
# ❌ WRONG: Incorrect indexing
for i, move in enumerate(moves_history):
    # Wrong: Using i+1 for game state (off-by-one error)
    game_state = dataset_game_states[i + 1]
    record = {"move": move, "game_state": game_state}
```

### **The Issue**
- **Moves**: 0-indexed array `[0, 1, 2, 3, ...]`
- **Game States**: 1-indexed keys `{1, 2, 3, 4, ...}`
- **Apple consumption**: Changes the indexing pattern
- **Result**: Misaligned moves and states

### **The Fix**
```python
# ✅ CORRECT: Proper indexing with apple consumption handling
for i in range(len(moves_history)):
    round_key = i + 2  # States start from round 2 (round 1 is empty)
    if round_key not in dataset_game_states:
        raise RuntimeError(f"Missing game state for round {round_key}")
    
    game_state = dataset_game_states[round_key]
    move = moves_history[i]
    record = {"move": move, "game_state": game_state}
```

### **Debugging Strategy**
```python
# Add indexing debug output
for i in range(len(moves_history)):
    round_key = i + 2
    game_state = dataset_game_states[round_key]
    move = moves_history[i]
    
    print(f"[DEBUG] Entry {i+1}: move={move}, round={round_key}, head={game_state['head_position']}")
    
    # Validate the mapping makes sense
    if i > 0:
        prev_state = dataset_game_states[round_key - 1]
        expected_head = self._calculate_next_position(prev_state['head_position'], moves_history[i-1])
        actual_head = game_state['head_position']
        
        if expected_head != actual_head:
            print(f"[ERROR] Head position mismatch at entry {i+1}")
            print(f"  Expected: {expected_head} (from {prev_state['head_position']} + {moves_history[i-1]})")
            print(f"  Actual: {actual_head}")
```

## 🚨 **Critical Pitfall #4: Apple Position Format Inconsistency**

### **The Problem**
Apple positions can be stored in different formats, causing **serialization issues**.

### **Common Mistakes**
```python
# ❌ WRONG: Assuming consistent apple position format
def _format_prompt(self, game_state):
    apple_position = game_state.get('apple_position', [])
    apple_x, apple_y = apple_position[0], apple_position[1]  # Assumes list format
```

### **The Issue**
- **Sometimes**: `apple_position = [x, y]` (list format)
- **Sometimes**: `apple_position = {"x": x, "y": y}` (dict format)
- **Result**: `TypeError` or incorrect coordinates

### **The Fix**
```python
# ✅ CORRECT: Handle both formats
def _format_prompt(self, game_state):
    apple_position = game_state.get('apple_position', [])
    
    # Handle both dict and list formats
    if isinstance(apple_position, dict):
        apple_x, apple_y = apple_position.get('x', 0), apple_position.get('y', 0)
    else:
        apple_x, apple_y = apple_position[0], apple_position[1]
```

### **Debugging Strategy**
```python
# Add format debug output
def _format_prompt(self, game_state):
    apple_position = game_state.get('apple_position', [])
    print(f"[DEBUG] Apple position: {apple_position}, type: {type(apple_position)}")
    
    try:
        if isinstance(apple_position, dict):
            apple_x, apple_y = apple_position.get('x', 0), apple_position.get('y', 0)
        else:
            apple_x, apple_y = apple_position[0], apple_position[1]
    except (KeyError, TypeError, IndexError) as e:
        print(f"[ERROR] Apple position parsing failed: {e}")
        apple_x, apple_y = 0, 0
```

## 🚨 **Critical Pitfall #5: Numpy Type Serialization Issues**

### **The Problem**
Numpy types cause **JSON serialization errors**.

### **Common Mistakes**
```python
# ❌ WRONG: Direct numpy serialization
import json
import numpy as np

data = {
    "head_position": np.array([4, 4]),  # numpy.int64
    "score": np.int64(5),               # numpy.int64
}

json.dumps(data)  # TypeError: Object of type numpy.int64 is not JSON serializable
```

### **The Fix**
```python
# ✅ CORRECT: Handle numpy types
def _json_serializer(self, obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Use in JSON serialization
json.dumps(data, default=self._json_serializer)
```

## 🔍 **SSOT Validation Strategy**

### **Step 1: Run SSOT Validation**
```bash
python validate_ssot.py "path/to/your/dataset.jsonl"
```

### **Step 2: Analyze Error Patterns**
```python
# Common error patterns and their causes:
# - "Head position mismatch" → Coordinate system issue
# - "Apple position mismatch" → Format inconsistency
# - "Move not found" → Indexing error
# - "Game state missing" → Timing issue
```

### **Step 3: Add Debug Output**
```python
# Add comprehensive debug output
def _extract_jsonl_record(self, record):
    game_state = record.get('game_state', {})
    move = record.get('move', 'UNKNOWN')
    
    # Debug: Check consistency
    head_pos = game_state.get('head_position', [0, 0])
    snake_positions = game_state.get('snake_positions', [])
    
    if snake_positions and head_pos != snake_positions[0]:
        print(f"[ERROR] Head position mismatch: {head_pos} vs {snake_positions[0]}")
    
    # Continue with normal processing...
```

## 🛠️ **Debugging Checklist**

### **Before Running Extension**
- [ ] **Coordinate System**: Verify `snake_positions[0]` is HEAD, `[-1]` is TAIL
- [ ] **State Recording**: Ensure state recorded BEFORE move execution
- [ ] **Indexing**: Check move-to-state mapping logic
- [ ] **Apple Format**: Handle both dict and list formats
- [ ] **Numpy Types**: Add JSON serialization handler

### **After Running Extension**
- [ ] **SSOT Validation**: Run `validate_ssot.py` on generated dataset
- [ ] **Error Analysis**: Identify patterns in validation failures
- [ ] **Debug Output**: Add temporary debug prints to trace issues
- [ ] **Manual Inspection**: Check first few entries manually

### **Common Debug Commands**
```bash
# Validate dataset
python validate_ssot.py "path/to/dataset.jsonl"

# Check specific entries
jq '.prompt' dataset.jsonl | head -5
jq '.completion' dataset.jsonl | head -5

# Compare head positions
for i in {1..5}; do
  echo "Entry $i:"
  sed -n "${i}p" dataset.jsonl | jq -r '.prompt' | grep "Head Position"
  sed -n "${i}p" dataset.jsonl | jq -r '.completion' | jq '.metrics.head_position'
  echo "---"
done
```

## 🎯 **Best Practices**

### **1. Always Use SSOT Validation**
```python
# Add validation to your extension
def validate_generated_dataset(self, dataset_path):
    """Validate generated dataset for SSOT compliance."""
    result = subprocess.run([
        "python", "validate_ssot.py", dataset_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] SSOT validation failed: {result.stderr}")
        return False
    
    print("[SUCCESS] SSOT validation passed")
    return True
```

### **2. Add Comprehensive Debug Output**
```python
# Add debug mode to your extension
def __init__(self, debug_mode=False):
    self.debug_mode = debug_mode

def debug_print(self, message):
    if self.debug_mode:
        print(f"[DEBUG] {message}")
```

### **3. Test with Small Datasets First**
```python
# Test with limited steps before full runs
python main.py --algorithm BFS --max-steps 5 --max-games 1
```

### **4. Document Your Coordinate System**
```python
# Always document your coordinate system assumptions
"""
Coordinate System:
- snake_positions[0] = HEAD (first element)
- snake_positions[-1] = TAIL (last element)
- Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1
- Origin: (0,0) at bottom-left, (N-1,N-1) at top-right
"""
```

## 🔗 **Related Documentation**

- **`coordinate-system.md`**: Official coordinate system specification
- **`core.md`**: Base class architecture and inheritance patterns
- **`final-decision-10.md`**: SUPREME_RULES governance system
- **`project-structure-plan.md`**: Overall project architecture

---

**This debugging guide ensures that your Snake Game AI extensions generate high-quality, SSOT-compliant datasets suitable for LLM fine-tuning. Always validate your datasets before using them for training!** 