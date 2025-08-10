# Game Data Generator

> This document aligns with `final-decision.md`. Use exact filenames when citing docs; prefer public core APIs and minimal examples. See also `core.md`, `round.md`, and `extension-skeleton.md`.

## Overview

## 🎯 **Purpose and Philosophy**

The `BaseGameData` serves as the **universal game data generator** for all Snake Game AI tasks (0-5). It implements the **Single Source of Truth (SSOT)** principle by providing a centralized, consistent approach to game state management and data tracking across all extensions.

### **Why This Generator Exists**

1. **Eliminates Code Duplication**: Before this generator, each extension implemented its own game state tracking, leading to inconsistencies and maintenance overhead
2. **Ensures Consistency**: All extensions now use identical core game state structure and tracking logic
3. **Supports Extension Development**: Provides clean hooks for extension-specific data without breaking the core state management
4. **Maintains Task-0 Compatibility**: Preserves existing functionality while enabling future extensions

## 🏗️ **Architecture and Design Patterns**

### **Template Method Pattern**
The generator uses the Template Method pattern to define the game data management workflow:

```python
class BaseGameData:
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        # 1. Core move processing (generic for all tasks)
        move = normalize_direction(move)
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        
        if apple_eaten:
            self.score += 1
            
        self.moves.append(move)
        self.current_game_moves.append(move)
        
        # 2. Extension hook for task-specific move processing
        if move not in ["INVALID_REVERSAL", "NO_PATH_FOUND"]:
            self._record_valid_step()
```

### **Strategy Pattern**
Extension-specific data tracking is handled through strategy hooks:

```python
def _record_valid_step(self) -> None:
    """Hook for recording valid steps - override in subclasses."""
    pass

def _record_step_start_time(self) -> None:
    """Hook for recording step timing - override in subclasses."""
    pass

def _record_step_end_time(self) -> None:
    """Hook for recording step timing - override in subclasses."""
    pass

def _record_invalid_reversal_step(self) -> None:
    """Hook for recording invalid reversal statistics - override in subclasses."""
    pass

def _record_no_path_found_step(self) -> None:
    """Hook for recording no path found statistics - override in subclasses."""
    pass
```

## 📊 **Core Game State Structure**

### **Universal Game State (Tasks 0-5)**
All extensions use these core game state attributes:

```python
# Core game state (generic for all tasks)
self.game_number = 0
self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
self.score = 0
self.steps = 0
self.game_over = False
self.game_end_reason = None

# Game board state (generic for all tasks)
self.snake_positions = []
self.apple_position = None
self.apple_positions = []
self.moves = []

# Move limits common across tasks
self.max_consecutive_invalid_reversals_allowed = MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED
self.max_consecutive_no_path_found_allowed = MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED

# Counter attributes for tracking consecutive moves (generic)
self.consecutive_invalid_reversals = 0
self.consecutive_no_path_found = 0
self.no_path_found_steps = 0

# Game flow control (generic for all tasks)
self.need_new_plan = True
self.planned_moves = []
self.current_direction = None
self.last_collision_type = None

# Move tracking (generic for all tasks)
self.move_index = 0
self.total_moves = 0
self.current_game_moves = []

# Apple history tracking (generic for all tasks)
self.apple_positions_history = []

# Statistics (generic) and round tracking
self.stats = BaseGameStatistics()  # Generic per-game statistics container
self.round_manager = RoundManager()  # Round tracking (optional but useful)
```

### **LLM-Specific Extensions (Task-0 only)**
Task-0 extends the base with LLM-specific attributes:

```python
# Task-0-specific move limits
self.max_consecutive_empty_moves_allowed = MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED
self.max_consecutive_something_is_wrong_allowed = MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED

# Task-0-specific consecutive counters
self.consecutive_empty_steps = 0
self.consecutive_something_is_wrong = 0

# LLM-specific statistics
self.stats = GameStatistics()  # Extends BaseGameStatistics with token/response data
```

## 🔧 **Integration with Game Logic**

The generator is integrated into the `BaseGameLogic` through the factory pattern:

```python
class BaseGameLogic:
    # Subclasses may override to inject their specialised data container.
    GAME_DATA_CLS = BaseGameData

    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        # Game state tracker for statistics
        self.game_state = self.GAME_DATA_CLS()
        self.game_state.reset()
        
        # ... other initialization
```

### **Heuristics Extension Integration**
The heuristics extension extends the base generator for algorithm-specific data:

```python
class HeuristicGameLogic(BaseGameLogic):
    # Use heuristic-specific data container
    GAME_DATA_CLS = HeuristicGameData

    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        
        # Heuristic-specific attributes
        self.agent: Optional[BFSAgent] = None
        self.algorithm_name: str = "BFS-Safe-Greedy"
```

## 🎮 **Usage in Extensions**

### **Task-0 (LLM) Usage**
Task-0 uses the full `GameData` class with LLM-specific extensions:

```python
# In core/game_data.py
class GameData(BaseGameData):
    """LLM-specific game data tracking for Task-0."""
    
    def __init__(self) -> None:
        super().__init__()
        # LLM-specific components
        self.stats = GameStatistics()
    
    def record_empty_move(self) -> None:
        """Record an empty move (LLM-specific)."""
        self.steps += 1
        self.move_index += 1
        self.total_moves += 1
        self.consecutive_empty_steps += 1
        self.moves.append("EMPTY")
        self.current_game_moves.append("EMPTY")
        # LLM-specific statistics
        self.stats.step_stats.empty += 1
```

### **Heuristics Extension Usage**
The heuristics extension extends the base generator for algorithm-specific data:

```python
# In extensions/heuristics-v0.04/game_data.py
class HeuristicGameData(BaseGameData):
    """Game data tracking for heuristic algorithms."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Heuristic-specific tracking
        self.algorithm_name: str = "BFS"
        self.path_calculations: int = 0
        self.successful_paths: int = 0
        self.failed_paths: int = 0
        
        # Track search performance
        self.total_search_time: float = 0.0
        self.nodes_explored: int = 0
        
        # Grid size (will be set by game logic)
        self.grid_size: int = 10
        
        # v0.04 Enhancement: Store move explanations for JSONL dataset generation
        self.last_move_explanation: str = ""
        self.move_explanations: list[str] = []
        self.move_metrics: list[dict] = []
```

## 🔄 **Extension Development Guide**

### **Creating Extension-Specific Game Data**

1. **Inherit from Base Class**:
```python
from core.game_data import BaseGameData

class MyExtensionGameData(BaseGameData):
    def __init__(self) -> None:
        super().__init__()
        # Add your extension's specific attributes
        self.my_algorithm = "MyAlgorithm"
        self.my_metrics = {}
    
    def _record_valid_step(self) -> None:
        # Add your extension's step recording logic
        self.my_metrics["valid_steps"] = self.my_metrics.get("valid_steps", 0) + 1
```

2. **Integrate with Game Logic**:
```python
class MyExtensionGameLogic(BaseGameLogic):
    # Use your extension's data container
    GAME_DATA_CLS = MyExtensionGameData
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        # Your extension-specific initialization
```

### **Best Practices**

1. **Maintain Core State**: Never remove or modify core game state attributes
2. **Use Extension Hooks**: Add extension-specific logic through the provided hooks
3. **Follow Naming Conventions**: Use consistent attribute names across your extension
4. **Document Custom Attributes**: Clearly document any extension-specific attributes
5. **Test State Consistency**: Ensure your game state works with existing replay tools

## 📈 **Benefits and Impact**

### **For Extension Developers**
- **90% Reduction in Boilerplate**: No need to implement game state tracking from scratch
- **Consistent State Management**: All extensions use identical core state structure
- **Easy Debugging**: Standardized state makes issues easier to identify
- **Future-Proof**: New extensions automatically benefit from core improvements

### **For Maintenance**
- **Single Point of Truth**: All game state logic centralized in one place
- **Consistent Behavior**: All extensions follow the same state management rules
- **Easy Testing**: Can test game state independently of extensions
- **Backward Compatibility**: Existing tools continue to work with new extensions

### **For Task-0**
- **Zero Impact**: Existing functionality preserved completely
- **Enhanced Reliability**: More robust game state management
- **Better Documentation**: Clear separation between core and LLM-specific state

## 🎓 **Educational Value**

The game data generator demonstrates several important software engineering principles:

1. **Template Method Pattern**: Defines state management workflow while allowing customization
2. **Strategy Pattern**: Enables different state tracking strategies for different extensions
3. **Single Responsibility**: Focused solely on game state management
4. **Open/Closed Principle**: Open for extension, closed for modification
5. **DRY Principle**: Eliminates code duplication across extensions
6. **Factory Pattern**: Pluggable data containers through `GAME_DATA_CLS`

## 🔗 **Related Documentation**

- **`game_summary_generator.md`**: Universal summary generation system
- **`core.md`**: Core architecture and base classes
- **`single-source-of-truth.md`**: SSOT principles and implementation
- **`factory-design-pattern.md`**: Factory pattern usage in the codebase
- **`final-decision.md`**: SUPREME_RULES and architectural standards

---

**The game data generator exemplifies the project's commitment to elegant, maintainable architecture that serves both current needs and future extensions while maintaining strict adherence to `final-decision.md` SUPREME_RULES.** 