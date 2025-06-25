# Utility Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Utility organization follows the universal vs. task-specific separation principle established in the GOOD_RULES.

## 🎯 **Utility Philosophy**

The utility architecture provides a clean separation between universal game mechanics and task-specific functionality. This enables all extensions to leverage shared infrastructure while maintaining clear boundaries between different AI approaches.


### **✅ Universal (Tasks 0-5) Utilities - Already Generic**

These utilities contain **zero LLM-specific dependencies** and work for any algorithm:

#### **1. `board_utils.py` - Pure Game Mechanics ✅**
```python
# ✅ Completely task-agnostic functions
def generate_random_apple(snake_positions, grid_size) -> NDArray:
    """Works for ANY task - BFS, RL, LLM, etc."""
    
def update_board_array(board, snake_positions, apple_position, board_info):
    """Universal board rendering for any visualization."""
    
def is_position_valid(position, grid_size) -> bool:
    """Generic position validation for all algorithms."""
    
def get_empty_positions(snake_positions, grid_size) -> List[Position]:
    """Universal empty space detection for pathfinding, apple placement."""
```

**How Tasks 1-5 Use This:**
- **Task-1 (Heuristics)**: BFS pathfinding uses `generate_random_apple()` for new apple placement
- **Task-2 (Supervised)**: Neural networks use `update_board_array()` for training visualization  
- **Task-3 (RL)**: DQN agents use both functions for environment state management
- **Task-4/5 (LLM variants)**: Same as Task-0, inherits seamlessly

#### **2. `collision_utils.py` - Universal Physics ✅**
```python
# ✅ Generic collision detection for any algorithm
def check_collision(position, snake_positions, grid_size, is_eating_apple_flag):
    """Returns (wall_collision, body_collision) - works universally."""
    
def check_wall_collision(head_position, grid_size) -> bool:
    """Boundary checking for any task."""
    
def check_body_collision(head_position, snake_body, is_apple_eaten) -> bool:
    """Self-collision detection with growth mechanics."""
    
def check_apple_collision(head_position, apple_position) -> bool:
    """Apple consumption detection."""
```

**How Tasks 1-5 Use This:**
- **Task-1**: BFS algorithms call this to validate path legality
- **Task-2**: Supervised models use this for loss function constraints
- **Task-3**: RL reward functions use this for termination detection
- **All Tasks**: Same exact collision rules, guaranteed consistency

#### **3. `moves_utils.py` - Direction Processing ✅**
```python
# ✅ Pure move processing - no LLM dependencies
def normalize_direction(move: str) -> str:
    """Convert 'left' -> 'LEFT' for any algorithm."""
    
def is_reverse(dir_a: str, dir_b: str) -> bool:
    """Check if two directions are opposites."""
    
def get_relative_apple_direction_text(head_pos, apple_pos) -> str:
    """Human-readable spatial relationship description."""
```

**How Tasks 1-5 Use This:**
- **Task-1**: Heuristic algorithms use `is_reverse()` for move validation
- **Task-2**: Training data preprocessing with `normalize_direction()`
- **Task-3**: RL action space normalization via these functions
- **All Tasks**: Consistent move representation across algorithms

#### **4. `json_utils.py` - Parsing Infrastructure ✅**
```python
# ✅ Generic JSON processing (not LLM-specific despite comments)
def validate_json_format(data: Any) -> Tuple[bool, str]:
    """Validate move sequences for any algorithm output."""
    
def safe_json_load(filepath: str) -> Optional[Dict]:
    """Safe JSON loading with error handling."""
    
def safe_json_save(data: Any, filepath: str) -> bool:
    """Safe JSON saving with NumPy array support."""
```

**How Tasks 1-5 Use This:**
- **Task-4/5**: LLM outputs use identical parsing pipeline

#### **5. `web_utils.py` - Universal Web Infrastructure ✅**
```python
# ✅ Generic web utilities (work for any task)
def build_state_dict(snake_positions, apple_position, score, steps, grid_size, *, extra=None):
    """Universal JSON state for any algorithm."""
    
def build_color_map() -> Dict[str, Tuple[int, int, int]]:
    """Universal color scheme for any visualization."""
    
def translate_end_reason(code: Optional[str]) -> Optional[str]:
    """Generic game termination mapping."""
    
def to_list(obj) -> list | object:
    """NumPy array serialization for any task."""
```

**How Tasks 1-5 Use This:**
- **All Tasks**: Universal web API format

---

### **❌ Task-0 Specific Utilities - Correctly Isolated**

These utilities are **correctly marked as Task-0 only** and **properly separated**:

#### **1. `continuation_utils.py` - LLM Session Recovery ❌**
```python
"""This module is Task0 specific. So no need for BaseGameManager.
IMPORTANT:
- We will have continuation mode for only Task0.
- For Task1, Task2, Task3, Task4, Task5, we will NOT have continuation mode.
"""

def setup_continuation_session(game_manager: "GameManager"):  # ← Task-0 only
def continue_from_directory(game_manager_class: "type[GameManager]"):
def handle_continuation_game_state(game_manager: "GameManager"):
```

#### **2. `initialization_utils.py` - LLM Client Setup ❌**
```python
# This function is Task0 specific, because LLM is involved
def setup_llm_clients(game_manager: "GameManager") -> None:
def setup_log_directories(game_manager: "GameManager") -> None:
```

#### **3. `session_utils.py` - Task-0 Script Launchers ❌**
```python
"""This whole module is Task0 specific."""
def run_main_web(), run_replay(), continue_game():  # All Task-0 specific
```

---

## **🎯 How Tasks 1-5 Will Use Utils - Detailed Examples**

### **Task-1 (Heuristics) Integration:**
```python
from utils.board_utils import generate_random_apple, update_board_array, get_empty_positions
from utils.collision_utils import check_collision, check_wall_collision, check_body_collision
from utils.moves_utils import normalize_direction, is_reverse
from utils.json_utils import validate_json_format, safe_json_save
from utils.web_utils import build_state_dict, build_color_map

```

### **Task-3 (RL) Integration:**
```python
from utils.board_utils import update_board_array, is_position_valid
from utils.collision_utils import check_collision
from utils.moves_utils import normalize_direction
from utils.json_utils import validate_json_format  
from utils.web_utils import build_state_dict

```

### **Task-2 (Supervised Learning) Integration:**
```python
from utils.board_utils import get_empty_positions, update_board_array
from utils.moves_utils import normalize_direction, normalize_directions
from utils.json_utils import safe_json_load, safe_json_save
```

---

## **🔄 Inter-Class Dependencies - Perfectly Managed**

### **✅ Correct Dependency Direction:**
```python
# ✅ Utils → Core (allowed)
# utils/board_utils.py - NO core imports (pure functions)
# utils/collision_utils.py - NO core imports (pure functions)
# utils/moves_utils.py - NO core imports (pure functions)

# ✅ Utils → Config (allowed - shared constants)
from config.game_constants import END_REASON_MAP, VALID_MOVES
from config.ui_constants import COLORS

# ✅ Task-specific utils use concrete classes  
def setup_llm_clients(game_manager: "GameManager"):  # Task-0 specific

# ✅ No circular dependencies
❌ No core/ → utils/ imports found (perfect separation)
```

### **✅ Universal Constants Usage:**
```python
# ✅ Generic utilities use universal constants, across task0-5, and across all extensions.
from config.game_constants import END_REASON_MAP       # Game termination (all tasks)
from config.ui_constants import COLORS                 # Visualization (all tasks)

# ✅ LLM-centric extensions (agentic-llms, vision-language-model, etc.) and Task-0 may import LLM constants
from config.llm_constants import TEMPERATURE, MAX_TOKENS  # LLM-only
```

---

## **📊 Naming Convention Compliance - Perfect**

### **✅ All Files Follow `*_utils.py` Pattern:**
```
✅ board_utils.py        - Board manipulation utilities
✅ collision_utils.py    - Collision detection utilities  
✅ moves_utils.py        - Movement processing utilities
✅ json_utils.py         - JSON serialization utilities
✅ web_utils.py          - Web interface utilities
✅ network_utils.py      - Network communication utilities
✅ text_utils.py         - Text formatting utilities
✅ path_utils.py         - File path utilities
✅ seed_utils.py         - Random seed utilities
✅ continuation_utils.py - Game continuation utilities (Task-0)
✅ session_utils.py      - Session management utilities (Task-0)
✅ initialization_utils.py - Setup utilities (Task-0)
```

---

## **🏆 Summary: Utils Architecture is Already Perfect**

### **✅ What's Already Working:**

1. **Perfect SOLID Compliance**: Utils follow dependency inversion - depend on abstractions
2. **Clean Task Separation**: LLM-specific vs generic utilities clearly marked and separated
3. **Universal Compatibility**: Core utilities work with any algorithm type immediately
4. **Zero Circular Dependencies**: Clean import hierarchy maintained
5. **Consistent File Patterns**: All utilities follow `*_utils.py` naming convention
6. **Pure Function Design**: Board, collision, and move utilities are stateless and reusable
7. **Type Safety**: Comprehensive type hints with NumPy array support

### **✅ Inter-Class Dependencies Verified:**
- ✅ `utils/board_utils.py` → No dependencies (pure functions)
- ✅ `utils/collision_utils.py` → No dependencies (pure functions)  
- ✅ `utils/moves_utils.py` → No dependencies (pure functions)
- ✅ `utils/json_utils.py` → Config only (universal constants)
- ✅ `utils/web_utils.py` → Config only (universal constants)
- ✅ `utils/continuation_utils.py` → Core GameManager (Task-0 only)
- ✅ `utils/session_utils.py` → Task-0 scripts only
- ✅ `utils/initialization_utils.py` → LLM modules (Task-0 only)

### **✅ Future Tasks Integration:**
```python
# ANY task gets utils for free:
from utils.board_utils import generate_random_apple, update_board_array
from utils.collision_utils import check_collision
from utils.moves_utils import normalize_direction, is_reverse
from utils.json_utils import validate_json_format, safe_json_save
from utils.web_utils import build_state_dict, build_color_map

class AnyTaskAgent(BaseAgent):
    pass
# Works immediately with zero modifications to utils!
```


### **Core Design Principles**
- **Universal Mechanics**: Core game utilities work for all tasks and extensions
- **Task-Specific Isolation**: LLM-specific utilities remain in Task-0 only
- **Dependency Direction**: Utils → Config (allowed), Core → Utils (forbidden)
- **Pure Functions**: Stateless utilities for maximum reusability

## 🏗️ **Utility Organization**

### **Universal Utilities (ROOT/utils/)**
These utilities serve ALL tasks and extensions without modification:

**Game Mechanics**:
- `board_utils.py`: Grid management, apple placement, position validation
- `collision_utils.py`: Wall/body collision detection, game physics
- `moves_utils.py`: Direction processing, move validation

**Data & Web**:
- `json_utils.py`: Safe JSON operations, data validation
- `web_utils.py`: Universal web state representation
- `path_utils.py`: Cross-platform path management

**Infrastructure**:
- `seed_utils.py`: Random number generation management
- `text_utils.py`: Common text formatting operations

### **Task-0 Specific Utilities (ROOT/utils/)**
These utilities are isolated to LLM-specific functionality:

**LLM Infrastructure**:
- `continuation_utils.py`: Session recovery and continuation
- `initialization_utils.py`: LLM client setup and configuration
- `session_utils.py`: Task-0 script launchers and session management

### **Extension-Specific Utilities (extensions/common/)**
Cross-extension utilities that don't belong in the universal ROOT/utils/:

**Configuration & Validation**:
- `config/`: Extension-specific configuration constants
- `validation/`: Schema validation and data integrity checks
- `path_utils.py`: Extension-specific path management patterns

## 🚀 **Integration Patterns**

### **Universal Utility Usage**
All extensions leverage the same core utilities:

```python
# Heuristics Extension
from utils.board_utils import generate_random_apple, get_empty_positions
from utils.collision_utils import check_collision, check_wall_collision
from utils.moves_utils import normalize_direction, is_reverse

class BFSAgent(BaseAgent):
    def plan_move(self, game_state):
        # Uses same collision detection as Task-0
        valid_moves = []
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            pos = self.calculate_next_position(direction)
            wall_hit, body_hit = check_collision(pos, self.snake_body, self.grid_size, False)
            if not (wall_hit or body_hit):
                valid_moves.append(normalize_direction(direction))
```

```python
# Supervised Learning Extension
from utils.json_utils import safe_json_save, validate_json_format
from utils.web_utils import build_state_dict

class SupervisedGameManager(BaseGameManager):
    def save_training_data(self):
        # Uses same JSON utilities as Task-0
        state_dict = build_state_dict(
            self.snake_positions, self.apple_position, 
            self.score, self.steps, self.grid_size
        )
        safe_json_save(state_dict, self.training_data_path)
```

### **Configuration Integration**
Following Final Decision 2, utilities access universal constants:

```python
# Universal utilities use universal constants
from config.game_constants import VALID_MOVES, DIRECTIONS, END_REASON_MAP
from config.ui_constants import COLORS, GRID_SIZE

# ✅ Allowed in Task-0 and LLM-centric extensions; 🚫 Do not use in other extension types
from config.llm_constants import TEMPERATURE, MAX_TOKENS  # LLM-specific
from config.prompt_templates import SYSTEM_PROMPT         # Task-0 only
```



## 🎓 **Educational and Design Benefits**

### **Consistency Across Extensions**
Universal utilities ensure all extensions behave identically:
- **Collision Detection**: Same physics rules across all AI approaches
- **Board Management**: Consistent grid representation and manipulation
- **Move Processing**: Uniform direction handling and validation
- **Web Interface**: Identical state representation for visualization

### **Rapid Extension Development**
Extensions inherit rich functionality immediately:
- **No Reimplementation**: Core game mechanics work out-of-the-box
- **Focus on Algorithms**: Developers focus on AI logic, not infrastructure
- **Guaranteed Compatibility**: All extensions interoperate seamlessly
- **Cross-Extension Replay**: Universal utilities enable cross-algorithm replay

### **Maintenance Benefits**
Centralized utilities simplify system maintenance:
- **Single Source of Truth**: Game rules defined once, used everywhere
- **Bug Fixes Propagate**: Fix once, benefits all extensions
- **Feature Enhancement**: Improvements automatically available to all tasks
- **Testing Efficiency**: Test utilities once, confidence across all extensions

## 🔧 **Dependency Management**

### **Clean Dependency Hierarchy**
The utility architecture maintains clear dependency direction:

```python
# ✅ Allowed Dependencies
utils/board_utils.py    → config/game_constants.py
utils/web_utils.py      → config/ui_constants.py
core/game_logic.py      → utils/collision_utils.py

# ❌ Forbidden Dependencies (circular)
utils/board_utils.py    ← core/game_logic.py  # Never happens
```

### **Task-Specific Isolation**
Task-0 utilities correctly depend on concrete classes:

```python
# Task-0 specific utilities can use Task-0 classes
def setup_llm_clients(game_manager: "GameManager"):  # Task-0 concrete class
    """Setup LLM clients for Task-0 only"""

# Universal utilities use abstractions only
def update_board_array(board, snake_positions, apple_position, board_info):
    """Works with any snake positions from any algorithm"""
```

## 🔮 **Future Extensibility**



### **Extension Points**
The utility architecture supports future enhancements:
- **Algorithm-Specific Utilities**: Extensions can add specialized utilities
- **Performance Optimizations**: Universal utilities can be optimized without breaking extensions
- **New Game Modes**: Additional utilities can be added while maintaining compatibility
- **Cross-Platform Support**: Path and system utilities enable deployment flexibility

### **Integration Guidelines**
When adding new utilities:
1. **Universal**: Add to `ROOT/utils/` if useful across all tasks
2. **Extension-Specific**: Add to `extensions/common/` if useful across multiple extensions
3. **Task-Specific**: Keep in extension directory if specific to one algorithm type
4. **Follow Patterns**: Use established naming and organizational conventions

---

**The utility architecture demonstrates the power of careful separation of concerns. By distinguishing between universal game mechanics and task-specific functionality, the system enables rapid extension development while maintaining consistency and compatibility across all AI approaches. This foundation supports the educational mission while providing the flexibility needed for advanced research and development.**


