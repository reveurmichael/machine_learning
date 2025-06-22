
## **🎯 Perfect Architecture: Task-Agnostic vs Task-Specific Separation**

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
- **Task-1**: Heuristic outputs can be validated via `validate_json_format()`
- **Task-2**: Neural network predictions parsed via these utilities
- **Task-3**: RL action sequences validated through same JSON schema
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
- **Task-1**: Web interface shows BFS search progress using same JSON format
- **Task-2**: Training visualization uses same color scheme and state structure
- **Task-3**: RL training dashboard inherits same web infrastructure
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

class BFSAgent:
    def get_move(self, game):
        # ✅ Use universal collision detection
        head_pos = game.head_position
        snake_positions = game.snake_positions
        grid_size = game.grid_size
        
        # ✅ Check each possible move for safety
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = DIRECTIONS[direction]
            new_pos = [head_pos[0] + dx, head_pos[1] + dy]
            
            # ✅ Universal collision checking
            wall_hit, body_hit = check_collision(new_pos, snake_positions, grid_size, False)
            
            if not (wall_hit or body_hit):
                # ✅ Universal move processing
                if is_reverse(direction, game.current_direction):
                    continue  # Skip reversal moves
                return normalize_direction(direction)  # ✅ Same format as Task-0!
        
        return "NO_PATH_FOUND"  # ✅ Same sentinel as Task-0!

class HeuristicWebController:
    def get_current_state(self):
        # ✅ Universal web state building
        return build_state_dict(
            self.snake_positions,     # ✅ From BaseGameController
            self.apple_position,      # ✅ From BaseGameController  
            self.score,              # ✅ From BaseGameController
            self.steps,              # ✅ From BaseGameController
            self.grid_size,          # ✅ From BaseGameController
            extra={
                "algorithm": "BFS",
                "search_time": 0.003,
                "nodes_explored": 42,
                "task_type": "heuristics"
            }
        )
        
    def generate_apple(self):
        # ✅ Universal apple placement
        return generate_random_apple(self.snake_positions, self.grid_size)
        
    def save_results(self, results):
        # ✅ Universal JSON saving
        return safe_json_save(results, f"logs/heuristics/game_{self.game_number}.json")
```

### **Task-3 (RL) Integration:**
```python
from utils.board_utils import update_board_array, is_position_valid
from utils.collision_utils import check_collision
from utils.moves_utils import normalize_direction
from utils.json_utils import validate_json_format  
from utils.web_utils import build_state_dict

class RLEnvironment:
    def step(self, action):
        # ✅ Universal move normalization
        action = normalize_direction(action)
        
        # ✅ Universal collision detection
        head_pos = self.get_head_position()
        dx, dy = DIRECTIONS[action]
        new_pos = [head_pos[0] + dx, head_pos[1] + dy]
        
        wall_hit, body_hit = check_collision(new_pos, self.snake_positions, self.grid_size, False)
        
        # ✅ Calculate reward based on universal collision system
        if wall_hit or body_hit:
            reward = -10  # Collision penalty
            done = True
        elif self.check_apple_collision(new_pos):
            reward = 10   # Apple reward
            done = False
        else:
            reward = -0.1 # Step penalty
            done = False
            
        # ✅ Universal board update
        update_board_array(self.board, self.snake_positions, self.apple_position, self.board_info)
        
        return self.get_observation(), reward, done, {}
        
    def get_web_state(self):
        # ✅ Universal web interface
        return build_state_dict(
            self.snake_positions, self.apple_position, self.score,
            self.steps, self.grid_size,
            extra={
                "episode": self.episode,
                "total_reward": self.total_reward,
                "epsilon": self.epsilon,
                "task_type": "reinforcement_learning"
            }
        )
```

### **Task-2 (Supervised Learning) Integration:**
```python
from utils.board_utils import get_empty_positions, update_board_array
from utils.moves_utils import normalize_direction, normalize_directions
from utils.json_utils import safe_json_load, safe_json_save

class DatasetGenerator:
    def generate_training_data(self, num_games=10000):
        """Generate training dataset using universal utilities."""
        dataset = []
        
        for game_idx in range(num_games):
            # ✅ Use universal board utilities
            snake_positions = self.initialize_snake()
            apple_position = generate_random_apple(snake_positions, self.grid_size)
            
            while not self.game_over:
                # ✅ Capture state using universal board update
                board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
                update_board_array(board, snake_positions, apple_position, self.board_info)
                
                # ✅ Get optimal action from heuristic (ground truth)
                optimal_action = self.get_optimal_action_bfs()
                optimal_action = normalize_direction(optimal_action)  # ✅ Universal format
                
                # ✅ Record training example
                dataset.append({
                    'board_state': board.tolist(),
                    'snake_positions': [pos.tolist() for pos in snake_positions],
                    'apple_position': apple_position.tolist(),
                    'optimal_action': optimal_action,
                    'score': self.score
                })
                
                # ✅ Execute move using universal collision detection
                self.execute_move(optimal_action)
                
        # ✅ Save using universal JSON utilities
        safe_json_save(dataset, "datasets/supervised_training_data.json")
        return dataset
        
    def validate_model_output(self, model_predictions):
        """Validate neural network predictions."""
        # ✅ Universal move validation
        for prediction in model_predictions:
            moves = prediction.get('moves', [])
            normalized_moves = normalize_directions(moves)  # ✅ Universal
            
            # ✅ Universal JSON format validation
            is_valid, error = validate_json_format({'moves': normalized_moves})
            if not is_valid:
                print(f"Invalid model output: {error}")
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
# ✅ Generic utilities use universal constants
from config.game_constants import END_REASON_MAP       # Game termination (all tasks)
from config.ui_constants import COLORS                 # Visualization (all tasks)

# ✅ Task-0 utilities use LLM constants
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

class AnyTaskAgent:
    def get_move(self, game):
        # Universal collision detection
        wall_hit, body_hit = check_collision(next_pos, game.snake_positions, game.grid_size, False)
        
        # Universal move processing
        if not (wall_hit or body_hit):
            return normalize_direction(best_move)  # Same format as all tasks
        else:
            return "NO_PATH_FOUND"  # Same sentinel as all tasks

class AnyTaskWebController:
    def get_current_state(self):
        # Universal web state building
        return build_state_dict(
            self.snake_positions, self.apple_position, self.score,
            self.steps, self.grid_size,
            extra={"task_specific_data": "..."}
        )

# Works immediately with zero modifications to utils!
```

### **✅ Generic Examples in Action:**

**Heuristics can immediately use:**
```python
# BFS pathfinding with universal collision detection
wall_hit, body_hit = check_collision(candidate_pos, snake_body, grid_size, False)
if not (wall_hit or body_hit):
    path.append(normalize_direction("UP"))  # Same format as LLM
```

**RL can immediately use:**
```python
# Environment step with universal utilities  
reward = 10 if check_apple_collision(head_pos, apple_pos) else -0.1
state_dict = build_state_dict(snake_pos, apple_pos, score, steps, grid_size)
```

**Supervised Learning can immediately use:**
```python
# Dataset generation with universal board management
update_board_array(board, snake_positions, apple_position, board_info)
training_example = {'state': board.tolist(), 'action': normalize_direction(action)}
```


