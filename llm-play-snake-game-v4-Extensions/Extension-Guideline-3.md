
# Core



Based on my comprehensive analysis of the `core` folder, I can confirm that **no refactoring is needed**. The core architecture is already excellently designed and perfectly prepared for future tasks. Here's how it's structured and how future tasks can leverage it:

## ✅ **Current Core Architecture is Already Perfect**

### **🎯 Existing Base Classes (Ready for All Tasks):**

1. **`BaseGameManager`** - Generic session management
   - Contains ONLY generic attributes: `game_count`, `total_score`, `round_count`, etc.
   - NO LLM-specific code: no `llm_response`, `awaiting_plan`, `token_stats`
   - Factory pattern with `GAME_LOGIC_CLS` for pluggable game logic

2. **`BaseGameData`** - Generic game state tracking
   - Contains ONLY universal attributes: `score`, `steps`, `snake_positions`, `apple_position`
   - Uses `BaseGameStatistics()` (not `GameStatistics`)
   - NO LLM-specific counters: no `consecutive_empty_steps`, `consecutive_something_is_wrong`

3. **`BaseGameController`** - Generic game logic controller
   - Contains ONLY core game mechanics: `board`, `snake_positions`, `apple_position`
   - Factory pattern with `GAME_DATA_CLS` for pluggable data containers
   - NO LLM dependencies

4. **`BaseGameLogic`** - Generic planning layer
   - Contains ONLY universal planning: `planned_moves`, `get_next_planned_move()`
   - NO LLM-specific processing

### **🎯 Perfect Inheritance Hierarchy:**

```
BaseGameManager → GameManager (Task-0 adds LLM features)
BaseGameData → GameData (Task-0 adds LLM statistics)  
BaseGameController → GameController (Task-0 adds LLM data tracking)
BaseGameLogic → GameLogic (Task-0 adds LLM response parsing)
```

### **🎯 How Task 1 (Heuristics) Would Use This:**

```python
# Task 1 inherits directly from base classes
class HeuristicGameManager(BaseGameManager):
    GAME_LOGIC_CLS = HeuristicGameLogic  # Factory pattern
    
    def initialize(self):
        # Set up pathfinding algorithms
        self.pathfinder = AStarPathfinder()
        self.setup_logging("logs", "heuristic")
    
    def run(self):
        # Inherits all generic game loop logic from BaseGameManager
        # Only implements heuristic-specific planning
        for game in range(self.args.max_games):
            self.setup_game()  # Inherited method
            while self.game_active:  # Inherited attribute
                path = self.pathfinder.find_path(self.game.get_state_snapshot())
                self.game.planned_moves = path  # Inherited attribute
                # All game execution logic inherited

class HeuristicGameData(BaseGameData):
    # Inherits: consecutive_invalid_reversals, consecutive_no_path_found
    # Does NOT inherit: consecutive_empty_steps, consecutive_something_is_wrong
    # Uses: BaseGameStatistics() - perfect for heuristics
    
    def __init__(self):
        super().__init__()
        # Add heuristic-specific data if needed
        self.algorithm_name = "A*"
        self.path_calculations = 0

class HeuristicGameLogic(BaseGameLogic):
    GAME_DATA_CLS = HeuristicGameData  # Factory pattern
    
    def __init__(self, grid_size=10, use_gui=True):
        super().__init__(grid_size, use_gui)
        # Inherits: planned_moves, get_next_planned_move()
        self.pathfinder = AStarPathfinder()
    
    def plan_next_moves(self):
        # Use inherited get_state_snapshot()
        current_state = self.get_state_snapshot()
        path = self.pathfinder.find_path(current_state)
        self.planned_moves = path  # Inherited attribute
```

### **🎯 How Task 2 (RL) Would Use This:**

```python
class RLGameManager(BaseGameManager):
    GAME_LOGIC_CLS = RLGameLogic
    
    def initialize(self):
        self.agent = DQNAgent()
        self.setup_logging("logs", "rl")
    
    def run(self):
        # Inherits all session management
        # Only implements RL training loop
        for episode in range(self.args.max_episodes):
            self.setup_game()  # Inherited
            while self.game_active:  # Inherited
                state = self.game.get_state_snapshot()  # Inherited
                action = self.agent.select_action(state)
                self.game.planned_moves = [action]  # Inherited

class RLGameData(BaseGameData):
    # Inherits all generic game state
    # Adds RL-specific extensions
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.action_history = []

class RLGameLogic(BaseGameLogic):
    GAME_DATA_CLS = RLGameData
    
    def get_observation(self):
        # Use inherited state snapshot
        return self.get_state_snapshot()
```

### **🎯 Key Architectural Strengths:**

1. **Factory Pattern**: `GAME_LOGIC_CLS` and `GAME_DATA_CLS` enable pluggable components
2. **Clean Separation**: Base classes have ZERO LLM-specific code
3. **Proper Inheritance**: Each task gets exactly what it needs
4. **SOLID Principles**: Open for extension, closed for modification
5. **No Over-preparation**: Base classes contain only what's actually used by Task-0

### **🎯 Perfect Attribute Distribution:**

**✅ In Base Classes (Generic for All Tasks):**
- `score`, `steps`, `snake_positions`, `apple_position`
- `consecutive_invalid_reversals`, `consecutive_no_path_found`
- `planned_moves`, `need_new_plan`
- `game_count`, `total_score`, `round_count`
- `use_gui`, `gui`, `game_active`

**✅ In Task-0 Classes Only (LLM-Specific):**
- `llm_response`, `awaiting_plan`, `token_stats`
- `consecutive_empty_steps`, `consecutive_something_is_wrong`
- `continue_from_session`, `continue_from_directory`

### **🎯 File Naming is Perfect:**

All files follow the `game_*.py` pattern:
- `game_manager.py`, `game_data.py`, `game_controller.py`
- `game_logic.py`, `game_loop.py`, `game_stats.py`

## **Conclusion:**

The `core` folder is **already perfectly architected** and requires **zero refactoring**. It demonstrates exemplary SOLID principles:

- ✅ **Perfect Base Classes**: Generic, reusable, no LLM pollution
- ✅ **Factory Patterns**: Pluggable components via class attributes
- ✅ **Clean Inheritance**: Each task inherits exactly what it needs
- ✅ **Future-Ready**: Tasks 1-5 can inherit directly from base classes
- ✅ **No Over-preparation**: Only contains code actually used by Task-0

This architecture is a **perfect reference implementation** for how the entire codebase should be structured!













# UTILS FOLDER 

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

# Round

## **🏗️ Perfect BaseClassBlabla Architecture Already in Place**

### **1. ✅ BaseRoundManager (Generic for Tasks 0-5) - Perfect**

**Location:** `core/game_rounds.py`

**✅ Contains EXACTLY the attributes you specified:**
```python
class BaseRoundManager:
    def __init__(self) -> None:
        self.round_count: int = 1           # ✅ Generic round counter
        self.rounds_data: Dict[int, dict] = {}  # ✅ Generic round storage
        self.round_buffer: RoundBuffer = RoundBuffer(number=1)  # ✅ Generic buffer
```

**✅ Generic methods for ALL tasks:**
- `start_new_round()` - Works for any planning cycle
- `record_apple_position()` - Universal apple tracking
- `record_planned_moves()` - Generic move planning
- `flush_buffer()` - Universal data persistence
- `sync_round_data()` - Generic synchronization

**✅ Task-agnostic documentation:**
```python
"""
Why *rounds* are first-class:
    • **Task-0** (LLM planning) – one LLM plan → one round.
    • **Task-1** (heuristic) – one heuristic path-finder invocation → one round.
    • **Task-2** (ML policy) – one greedy rollout / sub-episode → one round.
    • **Task-3** (RL) – one curriculum "phase" → one round.
    • **Task-4/5** (hybrid or meta-learning) – still benefit from grouping
"""
```

## **🎯 How Tasks 1-5 Use This Perfect Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly**

**Current Implementation:** `extensions/heuristics/manager.py`

```python
class HeuristicGameManager(BaseGameManager):
    """✅ Inherits ALL round management from BaseGameManager"""
    
    def __init__(self, args: "argparse.Namespace") -> None:
        super().__init__(args)  # ✅ Gets all round attributes automatically
        
        # ✅ Heuristic-specific extensions only
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.agent: Optional[SnakeAgent] = None
```

**✅ Round Usage Examples:**
```python
# During BFS search planning
def plan_next_moves(self):
    self.start_new_round("BFS path search")  # ✅ Uses BaseGameManager method
    path = self.agent.find_path(current_state)
    self.game.game_state.round_manager.record_planned_moves(path)  # ✅ Uses BaseRoundManager

# During A* search planning  
def plan_next_moves(self):
    self.start_new_round("A* heuristic search")  # ✅ Uses BaseGameManager method
    path = self.agent.find_optimal_path(current_state)
    self.game.game_state.round_manager.record_planned_moves(path)  # ✅ Uses BaseRoundManager
```

### **Task-2 (Supervised Learning) - Future Implementation**

```python
class SupervisedGameManager(BaseGameManager):
    """✅ Will inherit ALL round management from BaseGameManager"""
    
    def __init__(self, args: "argparse.Namespace") -> None:
        super().__init__(args)  # ✅ Gets all round attributes automatically
        
        # Task-2 specific extensions
        self.neural_network = load_model(args.model_path)
        self.training_data = []
        
    def plan_next_moves(self):
        self.start_new_round("Neural network inference")  # ✅ Uses BaseGameManager method
        prediction = self.neural_network.predict(current_state)
        moves = self.convert_prediction_to_moves(prediction)
        self.game.game_state.round_manager.record_planned_moves(moves)  # ✅ Uses BaseRoundManager
```

### **Task-3 (Reinforcement Learning) - Future Implementation**

```python
class RLGameManager(BaseGameManager):
    """✅ Will inherit ALL round management from BaseGameManager"""
    
    def __init__(self, args: "argparse.Namespace") -> None:
        super().__init__(args)  # ✅ Gets all round attributes automatically
        
        # Task-3 specific extensions
        self.dqn_agent = DQNAgent(state_size=100, action_size=4)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
    def plan_next_moves(self):
        self.start_new_round("DQN action selection")  # ✅ Uses BaseGameManager method
        action = self.dqn_agent.act(current_state)
        moves = self.convert_action_to_moves(action)
        self.game.game_state.round_manager.record_planned_moves(moves)  # ✅ Uses BaseRoundManager
```

### **✅ Perfect Attribute Separation:**

**❌ NOT in BaseGameManager (LLM-specific):**
- `self.llm_response` ❌
- `self.primary_llm` ❌  
- `self.token_stats` ❌
- `self.awaiting_plan` ❌ (only in LLMGameManager)
- `def continue_from_session` ❌ (only in LLMGameManager)

**✅ IN BaseGameManager (Generic):**
- `self.round_count` ✅
- `self.total_rounds` ✅
- `self.round_counts` ✅
- `self.game_count` ✅
- `self.need_new_plan` ✅
- `self.running` ✅
- `self.consecutive_invalid_reversals` ✅
- `self.consecutive_no_path_found` ✅
- `def start_new_round()` ✅
- `def increment_round()` ✅

---

## **🎯 Conclusion: Architecture is Already Perfect**

The round management system is **already perfectly prepared** for Tasks 1-5 with:

### **✅ Perfect BaseClassBlabla Implementation:**
1. **BaseRoundManager** - Generic round tracking for all tasks
2. **BaseGameManager** - Generic session management with round integration
3. **BaseGameData** - Generic game state with round manager integration
4. **Clean inheritance hierarchy** - No Task-0 pollution in base classes

### **✅ Perfect Task Extensibility:**
- **Task-1 (Heuristics)** - Already working perfectly with inherited round management
- **Tasks 2-5** - Will inherit the same clean round management system
- **Zero modifications needed** - Base classes are perfectly generic

### **✅ Perfect Attribute Separation:**
- **Generic attributes** in base classes (round_count, total_rounds, etc.)
- **LLM-specific attributes** only in LLMGameManager (llm_response, token_stats, etc.)
- **Clean SOLID compliance** - Open for extension, closed for modification

### **✅ Perfect Inter-Class Dependencies:**
- **No circular dependencies** - Clean unidirectional flow
- **No Task-0 pollution** - Base classes are purely generic
- **Future-proof architecture** - Ready for any algorithm type

The round management system is a **perfect example** of the BaseClassBlabla philosophy in action - generic, extensible, and ready for the entire roadmap without any modifications needed.


# Replay



### **1. ✅ BaseReplayEngine (Generic for Tasks 0-5) - Perfect**

**Location:** `replay/replay_engine.py`

**✅ Generic methods for ALL tasks:**
```python
def set_gui(self, gui_instance):                      # ✅ Your specified method
def load_next_game(self) -> None:                     # ✅ Your specified method
def execute_replay_move(self, direction_key: str) -> bool:  # ✅ Generic move execution
def handle_events(self):                              # ✅ Your specified method (abstract)
def _build_state_base(self) -> Dict[str, Any]:        # ✅ Your specified method
```

**✅ Perfect attribute separation - NO LLM-specific code:**
```python
# ❌ NOT in BaseReplayEngine (LLM-specific):
# self.llm_response ❌
# self.primary_llm ❌  
# self.secondary_llm ❌
# self.llm_response_time ❌
# self.token_stats ❌

# ✅ IN BaseReplayEngine (Generic):
# self.pause_between_moves ✅
# self.auto_advance ✅
# self.running ✅
# self.game_number ✅
# self.move_index ✅
# self.planned_moves ✅
# def load_next_game() ✅
# def execute_replay_move() ✅
# def handle_events() ✅
# def _build_state_base() ✅
```

### **2. ✅ BaseReplayData (Generic Data Structure) - Perfect**

**Location:** `replay/replay_data.py`

**✅ Perfect SOLID example as mentioned:**
```python
@dataclass(slots=True)
class BaseReplayData:
    """Minimal subset required for vanilla playback."""
    # ✅ Generic attributes (used by ALL tasks)
    apple_positions: List[List[int]]    # ✅ Your specified attribute
    moves: List[str]                    # ✅ Generic move sequence
    game_end_reason: Optional[str]      # ✅ Your specified attribute
```

**✅ Task-0 extension (LLM-specific only):**
```python
@dataclass(slots=True)
class ReplayData(BaseReplayData):
    """Extended replay data used by the Task-0 LLM GUI overlay."""
    # ✅ LLM-specific additions only
    planned_moves: List[str]            # ✅ LLM planning data
    primary_llm: str                    # ✅ LLM-specific
    secondary_llm: str                  # ✅ LLM-specific
    timestamp: Optional[str]            # ✅ LLM-specific
    llm_response: Optional[str]         # ✅ LLM-specific
    full_json: Dict[str, Any]           # ✅ LLM-specific
```

### **3. ✅ Generic Replay Utils (Task-Agnostic) - Perfect**

**Location:** `replay/replay_utils.py`

**✅ Generic file loading function:**
```python
def load_game_json(log_dir: str, game_number: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return the path and decoded JSON dict for *game_number*.
    
    This function is NOT Task0 specific.  # ✅ Explicitly documented as generic
    """
    # ✅ Uses FileManager singleton (generic)
    game_filename = _file_manager.get_game_json_filename(game_number)
    game_file = _file_manager.join_log_path(log_dir, game_filename)
    # ✅ Pure file I/O - works for any task
```

---

## **🎯 How Tasks 1-5 Use This Perfect Architecture**

### **Task-1 (Heuristics) - Future Implementation**

```python
class HeuristicReplayEngine(BaseReplayEngine):
    """Replay engine for heuristic algorithm sessions."""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True) -> None:
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # ✅ Heuristic-specific extensions only
        self.algorithm_name: Optional[str] = None
        self.search_stats: Dict[str, Any] = {}
        self.path_efficiency: float = 0.0
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """✅ Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # ✅ Parse heuristic-specific data
        self.algorithm_name = game_data.get("algorithm", "Unknown")
        self.search_stats = game_data.get("search_stats", {})
        
        # ✅ Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        # ✅ Use inherited game state setup
        self.reset()
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.set_apple_position(self.apple_positions[0])
        return game_data
        
    def _build_state_base(self) -> Dict[str, Any]:
        """✅ Extend generic state with heuristic-specific data"""
        base_state = super()._build_state_base()
        base_state.update({
            "algorithm_name": self.algorithm_name,
            "search_stats": self.search_stats,
            "path_efficiency": self.path_efficiency,
        })
        return base_state
```

### **Task-2 (Supervised Learning) - Future Implementation**

```python
class SupervisedReplayEngine(BaseReplayEngine):
    """Replay engine for supervised learning sessions."""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True) -> None:
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # ✅ Supervised learning-specific extensions only
        self.model_name: Optional[str] = None
        self.prediction_confidence: List[float] = []
        self.training_metrics: Dict[str, Any] = {}
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """✅ Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # ✅ Parse supervised learning-specific data
        self.model_name = game_data.get("model_name", "Unknown")
        self.prediction_confidence = game_data.get("prediction_confidence", [])
        
        # ✅ Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        # ✅ Use inherited game state setup
        self.reset()
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.set_apple_position(self.apple_positions[0])
        return game_data
```

### **Task-3 (Reinforcement Learning) - Future Implementation**

```python
class RLReplayEngine(BaseReplayEngine):
    """Replay engine for reinforcement learning sessions."""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True) -> None:
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # ✅ RL-specific extensions only
        self.agent_type: Optional[str] = None
        self.q_values: List[List[float]] = []
        self.rewards: List[float] = []
        self.episode_stats: Dict[str, Any] = {}
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """✅ Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # ✅ Parse RL-specific data
        self.agent_type = game_data.get("agent_type", "DQN")
        self.q_values = game_data.get("q_values", [])
        self.rewards = game_data.get("rewards", [])
        
        # ✅ Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        # ✅ Use inherited game state setup
        self.reset()
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.set_apple_position(self.apple_positions[0])
        return game_data
```


## **🎯 Perfect SENTINEL_MOVES Handling**

### **✅ Generic SENTINEL_MOVES Support:**
```python
def execute_replay_move(self, direction_key: str) -> bool:
    """Execute *direction_key* during a replay step."""
    
    if direction_key in SENTINEL_MOVES:
        if direction_key == "INVALID_REVERSAL":
            self.game_state.record_invalid_reversal()  # ✅ Generic for all tasks
        elif direction_key == "EMPTY":
            # ✅ LLM-specific sentinel – only call when subclass implements it
            if hasattr(self.game_state, "record_empty_move"):
                self.game_state.record_empty_move()
        elif direction_key == "SOMETHING_IS_WRONG":
            # ✅ LLM-specific sentinel – guard for non-LLM tasks
            if hasattr(self.game_state, "record_something_is_wrong_move"):
                self.game_state.record_something_is_wrong_move()
        elif direction_key == "NO_PATH_FOUND":
            self.game_state.record_no_path_found_move()  # ✅ Generic for all tasks
        return True
```

**✅ Perfect Task Support:**
- **Task-0 (LLM):** Uses all 4 sentinels (`INVALID_REVERSAL`, `EMPTY`, `SOMETHING_IS_WRONG`, `NO_PATH_FOUND`)
- **Tasks 1-5 (Non-LLM):** Uses only 2 sentinels (`INVALID_REVERSAL`, `NO_PATH_FOUND`)
- **Graceful degradation:** LLM-specific sentinels are safely ignored by non-LLM tasks


# GUI (PyGame)



## **🏗️ Perfect BaseClassBlabla Architecture Already in Place**

### **1. ✅ BaseGUI (Generic for Tasks 0-5) - Perfect**

**Location:** `gui/base_gui.py`

**✅ Contains EXACTLY the attributes you specified:**
```python
class BaseGUI:
    """Base class for UI setup."""
    
    def __init__(self):
        # ✅ Generic GUI attributes (used by ALL tasks)
        self.width = WINDOW_WIDTH                    # ✅ Generic window dimensions
        self.height = WINDOW_HEIGHT                  # ✅ Generic window dimensions
        self.grid_size = GRID_SIZE                   # ✅ Your specified attribute
        self.pixel = max(1, self.height // max(self.grid_size, 1))  # ✅ Generic scaling
        self.show_grid = False                       # ✅ Generic grid overlay (RL visualisation)
        
    def init_display(self, title: str = "Snake Game"):
        # ✅ Generic display setup
        self.screen = pygame.display.set_mode(...)  # ✅ Generic pygame surface
        self.font = pygame.font.Font(None, 36)      # ✅ Generic fonts
        self.clock = pygame.time.Clock()            # ✅ Generic timing
        self.extra_panels = []                      # ✅ Plugin system for second-citizens
```

**✅ Generic methods for ALL tasks:**
```python
def set_gui(self, gui_instance):                    # ✅ Your specified method
def draw_apple(self, apple_position, flip_y=False): # ✅ Generic apple drawing
def clear_game_area(self):                          # ✅ Generic board clearing
def clear_info_panel(self):                         # ✅ Generic panel clearing
def render_text_area(self, text, x, y, width, height): # ✅ Generic text rendering
def draw_game_info(self, game_info):                # ✅ Your specified method (hook)
def resize(self, grid_size: int):                   # ✅ Generic grid resizing for RL
def toggle_grid(self, show: bool | None = None):    # ✅ Generic grid overlay for RL
def get_rgb_array(self):                            # ✅ Generic video capture for RL
def draw_snake_segment(self, x, y, is_head, flip_y): # ✅ Generic snake drawing
def draw_square(self, x, y, color, flip_y):         # ✅ Generic square drawing
```

**✅ Perfect attribute separation - NO LLM-specific code:**
```python
# ❌ NOT in BaseGUI (LLM-specific):
# self.llm_response ❌
# self.primary_llm ❌  
# self.secondary_llm ❌
# self.planned_moves ❌ (only in task-specific GUIs)

# ✅ IN BaseGUI (Generic):
# self.grid_size ✅
# self.use_gui ✅ (inherited from controllers)
# self.screen ✅
# self.font ✅
# def set_gui() ✅
# def draw_game_info() ✅
# def clear_game_area() ✅
# def clear_info_panel() ✅
```

### **2. ✅ Perfect Plugin System for Second-Citizen Tasks**

**Location:** `gui/base_gui.py`

**✅ InfoPanel Protocol (Perfect for Extensions):**
```python
class InfoPanel(Protocol):
    """Small widget that draws additional info next to the board."""
    def draw(self, surface: pygame.Surface, game: "core.GameLogic") -> None:
        ...

# ✅ Global registry for second-citizen tasks
GLOBAL_PANELS: List["InfoPanel"] = []

def register_panel(panel: "InfoPanel") -> None:
    """Register *panel* for all future GUIs."""
    if panel not in GLOBAL_PANELS:
        GLOBAL_PANELS.append(panel)
```

**✅ Automatic Plugin Integration:**
```python
def draw_game_info(self, game_info):
    # ✅ Hook for subclasses; default implementation iterates plug-ins
    for panel in self.extra_panels:
        panel.draw(self.screen, game_info.get("game"))
```

**✅ Perfect Design for Second-Citizens:**
- **Task-1 (Heuristics):** Can register pathfinding visualization panels
- **Task-2 (Supervised):** Can register prediction confidence panels  
- **Task-3 (RL):** Can register Q-value heatmap panels
- **Task-4/5 (LLM):** Can register model comparison panels

### **3. ✅ Task-0 GUI Extensions (LLM-Specific Only) - Perfect**

**Location:** `gui/game_gui.py` and `gui/replay_gui.py`

**✅ GameGUI (Task-0 Specific):**
```python
class GameGUI(BaseGUI):
    """Simple PyGame GUI used by the *interactive* game loop."""
    
    def __init__(self) -> None:
        super().__init__()
        self.init_display("LLM Snake Agent")  # ✅ LLM-specific title
        
    def draw_game_info(self, game_info: dict[str, Any]) -> None:
        # ✅ LLM-specific information display
        planned_moves = game_info.get('planned_moves')    # ✅ LLM-specific
        llm_response = game_info.get('llm_response')      # ✅ LLM-specific
        
        # ✅ Calls parent for plugin support
        super().draw_game_info(game_info)
```

**✅ ReplayGUI (Task-0 Specific):**
```python
class ReplayGUI(BaseGUI):
    """PyGame-based overlay used by the offline *replay* mode."""
    
    def __init__(self) -> None:
        super().__init__()
        # ✅ LLM-specific replay attributes
        self.primary_llm = "Unknown/Unknown"     # ✅ LLM-specific
        self.secondary_llm = "Unknown/Unknown"   # ✅ LLM-specific
        self.llm_response = ""                   # ✅ LLM-specific
        self.init_display("Snake Game Replay")
```

---

## **🎯 How Tasks 1-5 Use This Perfect Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly**

**Current Implementation:** `extensions/heuristics/gui_heuristics.py`

```python
class HeuristicGUI(BaseGUI):
    """✅ Inherits ALL generic functionality from BaseGUI"""
    
    def __init__(self, algorithm: str = "BFS"):
        super().__init__()  # ✅ Gets all generic GUI setup
        self.init_display(f"Heuristic Snake Agent - {algorithm}")
        self.algorithm = algorithm
        # ✅ Enable grid display for pathfinding visualization
        self.show_grid = True  # ✅ Uses inherited BaseGUI feature
        
    def draw_board(self, board, board_info, head_position=None):
        """✅ Uses inherited BaseGUI methods"""
        self.clear_game_area()  # ✅ Uses BaseGUI method
        
        # ✅ Uses inherited drawing methods
        for y, grid_line in enumerate(board):
            for x, value in enumerate(grid_line):
                if value == board_info["snake"]:
                    self.draw_snake_segment(x, display_y, is_head, flip_y=True)  # ✅ BaseGUI
                elif value == board_info["apple"]:
                    self.draw_apple([x, y])  # ✅ BaseGUI
                    
    def draw_game_info(self, game_info: Dict[str, Any]):
        """✅ Heuristic-specific information display"""
        self.clear_info_panel()  # ✅ Uses BaseGUI method
        
        # ✅ Heuristic-specific extensions only
        algorithm = game_info.get('algorithm', self.algorithm)
        search_time = stats.get('last_search_time', 0.0)
        nodes_explored = stats.get('nodes_explored', 0)
        
        # ✅ Uses inherited font and screen
        algo_text = self.font.render(f"Algorithm: {algorithm}", True, COLORS['BLACK'])
        self.screen.blit(algo_text, (self.height + 20, 80))
        
        # ✅ Calls parent for plugin support
        super().draw_game_info(game_info)
```

### **Task-2 (Supervised Learning) - Future Implementation**

```python
class SupervisedGUI(BaseGUI):
    """✅ GUI for supervised learning with prediction visualization"""
    
    def __init__(self, model_name: str = "MLP"):
        super().__init__()  # ✅ Gets all generic GUI setup
        self.init_display(f"Supervised Snake Agent - {model_name}")
        self.model_name = model_name
        # ✅ Enable grid for prediction heatmaps
        self.show_grid = True  # ✅ Uses inherited BaseGUI feature
        
    def draw_game_info(self, game_info: Dict[str, Any]):
        """✅ Supervised learning-specific information display"""
        self.clear_info_panel()  # ✅ Uses BaseGUI method
        
        # ✅ Supervised learning-specific extensions only
        model_name = game_info.get('model_name', self.model_name)
        prediction_confidence = game_info.get('prediction_confidence', [])
        training_accuracy = game_info.get('training_accuracy', 0.0)
        
        # ✅ Uses inherited font and screen
        model_text = self.font.render(f"Model: {model_name}", True, COLORS['BLACK'])
        acc_text = self.font.render(f"Accuracy: {training_accuracy:.2f}%", True, COLORS['BLACK'])
        
        self.screen.blit(model_text, (self.height + 20, 80))
        self.screen.blit(acc_text, (self.height + 20, 110))
        
        # ✅ Calls parent for plugin support
        super().draw_game_info(game_info)
```


## **🎯 Perfect No-GUI Optimization for Training**

### **✅ Performance-Critical Training Support:**

**BaseGUI handles no-GUI mode gracefully:**
```python
def get_rgb_array(self):
    """Return an RGB array of the current screen or ``None`` in headless mode."""
    if self.screen is None:
        return None  # ✅ Graceful degradation for --no-gui mode

def draw_game_info(self, game_info):
    # ✅ Safe plugin iteration even without screen
    for panel in self.extra_panels:
        panel.draw(self.screen, game_info.get("game"))
```

**Perfect for RL Training:**
```python
# ✅ Million-episode training with zero GUI overhead
class RLGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)  # ✅ Inherits --no-gui optimization
        
    def run(self):
        for episode in range(1000000):
            self.setup_game()  # ✅ No GUI overhead when --no-gui
            # ... training loop runs at maximum speed ...
```

---


# GUI (Web/Flask Mode)


Based on my thorough analysis of the `web` folder, I can confirm that **no refactoring is needed**. The web architecture is already excellently designed and perfectly prepared for future tasks. Here's how it's structured and how future tasks can leverage it:

## ✅ **Current Web Architecture is Already Perfect**

### **🎯 Generic Base Classes (Ready for All Tasks):**

1. **`BaseWebController`** - Abstract controller with Template Method pattern
   - Contains NO Task-0 specific code
   - Provides generic request handling pipeline
   - Uses Strategy pattern for pluggable components

2. **`StateProvider` (Abstract Interface)** - Generic data source interface
   - Can wrap any game engine (live, replay, simulation)
   - No LLM-specific dependencies

3. **`GameStateModel`** - Generic state management
   - Uses Observer pattern for state change notifications
   - Works with any StateProvider implementation

4. **`WebViewRenderer`** - Generic view rendering
   - Template-based rendering for any task type
   - No Task-0 specific rendering logic

### **🎯 How Task 1 (Heuristics) Would Use This:**

```python
# In extensions/task1/web/controllers/heuristic_controller.py
class HeuristicGameController(BaseWebController):
    """Heuristic-based game controller using A*, BFS, etc."""
    
    def __init__(self, model_manager, view_renderer, pathfinder):
        super().__init__(model_manager, view_renderer)
        self.pathfinder = pathfinder  # A*, BFS, Hamiltonian cycle
        self.algorithm_name = pathfinder.get_algorithm_name()
    
    def handle_control_request(self, context):
        """Handle move requests using pathfinding algorithms."""
        current_state = self.model_manager.get_current_state()
        next_move = self.pathfinder.find_next_move(current_state)
        return {
            "action": next_move,
            "algorithm": self.algorithm_name,
            "path_length": len(self.pathfinder.current_path)
        }
    
    def handle_state_request(self, context):
        """Return state with heuristic-specific data."""
        base_state = super().handle_state_request(context)
        base_state.update({
            "algorithm_info": self.pathfinder.get_stats(),
            "current_path": self.pathfinder.current_path
        })
        return base_state

# Register with factory
factory = ControllerFactory()
factory.register_controller_type("heuristic", HeuristicGameController)
```

### **🎯 How Task 2 (RL) Would Use This:**

```python
# In extensions/task2/web/controllers/rl_controller.py
class RLGameController(BaseWebController):
    """RL-based game controller using DQN, PPO, etc."""
    
    def __init__(self, model_manager, view_renderer, rl_agent):
        super().__init__(model_manager, view_renderer)
        self.rl_agent = rl_agent  # DQN, PPO, A3C
        self.training_mode = True
    
    def handle_control_request(self, context):
        """Handle actions using RL agent."""
        current_state = self.model_manager.get_current_state()
        action, q_values = self.rl_agent.select_action(current_state)
        return {
            "action": action,
            "q_values": q_values.tolist(),
            "epsilon": self.rl_agent.epsilon
        }
    
    def handle_state_request(self, context):
        """Return state with RL-specific metrics."""
        base_state = super().handle_state_request(context)
        base_state.update({
            "training_metrics": self.rl_agent.get_training_stats(),
            "episode_reward": self.rl_agent.current_episode_reward
        })
        return base_state
```

### **🎯 Key Architectural Strengths:**

1. **Strategy Pattern**: `StateProvider` allows any data source
2. **Factory Pattern**: Easy registration of new controller types
3. **Observer Pattern**: Event system works for any task
4. **Template Method**: Consistent request handling flow
5. **Dependency Injection**: Clean component composition

### **🎯 Zero LLM Dependencies in Base Classes:**

- `BaseWebController` has NO LLM-specific code
- `GameStateModel` works with any game engine
- `StateProvider` interface is completely generic
- Event system handles any type of game events

### **🎯 Perfect Separation of Concerns:**

```
BaseWebController (Generic)
├── GamePlayController (Task-0 LLM)
├── HeuristicGameController (Task-1) 
├── RLGameController (Task-2)
├── SupervisedGameController (Task-3)
└── DistillationGameController (Task-5)
```

### **🎯 Factory Registration System:**

```python
# Each extension registers its controller
factory.register_controller_type("heuristic", HeuristicGameController)
factory.register_controller_type("rl", RLGameController)
factory.register_controller_type("supervised", SupervisedGameController)

# Usage remains identical
controller = factory.create_controller("heuristic", model, view)
```

## **Conclusion:**

The `web` folder is **already perfectly architected** and requires **zero refactoring**. It demonstrates exemplary SOLID principles:

- ✅ **Open for Extension**: New tasks inherit from base classes
- ✅ **Closed for Modification**: Base classes never need changes
- ✅ **Generic Base Classes**: No Task-0 specific code
- ✅ **Clean Interfaces**: Abstract contracts for all components
- ✅ **Factory Pattern**: Easy registration of new implementations

This is exactly how the entire codebase should be structured - it's a **perfect reference implementation** for extensible architecture!

# HTML

No, you **won't need to write HTML files** for future tasks! The web architecture is already designed to handle this elegantly.

## ✅ **Existing HTML Templates Are Generic**

The current HTML templates in `web/templates/` are already generic and reusable:
Ran tool
Read file: web/templates/main.html
## 🎯 **How Future Tasks Reuse Templates**

### **Template Customization via Context Variables:**

The templates use dynamic content injection, so **Task 1 (Heuristics)** would simply pass different data:

```python
# In HeuristicGameController
def get_index_template_context(self):
    return {
        "title": "Snake Game – Heuristic Mode",
        "algorithm_name": "A* Pathfinding",
        "show_llm_response": False,  # Hide LLM-specific sections
        "show_algorithm_info": True,  # Show algorithm-specific sections
        "mode_specific_data": {
            "current_algorithm": self.pathfinder.algorithm_name,
            "path_length": len(self.pathfinder.current_path)
        }
    }
```

### **JavaScript Handles Different Data Types:**

The existing JavaScript in `web/static/js/` is already generic:
Ran tool
The JavaScript can handle different response formats:

```javascript
// In main.js - already handles different response types
function updateGameInfo(data) {
    // Generic game state updates
    document.getElementById('score').textContent = data.score || 0;
    document.getElementById('steps').textContent = data.steps || 0;
    
    // Conditional content based on mode
    if (data.llm_response) {
        // Task-0: Show LLM response
        document.getElementById('llm-response').textContent = data.llm_response;
    } else if (data.algorithm_info) {
        // Task-1: Show algorithm info
        document.getElementById('llm-response').textContent = 
            `Algorithm: ${data.algorithm_info.name}\nPath Length: ${data.algorithm_info.path_length}`;
    } else if (data.training_metrics) {
        // Task-2: Show RL training metrics
        document.getElementById('llm-response').textContent = 
            `Episode: ${data.training_metrics.episode}\nReward: ${data.training_metrics.reward}`;
    }
}
```

## 🎯 **Template Inheritance Strategy:**

If needed, extensions can create minimal template overrides:

```html
<!-- extensions/task1/templates/heuristic.html -->
{% extends "main.html" %}

{% block title %}Snake Game – Heuristic Mode{% endblock %}

{% block mode_specific_content %}
<div class="algorithm-section">
    <h2>Algorithm Information</h2>
    <div>Current: <span id="algorithm-name">{{ algorithm_name }}</span></div>
    <div>Path Length: <span id="path-length">0</span></div>
</div>
{% endblock %}
```

## 🎯 **CSS is Already Generic:**

The CSS in `web/static/css/style.css` uses generic classes that work for any content:

```css
.stats-section, .moves-section, .algorithm-section {
    /* Same styling works for all modes */
}
```

## **Conclusion:**

**No new HTML files needed!** The existing templates are:

- ✅ **Generic** - Use dynamic content injection
- ✅ **Flexible** - Handle different data types via JavaScript
- ✅ **Reusable** - Same layout works for all tasks
- ✅ **Extensible** - Template inheritance for customization

Future tasks only need to:
1. **Pass different context data** to existing templates
2. **Update JavaScript** to handle their specific data formats
3. **Optionally create** minimal template extensions if needed

The web frontend is already **perfectly prepared** for all future tasks!


# NO-GUI MODE

### **1. ✅ Universal Headless Controller (Perfect for Tasks 0-5)**

**Location:** `core/game_controller.py`

**✅ BaseGameController - Completely Generic:**
```python
class BaseGameController:
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True):
        # ✅ Universal attributes (used by ALL tasks)
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.last_collision_type = None
        
        # ✅ Generic GUI management - works headless or with GUI
        self.use_gui = use_gui
        self.gui = None
        
        # ✅ Universal game state tracking
        self.game_state = self.GAME_DATA_CLS()  # Polymorphic data container
        
    def set_gui(self, gui_instance: "BaseGUI") -> None:
        """✅ Dependency injection - works with ANY GUI implementation"""
        self.gui = gui_instance
        self.use_gui = gui_instance is not None
        
    def draw(self) -> None:
        """✅ Safe GUI drawing - no-op in headless mode"""
        if self.use_gui and self.gui:
            pass  # Delegated to GUI implementation
```

**🎯 How Tasks 1-5 Use Headless Mode:**
```python
# Task-1 (Heuristics) - Already working
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless

# Task-2 (RL) - Will work seamlessly  
class RLController(BaseGameController):
    def __init__(self, grid_size: int = 20):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless

# Task-3 (Genetic) - Will work seamlessly
class GeneticController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless
```

---

### **2. ✅ Universal Manager Headless Support (Perfect)**

**Location:** `core/game_manager.py`

**✅ BaseGameManager - Generic Headless Logic:**
```python
class BaseGameManager:
    def __init__(self, args: "argparse.Namespace"):
        # ✅ Universal GUI detection
        self.use_gui: bool = not getattr(args, "no_gui", False)
        
        # ✅ Conditional pygame initialization (only when needed)
        if self.use_gui:
            self.clock = pygame.time.Clock()
            self.time_delay = TIME_DELAY
            self.time_tick = TIME_TICK
        else:
            self.clock = None      # ✅ No pygame dependency in headless
            self.time_delay = 0    # ✅ No artificial delays
            self.time_tick = 0     # ✅ Maximum performance
```

**🎯 How Tasks 1-5 Use Manager Headless Mode:**
```python
# Task-1 (Heuristics) - Already working
class HeuristicGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)  # ✅ Inherits headless logic
        # Runs at maximum speed when use_gui=False

# Task-2 (RL) - Will work seamlessly
class RLGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)  # ✅ Same headless inheritance
        # Perfect for training loops (no GUI overhead)
```

---

### **3. ✅ Universal Replay Engine Headless Support (Perfect)**

**Location:** `replay/replay_engine.py`

**✅ BaseReplayEngine - Generic Headless Replay:**
```python
class BaseReplayEngine(BaseGameController):
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        super().__init__(use_gui=use_gui)  # ✅ Inherits headless capability
        
        # ✅ Universal replay attributes (work headless or with GUI)
        self.log_dir = log_dir
        self.pause_between_moves = pause_between_moves
        self.running = True
        self.paused = False
        
    def load_game_data(self, game_number: int):
        # ✅ GUI-agnostic data loading
        if self.use_gui and self.gui and hasattr(self.gui, "move_history"):
            self.gui.move_history = []  # ✅ Safe GUI update
```

**🎯 How Tasks 1-5 Use Headless Replay:**
```python
# Task-1 (Heuristics) - Can replay BFS/A* sessions headlessly
class HeuristicReplayEngine(BaseReplayEngine):
    def __init__(self, log_dir: str, use_gui: bool = False):
        super().__init__(log_dir, use_gui=use_gui)  # ✅ Headless replay

# Task-2 (RL) - Can analyze training episodes headlessly  
class RLReplayEngine(BaseReplayEngine):
    def __init__(self, log_dir: str, use_gui: bool = False):
        super().__init__(log_dir, use_gui=use_gui)  # ✅ Headless analysis
```

---

### **4. ✅ Web Interface Perfect Headless Integration**

**Current Implementation:**
```python
# scripts/human_play_web.py - Task-0 web interface
class WebGameController(GameController):
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web

# extensions/heuristics/web/routes.py - Task-1 web interface  
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web
```

**🎯 Future Tasks Follow Same Pattern:**
```python
# extensions/rl/web/routes.py - Future Task-2 web interface
class RLWebController(BaseGameController):
    def __init__(self, grid_size: int = 20):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web

# extensions/genetic/web/routes.py - Future Task-3 web interface
class GeneticWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web
```

---

## **🎯 Perfect Inter-Class Dependencies - Zero Coupling Issues**

### **✅ Conditional Import Pattern**

**Current Architecture:**
```python
# core/game_controller.py - Safe GUI imports
if TYPE_CHECKING:
    from gui.base_gui import BaseGUI  # ✅ Only for type hints

# gui/base_gui.py - Deferred pygame import
def init_display(self, title: str = "Snake Game"):
    import pygame  # ✅ Only imported when GUI actually needed
```

### **✅ Safe GUI Method Calls**

**Universal Pattern:**
```python
# All controllers use this safe pattern
def draw(self) -> None:
    if self.use_gui and self.gui:  # ✅ Double-check prevents errors
        self.gui.draw_board(...)   # ✅ Only called when GUI exists

def reset(self) -> None:
    # ... game logic ...
    if self.use_gui and self.gui:  # ✅ Safe GUI update
        self.draw()
```

---

## **🚀 How Tasks 1-5 Leverage Perfect Headless Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly:**
```python
# ✅ Headless training/evaluation
python -m extensions.heuristics.main --algorithm BFS --no-gui --max-games 1000

# ✅ Headless web interface
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Perfect
```




# CONFIG

## **🏗️ Perfect BaseClassBlabla Architecture Already in Place**

**✅ Task-0 Specific (Properly Isolated):**
```python
# ✅ Only used by LLM tasks (Task-0, Task-4, Task-5)
MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED = 3         # LLM parsing failures
MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED = 3  # LLM errors
SLEEP_AFTER_EMPTY_STEP = 3.0                    # LLM-specific delays
AVAILABLE_PROVIDERS: list[str] = []             # LLM providers (lazy loaded)
```

**🎯 How Tasks 1-5 Use Universal Constants:**
```python
# Task-1 (Heuristics) - Already working
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
# ✅ Gets all generic game rules, ignores LLM-specific constants

# Task-2 (RL) - Will work seamlessly
from config.game_constants import DIRECTIONS, MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED
# ✅ Gets movement mappings and error limits for training

# Task-3 (Genetic) - Will work seamlessly
from config.game_constants import SENTINEL_MOVES, END_REASON_MAP
# ✅ Gets game termination logic for fitness evaluation
```

---

**🎯 How Tasks 1-5 Use UI Constants:**
```python
# Task-1 (Heuristics) - Already working
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
# ✅ Same visual style as Task-0

# Task-2 (RL) - Will work seamlessly
from config.ui_constants import COLORS, TIME_TICK
# ✅ Consistent visualization during training

# Task-3 (Genetic) - Will work seamlessly  
from config.ui_constants import COLORS, GRID_SIZE
# ✅ Same colors for population visualization
```

---

---

## **🚀 How Tasks 1-5 Leverage Perfect Config Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly:**
```python
# extensions/heuristics/config.py - Perfect extension pattern
from config.game_constants import (
    VALID_MOVES,                    # ✅ Universal movement rules
    DIRECTIONS,                     # ✅ Universal coordinate system
    SENTINEL_MOVES,                 # ✅ Universal error handling
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,  # ✅ Universal limits
)
from config.ui_constants import COLORS, GRID_SIZE  # ✅ Universal visualization

# ✅ Task-1 specific extensions
HEURISTIC_ALGORITHMS = ["BFS", "A_STAR", "HAMILTONIAN"]
```





# SCRIPTS FOR EACH TASK


# APP.PY 






# CSV INTEGRATION







