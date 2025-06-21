
# Core

The `core/` folder has been expertly refactored following SOLID principles with a clear **Base Class vs Concrete Implementation** pattern. Here's how each file enables future tasks:

## 1. **`core/game_agents.py` - Universal Agent Contract**

**Generic Architecture:**
- Defines `SnakeAgent` protocol with single method: `get_move(game: Any) -> str | None`
- **Completely task-agnostic** - works for ANY algorithm type
- Runtime-checkable for type safety

**How Tasks 1-5 Use It:**
```python
# Task-1 (Heuristics)
class BFSAgent(SnakeAgent):
    def get_move(self, game: Any) -> str | None:
        path = self.bfs_algorithm(game.board, game.head_position, game.apple_position)
        return path[0] if path else "NO_PATH_FOUND"

# Task-2 (Supervised Learning) 
class MLAgent(SnakeAgent):
    def get_move(self, game: Any) -> str | None:
        features = self.extract_features(game)
        prediction = self.model.predict(features)
        return self.action_map[prediction]

# Task-3 (Reinforcement Learning)
class DQNAgent(SnakeAgent):
    def get_move(self, game: Any) -> str | None:
        state = self.preprocess_state(game)
        q_values = self.network(state)
        return self.epsilon_greedy_action(q_values)
```

---

## 2. **`core/game_manager.py` - Session Management Hierarchy**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
BaseGameManager                    # For Tasks 1-5
├── Core session metrics (game_count, total_score, game_scores)
├── Game state management (game_active, need_new_plan, running)  
├── Visualization (use_gui, pause_between_moves, clock)
├── Factory hook (GAME_LOGIC_CLS = BaseGameLogic)
└── Abstract methods (initialize(), run())

LLMGameManager(BaseGameManager)    # Task-0 only
├── LLM-specific counters (empty_steps, something_is_wrong_steps)
├── LLM infrastructure (llm_client, time_stats, token_stats)
└── Continuation features (continue_from_session())
```

**How Tasks 1-5 Use BaseGameManager:**
```python
# Task-1 (Heuristics)
class HeuristicGameManager(BaseGameManager):
    GAME_LOGIC_CLS = BaseGameLogic  # Use generic logic
    
    def initialize(self) -> None:
        self.agent = BFSAgent()  # or A*, Hamiltonian, etc.
        self.setup_game()        # Inherited method
        
    def run(self) -> None:
        run_game_loop(self)      # Uses same game loop as Task-0!

# Task-3 (Reinforcement Learning)  
class RLGameManager(BaseGameManager):
    GAME_LOGIC_CLS = BaseGameLogic
    
    def initialize(self) -> None:
        self.agent = DQNAgent(state_dim=..., action_dim=4)
        self.setup_game()
        self.replay_buffer = ReplayBuffer(10000)
        
    def run(self) -> None:
        # Training loop with experience collection
        for episode in range(self.args.max_episodes):
            run_game_loop(self)  # Collect experience
            if episode % 10 == 0:
                self.agent.train(self.replay_buffer)
```

**Inherited Benefits:**
- ✅ Session logging (`setup_logging()`, `save_session_summary()`)
- ✅ Game lifecycle (`setup_game()`, `get_pause_between_moves()`) 
- ✅ Round management (`start_new_round()`)
- ✅ Error tracking (invalid_reversals, no_path_found_steps)
- ✅ GUI integration (automatic pygame setup when `use_gui=True`)

---

## 3. **`core/game_controller.py` - Game Engine Base**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
BaseGameController                 # Pure game engine
├── Board management (board, snake_positions, apple_position)
├── Collision detection (check_collision, _check_collision)  
├── Apple generation (generate_random_apple, _generate_apple)
├── Move validation (filter_invalid_reversals, normalize_direction)
├── Factory hook (GAME_DATA_CLS = BaseGameData)
└── GUI abstraction (set_gui, draw)

GameController(BaseGameController) # Task-0 specific  
└── GAME_DATA_CLS = GameData       # Adds LLM statistics
```

**How Tasks 1-5 Use BaseGameController:**
```python
# Task-1: Heuristic algorithms access core game state
class BFSAgent:
    def get_move(self, game: BaseGameController) -> str | None:
        # Access generic game state
        board = game.board                    # Numpy array [grid_size, grid_size]
        head = game.head_position            # [x, y] coordinates  
        apple = game.apple_position          # [x, y] coordinates
        snake = game.snake_positions         # List of [x, y] positions
        
        # Use generic utilities
        valid_moves = game.filter_invalid_reversals(["UP", "DOWN", "LEFT", "RIGHT"])
        
        # Run BFS pathfinding
        path = self.bfs(board, head, apple, snake)
        return path[0] if path else "NO_PATH_FOUND"

# Task-3: RL agents extract features from game state  
class RLEnvironment:
    def __init__(self):
        self.game = BaseGameController(grid_size=15, use_gui=False)
        
    def get_observation(self):
        # Extract features from BaseGameController
        return {
            'board': self.game.board,                    # Full board state
            'head': self.game.head_position,            # Snake head
            'apple': self.game.apple_position,          # Apple location  
            'score': self.game.score,                   # Current score
            'snake_length': self.game.snake_length      # Snake size
        }
```

**Inherited Capabilities:**
- ✅ **Collision detection**: Wall/self-collision with detailed reasons
- ✅ **Apple generation**: Random placement avoiding snake body
- ✅ **Move validation**: Automatic reversal filtering  
- ✅ **Board updates**: Automatic numpy array synchronization
- ✅ **State snapshots**: JSON-serializable game state for replay
- ✅ **GUI integration**: Optional pygame rendering

---

## 4. **`core/game_data.py` - Statistics Tracking Hierarchy**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
BaseGameData                       # Generic for all tasks
├── Core state (score, steps, game_over, snake_positions)
├── Move tracking (moves, current_game_moves, planned_moves)  
├── Apple history (apple_positions, apple_positions_history)
├── Error counters (consecutive_invalid_reversals, no_path_found_steps)
├── Statistics (stats: BaseGameStatistics)
└── Round tracking (round_manager: RoundManager)

GameData(BaseGameData)             # Task-0 LLM-specific
├── LLM counters (empty_steps, something_is_wrong_steps)
├── LLM timings (llm_communication_start/end, response_times)
├── Token statistics (primary/secondary token usage)
└── LLM response logging (record_parsed_llm_response)
```

**How Tasks 1-5 Use BaseGameData:**
```python
# Task-1: Heuristic data tracking
class HeuristicGameData(BaseGameData):
    def __init__(self):
        super().__init__()
        # Add heuristic-specific metrics
        self.path_lengths = []
        self.search_times = []
        
    def record_search_result(self, path_length: int, search_time: float):
        self.path_lengths.append(path_length)  
        self.search_times.append(search_time)

# Task-3: RL episode tracking
class RLGameData(BaseGameData):
    def __init__(self):
        super().__init__()
        # Add RL-specific metrics
        self.episode_rewards = []
        self.q_values = []
        
    def record_step(self, action, reward, q_val):
        super().record_move(action)  # Use base move tracking
        self.episode_rewards.append(reward)
        self.q_values.append(q_val)
```

**Inherited Features:**
- ✅ **Move recording**: `record_move()`, `record_apple_position()`
- ✅ **Game lifecycle**: `reset()`, `record_game_end()`  
- ✅ **Error tracking**: `record_invalid_reversal()`, `record_no_path_found_move()`
- ✅ **Round management**: Automatic round tracking for any planning algorithm
- ✅ **State snapshots**: `get_basic_game_state()` for replays
- ✅ **JSON serialization**: Compatible with existing replay infrastructure

---

## 5. **`core/game_logic.py` - Planning-Based Game Logic**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
BaseGameLogic(BaseGameController)  # Generic planning support
├── Planned moves (planned_moves: List[str])
├── Move execution (get_next_planned_move)
└── State snapshots (get_state_snapshot)

GameLogic(BaseGameLogic)           # Task-0 LLM-specific  
├── LLM integration (parse_llm_response, get_state_representation)
├── Rich properties (head, apple, body for prompt templates)
└── GUI integration (draw with LLM response display)
```

**How Tasks 1-5 Use BaseGameLogic:**
```python
# Task-1: Multi-move heuristic planning
class HeuristicGameLogic(BaseGameLogic):
    def plan_path(self, agent):
        """Generate multi-step path using heuristic algorithm."""
        # Use inherited planned_moves for multi-step execution
        path = agent.get_full_path(self)  # Returns ["UP", "RIGHT", "DOWN", ...]
        self.planned_moves = path
        
        # Use inherited move execution
        next_move = self.get_next_planned_move()  # Pops first move
        return next_move

# Task-2: Model-based planning  
class MLGameLogic(BaseGameLogic):
    def plan_sequence(self, model):
        """Use ML model to generate move sequences."""
        state = self.get_state_snapshot()  # Inherited method
        predicted_sequence = model.predict_sequence(state, horizon=5)
        self.planned_moves = predicted_sequence
        return self.get_next_planned_move()
```

**Key Benefits:**
- ✅ **Multi-move planning**: Any task can use `planned_moves` for lookahead
- ✅ **Automatic execution**: `get_next_planned_move()` handles sequence execution  
- ✅ **State representation**: `get_state_snapshot()` provides neutral game state
- ✅ **Reset handling**: Automatic `planned_moves` clearing on game reset

---

## 6. **`core/game_loop.py` - Universal Game Loop**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
run_game_loop(manager)             # Universal function for all tasks
├── Frame pacing (pygame timing, delays)
├── Event handling (process_events via utils)
├── Game lifecycle (_handle_game_over, reset logic)  
├── Agent integration (_process_agent_game)
├── Manager abstraction (works with BaseGameManager)
└── Statistics processing (universal game over handling)
```

**How Tasks 1-5 Use run_game_loop:**
```python
# Task-1: Heuristic game session  
class HeuristicGameManager(BaseGameManager):
    def run(self):
        run_game_loop(self)  # Same function as Task-0!

# Task-3: RL training sessions
class RLGameManager(BaseGameManager):
    def run(self):
        for episode in range(self.args.max_episodes):
            run_game_loop(self)  # Collect experience
            if episode % 10 == 0:
                self.agent.train()
```

**Inherited Infrastructure:**
- ✅ **Frame timing**: Perfect pygame timing for GUI mode, max speed for headless
- ✅ **Event handling**: Window close, keyboard input via utilities  
- ✅ **Game transitions**: Automatic game-over detection and reset
- ✅ **Statistics**: Session-level tracking via helper utilities
- ✅ **Error handling**: Exception safety with graceful pygame cleanup

---

## 7. **`core/game_stats.py` - Metrics Hierarchy**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Architecture:**
```python
BaseStepStats                      # Move counters for all tasks
├── valid, invalid_reversals, no_path_found
└── asdict() for JSON serialization

StepStats(BaseStepStats)           # Task-0 LLM counters
├── empty, something_wrong (LLM-specific)
└── Extended asdict() 

BaseGameStatistics                 # Generic session stats  
├── time_stats: TimeStats
├── step_stats: BaseStepStats  
└── Universal helpers (valid_steps, invalid_reversals)

GameStatistics(BaseGameStatistics) # Task-0 LLM stats
├── Response times, token usage
└── LLM-specific methods
```

**How Tasks 1-5 Use Statistics:**
```python
# Task-1: Custom heuristic statistics
class HeuristicStepStats(BaseStepStats):
    def __init__(self):
        super().__init__()
        self.search_failures = 0      # Algorithm-specific
        self.optimal_paths = 0        # Heuristic-specific
        
    def asdict(self):
        base = super().asdict()       # Gets valid, invalid_reversals, no_path_found  
        base.update({
            'search_failures': self.search_failures,
            'optimal_paths': self.optimal_paths
        })
        return base

class HeuristicGameStatistics(BaseGameStatistics):
    step_stats: HeuristicStepStats = field(default_factory=HeuristicStepStats)
```

---

## 8. **`core/game_rounds.py` - Universal Round Tracking**

**Status:** ✅ **Perfect - No modifications needed**

**Generic Round Concept:**
- **Task-0 (LLM)**: One LLM prompt/response = one round
- **Task-1 (Heuristics)**: One path-finding invocation = one round  
- **Task-2 (Supervised)**: One model inference = one round
- **Task-3 (RL)**: One action selection = one round
- **Task-4/5 (LLM variants)**: One model query = one round

**How All Tasks Use Rounds:**
```python
# Any task can track planning cycles:
class AnyGameData(BaseGameData):
    def start_planning_cycle(self, apple_pos):
        # Inherited round tracking works for ANY algorithm
        self.round_manager.start_new_round(apple_pos)
        
    def record_plan(self, moves):
        # Works for heuristic paths, RL sequences, LLM responses  
        self.round_manager.record_planned_moves(moves)
        
    def finish_planning_cycle(self):
        self.round_manager.sync_round_data()
```

---

## 9. **`core/game_runner.py` - Quick-Play Utility**

**Status:** ✅ **Perfect - No modifications needed**

**Universal Agent Testing:**
```python
# Test ANY agent type with same interface:
from core.game_runner import play

# Test heuristic agent
trajectory = play(BFSAgent(), max_steps=500, render=True)

# Test RL agent  
trajectory = play(DQNAgent(), max_steps=1000, render=False, seed=42)

# Test custom agent
trajectory = play(MyCustomAgent(), max_steps=300, render=True)
```

---

## **Complete Task Integration Example**

Here's how a **Task-1 (Heuristics)** would integrate with zero modifications to core files:

```python
# extensions/heuristics/manager.py
from core.game_manager import BaseGameManager
from core.game_loop import run_game_loop  
from core.game_logic import BaseGameLogic
from extensions.heuristics.agents import BFSAgent

class HeuristicGameManager(BaseGameManager):
    GAME_LOGIC_CLS = BaseGameLogic      # Use generic logic
    
    def initialize(self) -> None:
        self.agent = BFSAgent()         # Heuristic agent
        self.setup_game()               # Inherited setup
        
    def run(self) -> None:
        run_game_loop(self)             # Same loop as Task-0!

# Entry point - identical to Task-0  
if __name__ == "__main__":
    args = parse_args()                 # Same argument parsing
    manager = HeuristicGameManager(args)
    manager.initialize()
    manager.run()                       # Same workflow
```

**Everything Works Identically:**
- ✅ GUI/no-GUI modes via `--no-gui` flag
- ✅ Session logging to `logs/heuristics/`
- ✅ Round tracking and replay files  
- ✅ Error handling and statistics
- ✅ Game reset and multi-game sessions
- ✅ Pygame timing and frame pacing


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


## **🏗️ Perfect BaseClassBlabla Architecture Already in Place**

### **1. ✅ Universal Web Infrastructure (Perfect for Tasks 0-5)**

**Location:** `config/web_constants.py`, `utils/web_utils.py`

**✅ Completely Task-Agnostic Components:**
```python
# config/web_constants.py - Universal Flask configuration
FLASK_STATIC_FOLDER: Final[str] = str(REPO_ROOT / "web" / "static")
FLASK_TEMPLATE_FOLDER: Final[str] = str(REPO_ROOT / "web" / "templates")
DEFAULT_HOST: Final[str] = "127.0.0.1"
FLASK_THREADED: Final[bool] = True

# utils/web_utils.py - Generic web utilities
def build_state_dict(snake_positions, apple_position, score, steps, grid_size, *, extra=None):
    """Constructs a generic, JSON-serializable game state dictionary for web UIs."""
    
def build_color_map() -> Dict[str, Tuple[int, int, int]]:
    """Builds the color map required by the web front-end."""
    
def create_health_check_response(components, error_threshold=5):
    """Create standardized health check response for web endpoints."""
```

**🎯 How Tasks 1-5 Use These:**
- **Task-1 (Heuristics):** Uses `build_state_dict()` with `extra={"algorithm": "BFS", "search_stats": {...}}`
- **Task-2 (RL):** Uses `build_state_dict()` with `extra={"neural_network": "DQN", "training_stats": {...}}`
- **Task-3 (Genetic):** Uses `build_state_dict()` with `extra={"generation": 42, "fitness_score": 0.95}`
- **Task-4/5:** Similar pattern with algorithm-specific extras

---

### **2. ✅ Flask Blueprint Architecture (Perfect Extension Pattern)**

**Current Implementation:** `extensions/heuristics/web/routes.py`

**✅ Perfect Blueprint Pattern Already Working:**
```python
# extensions/heuristics/web/routes.py
heuristics_bp = Blueprint('heuristics', __name__, url_prefix='/heuristics')

class HeuristicWebController(BaseGameController):  # ✅ Inherits from base
    def get_current_state(self) -> Dict[str, Any]:
        # ✅ Uses universal build_state_dict()
        base_state = build_state_dict(
            self.snake_positions, self.apple_position, self.score, 
            self.steps, self.grid_size,
            extra={
                "algorithm": self.current_algorithm,      # ✅ Task-1 specific
                "search_stats": self.search_stats,        # ✅ Task-1 specific
                "task_type": "heuristics"                 # ✅ Task-1 specific
            }
        )
        return base_state

# Flask routes using universal patterns
@heuristics_bp.route('/api/state')
def api_state():
    return jsonify(heuristic_controller.get_current_state())  # ✅ Generic pattern
```

**🎯 How Future Tasks Use This Pattern:**
```python
# extensions/reinforcement_learning/web/routes.py (Future Task-2)
rl_bp = Blueprint('rl', __name__, url_prefix='/rl')

class RLWebController(BaseGameController):  # ✅ Same inheritance
    def get_current_state(self):
        return build_state_dict(  # ✅ Same universal function
            self.snake_positions, self.apple_position, self.score,
            self.steps, self.grid_size,
            extra={
                "neural_network": "DQN",           # ✅ Task-2 specific
                "training_episode": 1000,          # ✅ Task-2 specific
                "epsilon": 0.1,                    # ✅ Task-2 specific
                "task_type": "reinforcement_learning"
            }
        )

# extensions/genetic_algorithm/web/routes.py (Future Task-3)
genetic_bp = Blueprint('genetic', __name__, url_prefix='/genetic')
# Same pattern with genetic-specific extras...
```

---

### **3. ✅ Universal Client-Side Architecture (Perfect for All Tasks)**

**Location:** `web/static/js/common.js`

**✅ Completely Generic JavaScript Functions:**
```javascript
// web/static/js/common.js - Works for ANY algorithm type
function drawGrid(ctx, gridSize, pixelSize) { /* Universal grid drawing */ }
function drawRect(ctx, x, y, color, pixelSize) { /* Universal shape drawing */ }
async function sendApiRequest(url, method = 'GET', data = null) { /* Universal API */ }

// Color system works for all tasks
let COLORS = {
    SNAKE_HEAD: '#3498db',    // ✅ Universal colors
    SNAKE_BODY: '#2980b9',    // ✅ Used by all algorithms
    APPLE: '#e74c3c',         // ✅ Same across tasks
    BACKGROUND: '#2c3e50',    // ✅ Consistent UI
    GRID: '#34495e'           // ✅ Generic grid
};
```

**🎯 How Tasks 1-5 Use Client-Side:**
- **Task-1:** `extensions/heuristics/web/static/js/heuristics.js` extends common functions
- **Task-2:** `extensions/rl/web/static/js/rl.js` adds neural network visualization
- **Task-3:** `extensions/genetic/web/static/js/genetic.js` adds population visualization
- **All tasks:** Use same `drawGrid()`, `drawRect()`, `sendApiRequest()` functions

---

### **4. ✅ Template System (Perfect Inheritance Pattern)**

**Current Architecture:**
```html
<!-- web/templates/main.html - Task-0 LLM specific -->
<div class="llm-response-section">
    <h2>LLM Response</h2>
    <pre id="llm-response" class="llm-response"></pre>
</div>

<!-- extensions/heuristics/web/templates/heuristics.html - Task-1 specific -->
<div class="algorithm-section">
    <h2>Algorithm: BFS</h2>
    <div id="search-stats" class="search-stats"></div>
</div>
```

**🎯 How Future Tasks Extend Templates:**
```html
<!-- extensions/rl/web/templates/rl.html - Future Task-2 -->
<div class="neural-network-section">
    <h2>Neural Network: DQN</h2>
    <div id="training-stats" class="training-stats"></div>
</div>

<!-- extensions/genetic/web/templates/genetic.html - Future Task-3 -->
<div class="evolution-section">
    <h2>Generation: 42</h2>
    <div id="population-stats" class="population-stats"></div>
</div>
```

---

## **🎯 Perfect Inter-Class Dependencies - Zero Coupling Issues**

### **✅ Dependency Injection Pattern**

**Current Implementation:**

```python
# scripts/main_web.py - Task-0 specific
from core.game_manager import GameManager

_game_manager = GameManager(args)  # ✅ Task-0 implementation

# extensions/heuristics/web/routes.py - Task-1 specific  
from extensions.heuristics.manager import HeuristicGameManager

heuristic_controller = HeuristicWebController(grid_size)  # ✅ Task-1 implementation
```

**🎯 Future Tasks Follow Same Pattern:**
```python
# extensions/rl/web/routes.py - Future Task-2
from extensions.rl.manager import RLGameManager
rl_controller = RLWebController(grid_size)  # ✅ Task-2 implementation

# extensions/genetic/web/routes.py - Future Task-3
from extensions.genetic.manager import GeneticGameManager  
genetic_controller = GeneticWebController(grid_size)  # ✅ Task-3 implementation
```

---

### **✅ Universal Health Check System**

**Current Implementation:**
```python
# utils/web_utils.py - Works for ALL tasks
def create_health_check_response(components, error_threshold=5):
    """Create standardized health check response for web endpoints."""
    
# scripts/human_play_web.py - Task-0 usage
components = {
    "web_server": app,                    # ✅ Universal
    "game_controller": _game_controller,  # ✅ Task-specific instance
    "heartbeat_thread": _heartbeat_thread # ✅ Universal pattern
}
health_response = create_health_check_response(components)

# extensions/heuristics/web/routes.py - Task-1 usage
components = {
    "web_server": app,                    # ✅ Same universal component
    "heuristic_controller": heuristic_controller,  # ✅ Task-1 specific
    "algorithm_thread": algorithm_thread  # ✅ Task-1 specific
}
health_response = create_health_check_response(components)  # ✅ Same function
```

---

## **🚀 How Tasks 1-5 Leverage This Perfect Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly:**
```python
# ✅ Uses BaseGameController inheritance
class HeuristicWebController(BaseGameController):
    def get_current_state(self):
        return build_state_dict(...)  # ✅ Universal function

# ✅ Uses Flask Blueprint pattern
heuristics_bp = Blueprint('heuristics', __name__, url_prefix='/heuristics')

# ✅ Uses universal health checks
health_response = create_health_check_response(components)
```

### **Task-2 (Reinforcement Learning) - Will Work Seamlessly:**
```python
# extensions/rl/web/routes.py
class RLWebController(BaseGameController):  # ✅ Same inheritance
    def get_current_state(self):
        return build_state_dict(  # ✅ Same universal function
            extra={"neural_network": "DQN", "training_stats": {...}}
        )

rl_bp = Blueprint('rl', __name__, url_prefix='/rl')  # ✅ Same pattern
```

### **Task-3 (Genetic Algorithm) - Will Work Seamlessly:**
```python
# extensions/genetic/web/routes.py  
class GeneticWebController(BaseGameController):  # ✅ Same inheritance
    def get_current_state(self):
        return build_state_dict(  # ✅ Same universal function
            extra={"generation": 42, "population_size": 100, "fitness": {...}}
        )

genetic_bp = Blueprint('genetic', __name__, url_prefix='/genetic')  # ✅ Same pattern
```

---

## **✅ Summary: Web System Requires ZERO Modifications**

The web system demonstrates **perfect BaseClassBlabla architecture** with:

1. **✅ Universal Infrastructure:** `web_utils.py`, `web_constants.py` work for all tasks
2. **✅ Perfect Blueprint Pattern:** Each task gets its own `/task-name/` URL namespace
3. **✅ Generic Client-Side:** `common.js` functions work for any algorithm type
4. **✅ Flexible Template System:** Task-specific templates extend universal patterns
5. **✅ Zero Coupling:** Each task uses dependency injection with task-specific controllers
6. **✅ Standardized Health Checks:** Universal monitoring across all web interfaces
7. **✅ Consistent API Patterns:** All tasks use same REST endpoint structure

**The web architecture is already future-proof and ready for Tasks 1-5 with zero modifications needed.**

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







