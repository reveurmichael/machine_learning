# Final Decision 3: Simple Utility Functions Architecture

> **SUPREME AUTHORITY**: This document establishes the definitive standards for simple utility functions following SUPREME_RULE NO.3.

## 🎯 **Core Philosophy: Lightweight, OOP-Based Common Utilities**

### **Guidelines Alignment**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making utility architectural changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining utility patterns through inheritance rather than tight coupling

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`utils.md`**: Authoritative reference for utility function standards
- **`elegance.md`**: Authoritative reference for code quality and simplicity standards
- **`single-source-of-truth.md`**: Ensures utility consistency across all extensions
- **`no-over-preparation.md`**: Prevents over-engineering of utility functions

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str) -> str:
    """Simple dataset path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}"
    print(f"[Path] Generated dataset path: {path}")  # SUPREME_RULE NO.3
    return path

def validate_grid_size(grid_size: int):
    """Simple grid size validation"""
    if grid_size < 5 or grid_size > 50:
        print(f"[Validator] Invalid grid size: {grid_size} (must be 5-50)")  # SUPREME_RULE NO.3
        raise ValueError(f"Grid size should be reasonable (5-50), got {grid_size}")
    print(f"[Validator] Grid size {grid_size} is valid")  # SUPREME_RULE NO.3

def register_validator(data_type: str, validator_func):
    """Simple validator registration"""
    print(f"[Registry] Registering validator for {data_type}")  # SUPREME_RULE NO.3
    _validators[data_type] = validator_func
```

The `extensions/common/` folder should serve as a lightweight, reusable foundation for all extensions, supporting experimentation and flexibility. Its code must be simple, preferably object-oriented (OOP) but never over-engineered.

## 🎯 **Executive Summary**

This document establishes **lightweight utility functions** for the Snake Game AI project following **SUPREME_RULE NO.3**: "The extensions/common/ folder should stay lightweight and generic." Complex singleton patterns have been simplified to simple, easy-to-understand functions.

## 🛠️ **Existing Infrastructure (Available When Needed)**

The project includes a robust singleton implementation in `ROOT/utils/singleton_utils.py`:
- **`SingletonABCMeta`**: Generic, thread-safe metaclass for all tasks
- **Double-checked locking**: High-performance singleton implementation
- **Testing utilities**: Available for any extension that truly needs singleton behavior

## 🚫 **EXPLICIT DECISION: NO singleton_utils.py in extensions/common/**

**CRITICAL ARCHITECTURAL DECISION**: This project **explicitly rejects**:
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ❌ **Duplicating singleton functionality in extensions/common/**

**Rationale**: 
- **ROOT/utils/singleton_utils.py is already generic** and works for all tasks (0-5)
- **SUPREME_RULE NO.3**: Avoid unnecessary duplication and complexity
- **Most use cases should use simple functions** instead of singletons

## 🔄 **DECISION: Simple Functions Over Complex Singletons**

### **✅ SIMPLIFIED UTILITY FUNCTIONS**

#### **1. Simple Path Management Functions**
```python
from abc import ABC, abstractmethod
from utils.singleton_utils import SingletonABCMeta

# SUPREME_RULE NO.3: Simple path functions instead of complex managers
from datetime import datetime
from pathlib import Path

def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str) -> str:
    """Simple dataset path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}"
    print(f"[Path] Generated dataset path: {path}")
    return path

def get_model_path(extension_type: str, version: str, grid_size: int, model_name: str) -> str:
    """Simple model path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/models/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{model_name}"
    print(f"[Path] Generated model path: {path}")
    return path

def ensure_directory_exists(path: str):
    """Simple directory creation"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"[Path] Ensured directory exists: {path}")

def validate_grid_size(grid_size: int):
    """Simple grid size validation"""
    if grid_size < 5 or grid_size > 50:
        raise ValueError(f"Grid size should be reasonable (5-50), got {grid_size}")
    print(f"[Path] Grid size {grid_size} is valid")
```

#### **2. ConfigurationManager**
```python
# SUPREME_RULE NO.3: Simple configuration access instead of complex singletons
def get_universal_config(module: str, key: str):
    """Simple universal configuration access"""
    print(f"[Config] Accessing universal config: {module}.{key}")
    
    # Direct imports - simple and clear
    if module == "game":
        from config.game_constants import VALID_MOVES, DIRECTIONS
        config_map = {"VALID_MOVES": VALID_MOVES, "DIRECTIONS": DIRECTIONS}
    elif module == "ui":
        from config.ui_constants import COLORS, GRID_SIZE
        config_map = {"COLORS": COLORS, "GRID_SIZE": GRID_SIZE}
    else:
        config_map = {}
    
    return config_map.get(key)

def get_extension_config(module: str, key: str, default=None):
    """Simple extension configuration access"""
    print(f"[Config] Accessing extension config: {module}.{key}")
    
    # Extension-specific constants defined locally (SUPREME_RULE NO.3)
    if module == "dataset":
        local_config = {"CSV_SCHEMA_VERSION": "1.0", "FEATURE_COUNT": 16}
    else:
        local_config = {}
    
    return local_config.get(key, default)
```

#### **3. ValidationRegistry**
```python
# SUPREME_RULE NO.3: Simple validation functions instead of complex registries
_validators = {}  # Simple module-level registry

def register_validator(data_type: str, validator_func):
    """Simple validator registration"""
    print(f"[Validation] Registering validator for {data_type}")
    _validators[data_type] = validator_func

def validate_data(data_type: str, data):
    """Simple data validation"""
    validator = _validators.get(data_type, lambda x: True)  # Default: always valid
    try:
        result = validator(data)
        print(f"[Validation] {data_type} validation: {'PASS' if result else 'FAIL'}")
        return result
    except Exception as e:
        print(f"[Validation] {data_type} validation error: {e}")
        return False

def get_schema(schema_type: str, version: str = "latest"):
    """Simple schema retrieval"""
    print(f"[Schema] Getting {schema_type} schema v{version}")
    
    # Simple schema definitions without complex caching
    if schema_type == "csv":
        return ["head_x", "head_y", "apple_x", "apple_y", "snake_length", "target_move"]
    
    return []
```

#### **4. DatasetSchemaManager**
```python
# SUPREME_RULE NO.3: Simple schema functions instead of complex managers
def get_csv_schema(grid_size: int, version: str = "v1"):
    """Simple CSV schema retrieval - grid-size agnostic"""
    print(f"[Schema] Getting CSV schema v{version} for grid {grid_size}x{grid_size}")
    
    # Standard 16-feature schema works for any grid size
    return [
        'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
        'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
        'danger_straight', 'danger_left', 'danger_right',
        'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
        'game_id', 'step_in_game', 'target_move'
    ]

def extract_features(game_state, grid_size: int):
    """Simple feature extraction function"""
    print(f"[Features] Extracting features for grid {grid_size}x{grid_size}")
    
    # Simple feature extraction without complex classes
    features = {
        'head_x': game_state.get('head_position', [0, 0])[0],
        'head_y': game_state.get('head_position', [0, 0])[1],
        'snake_length': len(game_state.get('snake_positions', []))
    }
    return features

def validate_dataset_compatibility(dataset_path: str, expected_schema: str):
    """Simple dataset validation"""
    print(f"[Validation] Checking dataset {dataset_path} against {expected_schema}")
    # Simple validation without complex class hierarchies
    return True  # Flexible validation following SUPREME_RULE NO.3
```

## ❌ **NON-SINGLETON CLASSES**

### **Classes That Should NOT Be Singletons**

#### **1. Game-Specific Classes**
```python
# ❌ NOT SINGLETONS - Need multiple instances

class GameManager:
    """
    Manages individual game sessions.
    
    Why NOT Singleton:
    - Multiple games can run simultaneously
    - Each game has independent state and lifecycle
    - Comparison experiments need separate game instances
    - Parallel processing requires separate managers
    """
    def __init__(self, game_config):
        # Individual game session initialization
        self.config = game_config
        self.game_state = {}
        self.is_running = False
        print(f"[GameManager] Initialized with config: {game_config}")  # SUPREME_RULE NO.3
    
    def run_game(self):
        # Game execution logic
        print(f"[GameManager] Starting game execution")  # SUPREME_RULE NO.3
        self.is_running = True
        # Game loop implementation here
        print(f"[GameManager] Game execution completed")  # SUPREME_RULE NO.3

class GameData:
    """
    Stores data for individual games.
    
    Why NOT Singleton:
    - Each game has unique data (score, steps, moves)
    - Historical analysis requires multiple game data instances
    - Concurrent games need separate data containers
    - Memory efficiency requires data cleanup after games
    """
    def __init__(self):
        self.score = 0
        self.steps = 0
        self.moves = []
        self.snake_positions = []
        self.apple_positions = []
        print(f"[GameData] Initialized new game data")  # SUPREME_RULE NO.3

class GameController:
    """
    Controls individual game logic and state.
    
    Why NOT Singleton:
    - Multiple games with different rules/settings
    - A/B testing requires separate controllers
    - Extension-specific game logic variations
    - Independent game state management
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.current_state = {}
        self.game_rules = {}
        print(f"[GameController] Initialized for grid size {grid_size}")  # SUPREME_RULE NO.3
```

#### **2. Agent Classes**
```python
# ❌ NOT SINGLETONS - Need multiple instances for comparison

class BFSAgent:
    """
    Breadth-First Search agent implementation.
    
    Why NOT Singleton:
    - Multiple agents for comparison experiments
    - Different parameter configurations for same algorithm
    - Parallel agent execution
    - Independent agent state and performance tracking
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.visited = set()
        self.queue = []
        print(f"[BFSAgent] Initialized BFS agent for grid size {grid_size}")  # SUPREME_RULE NO.3
    
    def plan_move(self, game_state):
        # BFS pathfinding logic
        print(f"[BFSAgent] Planning move using BFS")  # SUPREME_RULE NO.3
        # BFS implementation here
        return "UP"  # Default move

class MLPAgent:
    """
    Multi-Layer Perceptron agent implementation.
    
    Why NOT Singleton:
    - Multiple model instances with different architectures
    - Ensemble methods require multiple agent instances
    - Different training states and checkpoints
    - A/B testing different model configurations
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = self._build_model()
        self.optimizer = None
        print(f"[MLPAgent] Initialized MLP agent")  # SUPREME_RULE NO.3

class DQNAgent:
    """
    Deep Q-Network agent implementation.
    
    Why NOT Singleton:
    - Multiple agents with different network architectures
    - Target vs main network instances
    - Population-based training methods
    - Independent exploration and exploitation strategies
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        print(f"[DQNAgent] Initialized DQN agent")  # SUPREME_RULE NO.3
```

#### **3. Training Classes**
```python
# ❌ NOT SINGLETONS - Need separate instances for different experiments

class TrainingManager:
    """
    Manages training processes for ML models.
    
    Why NOT Singleton:
    - Multiple simultaneous training sessions
    - Different algorithms training in parallel
    - Independent training state and progress tracking
    - Experiment isolation and reproducibility
    """
    def __init__(self, model_type: str, config: dict):
        self.model_type = model_type
        self.config = config
        self.training_history = []
        print(f"[TrainingManager] Initialized for {model_type}")  # SUPREME_RULE NO.3
    
    def start_training(self):
        """Start training process"""
        print(f"[TrainingManager] Starting {self.model_type} training")  # SUPREME_RULE NO.3
        # Training implementation here

class ModelTrainer:
    """
    Handles specific model training workflows.
    
    Why NOT Singleton:
    - Different models training simultaneously
    - Independent training configurations
    - Separate validation and checkpoint management
    - Parallel hyperparameter optimization
    """
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = 0
        print(f"[ModelTrainer] Initialized trainer")  # SUPREME_RULE NO.3
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        print(f"[ModelTrainer] Training epoch {self.epoch}")  # SUPREME_RULE NO.3
        # Epoch training implementation here

class DatasetLoader:
    """
    Loads and preprocesses datasets for training.
    
    Why NOT Singleton:
    - Different datasets for different experiments
    - Independent data preprocessing pipelines
    - Memory management and batch loading
    - Concurrent data loading for parallel training
    """
    def __init__(self, dataset_path: str, batch_size: int):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.data = None
        print(f"[DatasetLoader] Initialized for {dataset_path}")  # SUPREME_RULE NO.3
    
    def load_data(self):
        """Load dataset from path"""
        print(f"[DatasetLoader] Loading dataset")  # SUPREME_RULE NO.3
        # Data loading implementation here
```

## 🏗️ **Singleton Implementation Standards**

### **Singleton Implementation Reference**

**Note**: Robust singleton implementation already exists in `ROOT/utils/singleton_utils.py`:

```python
# Reference: ROOT/utils/singleton_utils.py (already implemented)
from utils.singleton_utils import SingletonABCMeta

class MyManager(metaclass=SingletonABCMeta):
    """Use existing singleton implementation when truly needed"""
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        # Actual initialization code here
        self.config = {}
        self.resources = []
        print(f"[MyManager] Singleton initialized")  # SUPREME_RULE NO.3
```

### **Usage Pattern Example**
```python
# Example: Using singleton in extension
from utils.singleton_utils import SingletonABCMeta

class HeuristicPathManager(metaclass=SingletonABCMeta):
    """Extension-specific path manager inheriting singleton behavior"""
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Extension-specific initialization
        self.heuristic_algorithms = ["BFS", "ASTAR", "HAMILTONIAN"]
        print(f"[HeuristicPathManager] Initialized singleton")  # SUPREME_RULE NO.3
    
    def get_algorithm_dataset_path(self, algorithm: str, grid_size: int) -> Path:
        """Get dataset path for specific heuristic algorithm"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.03_{timestamp}/{algorithm}")
        print(f"[HeuristicPathManager] Generated path: {path}")  # SUPREME_RULE NO.3
        return path

# Usage in heuristics extension
path_manager = HeuristicPathManager()  # Always returns same instance
dataset_path = path_manager.get_algorithm_dataset_path("BFS", 10)
```