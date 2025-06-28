# Final Decision 8: Configuration and Validation Standards

> **Guidelines Alignment:**
> - This document is governed by the SUPREME_RULES in `final-decision-10.md`.
> - All configuration must use simple, lightweight patterns following SUPREME_RULE NO.3.
> - Reference: `config.md` for comprehensive configuration architecture.
> - This file is a GOOD_RULES authoritative reference and must be cross-referenced by all related documentation.

> **See also:** `config.md`, `core.md`, `final-decision-10.md`, `single-source-of-truth.md`.

## 🎯 **Core Philosophy: Simple Configuration Management**

Configuration and validation in the Snake Game AI project follows lightweight, extensible patterns that support diverse algorithm requirements while maintaining simplicity and educational value.

### **Guidelines Alignment**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making configuration changes
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing configuration decisions
- **SUPREME_RULE NO.3**: Enables lightweight configuration utilities with simple logging (print statements only)

## 🏗️ **Configuration Architecture**

### **Universal Configuration Constants**
```python
# config/game_constants.py - Universal for all tasks
GRID_SIZE_DEFAULT = 10
MAX_GAMES_DEFAULT = 1
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# config/ui_constants.py - Universal UI settings
COLORS = {
    'SNAKE': (0, 255, 0),
    'APPLE': (255, 0, 0),
    'BACKGROUND': (0, 0, 0)
}
```

### **Extension-Specific Configuration**
```python
# extensions/heuristics-v0.03/heuristic_config.py
HEURISTIC_ALGORITHMS = ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
PATHFINDING_TIMEOUT = 5.0
VISUALIZATION_SPEED = 1.0

# extensions/supervised-v0.03/supervised_config.py  
SUPERVISED_MODELS = ["MLP", "CNN", "XGBOOST", "LIGHTGBM"]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
```

## 📊 **Validation Standards**

### **Simple Validation Functions**
```python
# extensions/common/validation/dataset_validator.py
def validate_grid_size(grid_size: int) -> bool:
    """Simple grid size validation"""
    if grid_size < 5 or grid_size > 50:
        print(f"[Validator] Invalid grid size: {grid_size} (must be 5-50)")  # SUPREME_RULE NO.3
        return False
    print(f"[Validator] Grid size {grid_size} is valid")  # SUPREME_RULE NO.3
    return True

def validate_csv_schema(csv_data: dict) -> bool:
    """Simple CSV schema validation"""
    required_fields = ['head_x', 'head_y', 'apple_x', 'apple_y', 'target_move']
    
    for field in required_fields:
        if field not in csv_data:
            print(f"[Validator] Missing required field: {field}")  # SUPREME_RULE NO.3
            return False
    
    print(f"[Validator] CSV schema validation passed")  # SUPREME_RULE NO.3
    return True

def validate_agent_config(agent_type: str, config: dict) -> bool:
    """Simple agent configuration validation"""
    if agent_type.upper() in ["MLP", "CNN", "LSTM"]:
        required = ['learning_rate', 'batch_size', 'epochs']
    elif agent_type.upper() in ["DQN", "PPO", "A3C"]:
        required = ['epsilon_start', 'epsilon_decay', 'num_episodes']
    else:
        required = ['grid_size']  # Minimal requirements for heuristics
    
    for field in required:
        if field not in config:
            print(f"[Validator] Missing {agent_type} config field: {field}")  # SUPREME_RULE NO.3
            return False
    
    print(f"[Validator] {agent_type} configuration validation passed")  # SUPREME_RULE NO.3
    return True
```

## 🚀 **Configuration Loading Patterns**

### **Runtime Configuration**
```python
class RuntimeConfig:
    """Simple runtime configuration management"""
    
    def __init__(self, extension_type: str, **kwargs):
        self.extension_type = extension_type
        self.config = {}
        self._load_global_config()
        self._load_extension_config()
        self._load_runtime_config(kwargs)
        print(f"[RuntimeConfig] Initialized for {extension_type}")  # SUPREME_RULE NO.3
    
    def _load_global_config(self):
        """Load universal configuration constants"""
        from config.game_constants import GRID_SIZE_DEFAULT, MAX_GAMES_DEFAULT
        self.config.update({
            'grid_size': GRID_SIZE_DEFAULT,
            'max_games': MAX_GAMES_DEFAULT
        })
    
    def _load_extension_config(self):
        """Load extension-specific configuration"""
        if self.extension_type == 'heuristics':
            from extensions.heuristics_v0_03.heuristic_config import (
                HEURISTIC_ALGORITHMS, PATHFINDING_TIMEOUT, VISUALIZATION_SPEED
            )
            self.config.update({
                'algorithms': HEURISTIC_ALGORITHMS,
                'pathfinding_timeout': PATHFINDING_TIMEOUT,
                'visualization_speed': VISUALIZATION_SPEED
            })
            print(f"[RuntimeConfig] Loaded heuristics configuration")  # SUPREME_RULE NO.3
        elif self.extension_type == 'supervised':
            from extensions.supervised_v0_03.supervised_config import (
                SUPERVISED_MODELS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
            )
            self.config.update({
                'models': SUPERVISED_MODELS,
                'learning_rate': DEFAULT_LEARNING_RATE,
                'batch_size': DEFAULT_BATCH_SIZE,
                'epochs': DEFAULT_EPOCHS
            })
            print(f"[RuntimeConfig] Loaded supervised configuration")  # SUPREME_RULE NO.3
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def validate(self) -> bool:
        """Validate configuration"""
        return validate_grid_size(self.config.get('grid_size', 10))
```

## 📋 **Configuration Best Practices**

### **Simple Configuration Standards**
- ✅ **Universal constants** in `ROOT/config/` for all tasks
- ✅ **Extension-specific constants** in extension directories
- ✅ **Simple validation functions** with clear error messages
- ✅ **Runtime configuration** for dynamic parameter management
- ✅ **Simple logging** using print statements (SUPREME_RULE NO.3)

### **Validation Requirements**
- ✅ **Grid size validation** (5-50 range)
- ✅ **Algorithm type validation** (against available algorithms)
- ✅ **Parameter range validation** (learning rates, batch sizes, etc.)
- ✅ **Schema validation** (CSV, JSON, dataset formats)
- ✅ **Path validation** (dataset and model paths)

---

**This configuration and validation system ensures consistent, reliable parameter management across all Snake Game AI extensions while maintaining simplicity and educational value.**
