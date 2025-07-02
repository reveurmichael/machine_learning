# Configuration Architecture for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for configuration architecture and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `core.md`, `standalone.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Flexible Configuration Management**

Configuration in the Snake Game AI project follows a hierarchical, extensible architecture that supports both simple parameter management and complex multi-extension configurations. The system is designed to be lightweight, educational, and maintainable while supporting the diverse needs of different algorithm types, strictly following SUPREME_RULES from `final-decision-10.md`.

### **Educational Value**
- **Configuration Patterns**: Demonstrates best practices for parameter management
- **Hierarchical Design**: Shows how to structure complex configurations
- **Extensibility**: Framework for adding new configuration types
- **Validation**: Proper parameter validation and error handling

## 🏗️ **Factory Pattern: Canonical Method is create()**

All configuration factories must use the canonical method name `create()` for instantiation, not `create_config()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision-10.md`.

### Reference Implementation

A generic, educational `SimpleFactory` is provided in `utils/factory_utils.py`:

```python
from utils.factory_utils import SimpleFactory

class MyConfig:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myconfig", MyConfig)
config = factory.create("myconfig", name="TestConfig")  # CANONICAL create() method per SUPREME_RULES
print(config.name)  # Output: TestConfig
```

### Example Configuration Factory

```python
class ConfigFactory:
    _registry = {
        "HEURISTIC": HeuristicConfig,
        "SUPERVISED": SupervisedConfig,
        "REINFORCEMENT": ReinforcementConfig,
    }
    @classmethod
    def create(cls, config_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        config_class = cls._registry.get(config_type.upper())
        if not config_class:
            raise ValueError(f"Unknown config type: {config_type}")
        print(f"[ConfigFactory] Creating config: {config_type}")  # SUPREME_RULES compliant logging
        return config_class(**kwargs)
```

## 🏗️ **Configuration Hierarchy**

### **1. Global Configuration (ROOT/config/)**

TODO: list 
TODO: list 
TODO: list 
TODO: list 
TODO: list 
TODO: list 


### **2. Extension-Specific Configuration**
```python
# extensions/heuristics-v0.03/config.py
HEURISTIC_ALGORITHMS = ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
PATHFINDING_TIMEOUT = 5.0
VISUALIZATION_SPEED = 1.0

# extensions/supervised-v0.03/config.py
SUPERVISED_MODELS = ["MLP", "CNN", "XGBOOST", "LIGHTGBM"]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# extensions/reinforcement-v0.03/config.py
RL_ALGORITHMS = ["DQN", "PPO", "A3C"]
DEFAULT_EPSILON_START = 0.9
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_REWARD_APPLE = 10
DEFAULT_REWARD_DEATH = -10
```

### **3. Runtime Configuration**
```python
class RuntimeConfig:
    """
    Runtime configuration that combines global and extension-specific settings.
    
    Design Pattern: Builder Pattern
    - Constructs configuration from multiple sources
    - Validates configuration parameters
    - Provides default values for missing parameters
    """
    
    def __init__(self, extension_type: str, **kwargs):
        self.extension_type = extension_type
        self.config = {}
        self._load_global_config()
        self._load_extension_config()
        self._load_runtime_config(kwargs)
        self._validate_config()
        print(f"[RuntimeConfig] Initialized for {extension_type}")  # SUPREME_RULES compliant logging
    
    def _load_global_config(self):
        """Load global configuration constants"""
        from config.game_constants import GRID_SIZE_DEFAULT, MAX_GAMES_DEFAULT
        self.config.update({
            'grid_size': GRID_SIZE_DEFAULT,
            'max_games': MAX_GAMES_DEFAULT,
            'visualization': True
        })
    
    def _load_extension_config(self):
        """Load extension-specific configuration"""
        if self.extension_type == 'heuristics':
            from extensions.heuristics_v0_03.config import HEURISTIC_ALGORITHMS
            self.config.update({
                'algorithms': HEURISTIC_ALGORITHMS,
                'pathfinding_timeout': 5.0,
                'visualization_speed': 1.0
            })
        elif self.extension_type == 'supervised':
            from extensions.supervised_v0_03.config import SUPERVISED_MODELS
            self.config.update({
                'models': SUPERVISED_MODELS,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            })
        elif self.extension_type == 'reinforcement':
            from extensions.reinforcement_v0_03.config import RL_ALGORITHMS
            self.config.update({
                'algorithms': RL_ALGORITHMS,
                'epsilon_start': 0.9,
                'epsilon_decay': 0.995,
                'reward_apple': 10,
                'reward_death': -10
            })
    
    def _load_runtime_config(self, runtime_params: dict):
        """Load runtime configuration parameters"""
        self.config.update(runtime_params)
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate grid size
        if self.config.get('grid_size', 0) < 5 or self.config.get('grid_size', 0) > 50:
            raise ValueError(f"Invalid grid size: {self.config.get('grid_size')}")
        
        # Validate max games
        if self.config.get('max_games', 0) < 1:
            raise ValueError(f"Invalid max games: {self.config.get('max_games')}")
        
        print(f"[RuntimeConfig] Configuration validated successfully")  # SUPREME_RULES compliant logging
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
        print(f"[RuntimeConfig] Set {key} = {value}")  # SUPREME_RULES compliant logging
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return self.config.copy()
```

## 🚀 **Configuration Implementation Examples**

### **Heuristic Configuration**
```python
class HeuristicConfig:
    """
    Configuration for heuristic pathfinding algorithms.
    
    Design Pattern: Strategy Pattern
    - Configures different pathfinding algorithms
    - Provides algorithm-specific parameters
    - Supports runtime parameter adjustment
    """
    
    def __init__(self, algorithm: str = "BFS", **kwargs):
        self.algorithm = algorithm.upper()
        self.grid_size = kwargs.get('grid_size', 10)
        self.max_games = kwargs.get('max_games', 1)
        self.visualization = kwargs.get('visualization', True)
        self.pathfinding_timeout = kwargs.get('pathfinding_timeout', 5.0)
        self.visualization_speed = kwargs.get('visualization_speed', 1.0)
        
        # Algorithm-specific parameters
        if self.algorithm == "ASTAR":
            self.heuristic_weight = kwargs.get('heuristic_weight', 1.0)
        elif self.algorithm == "HAMILTONIAN":
            self.cycle_type = kwargs.get('cycle_type', 'standard')
        
        print(f"[HeuristicConfig] Initialized {algorithm} config")  # SUPREME_RULES compliant logging
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid_algorithms = ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        
        if self.grid_size < 5 or self.grid_size > 50:
            raise ValueError(f"Invalid grid size: {self.grid_size}")
        
        if self.max_games < 1:
            raise ValueError(f"Invalid max games: {self.max_games}")
        
        print(f"[HeuristicConfig] Configuration validated")  # SUPREME_RULES compliant logging
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'max_games': self.max_games,
            'visualization': self.visualization,
            'pathfinding_timeout': self.pathfinding_timeout,
            'visualization_speed': self.visualization_speed,
            'heuristic_weight': getattr(self, 'heuristic_weight', None),
            'cycle_type': getattr(self, 'cycle_type', None)
        }
```

## 📊 **Configuration Management Standards**

### **Configuration Loading**
```python
class ConfigLoader:
    """
    Standardized configuration loading across extensions.
    
    Design Pattern: Factory Pattern
    - Loads configuration from multiple sources
    - Supports different file formats
    - Provides validation and error handling
    """
    
    def __init__(self, extension_type: str):
        self.extension_type = extension_type
        print(f"[ConfigLoader] Initialized for {extension_type}")  # SUPREME_RULES compliant logging
    
    def load_from_file(self, file_path: str) -> dict:
        """Load configuration from file"""
        if file_path.endswith('.json'):
            return self._load_json(file_path)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return self._load_yaml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def load_from_dict(self, config_dict: dict) -> dict:
        """Load configuration from dictionary"""
        return config_dict.copy()
    
    def load_from_env(self) -> dict:
        """Load configuration from environment variables"""
        config = {}
        prefix = f"SNAKE_{self.extension_type.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config[config_key] = self._parse_env_value(value)
        
        print(f"[ConfigLoader] Loaded {len(config)} env vars")  # SUPREME_RULES compliant logging
        return config
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Global Configuration**: Constants in ROOT/config/
- [ ] **Extension Configuration**: Extension-specific settings
- [ ] **Runtime Configuration**: Dynamic parameter management
- [ ] **Configuration Factory**: Canonical `create()` method per SUPREME_RULES
- [ ] **Configuration Loading**: Support for multiple sources
- [ ] **Configuration Validation**: Comprehensive parameter validation
- [ ] **Error Handling**: Graceful handling of configuration errors

### **Quality Standards**
- [ ] **Hierarchical Design**: Clear configuration hierarchy
- [ ] **Extensibility**: Easy to add new configuration types
- [ ] **Validation**: Comprehensive parameter validation
- [ ] **Documentation**: Clear parameter descriptions
- [ ] **Type Safety**: Proper type hints and validation

### **Integration Requirements**
- [ ] **Factory Pattern**: Compatible with configuration factory patterns
- [ ] **Standalone Principle**: Configuration follows standalone principles
- [ ] **Logging**: Uses SUPREME_RULES compliant logging (print() statements)
- [ ] **Error Recovery**: Robust error handling and recovery
- [ ] **Serialization**: Support for configuration serialization

---

**Configuration architecture ensures flexible, maintainable, and extensible parameter management across all Snake Game AI extensions. By following these standards, developers can create robust configuration systems that support diverse algorithm requirements while maintaining educational value and technical excellence.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization