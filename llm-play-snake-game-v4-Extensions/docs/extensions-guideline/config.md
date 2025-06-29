# Configuration Architecture for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for configuration architecture and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `core.md`, `standalone.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Flexible Configuration Management**

Configuration in the Snake Game AI project follows a hierarchical, extensible architecture that supports both simple parameter management and complex multi-extension configurations. The system is designed to be lightweight, educational, and maintainable while supporting the diverse needs of different algorithm types.

### **Educational Value**
- **Configuration Patterns**: Demonstrates best practices for parameter management
- **Hierarchical Design**: Shows how to structure complex configurations
- **Extensibility**: Framework for adding new configuration types
- **Validation**: Proper parameter validation and error handling

## 🏗️ **Factory Pattern: Canonical Method is create()**

All configuration factories must use the canonical method name `create()` for instantiation, not `create_config()` or any other variant. This ensures consistency and aligns with the KISS principle. Factories should be simple, dictionary-based, and avoid over-engineering.

### Reference Implementation

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
from extensions.common.utils.factory_utils import SimpleFactory

class MyConfig:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myconfig", MyConfig)
config = factory.create("myconfig", name="TestConfig")  # CANONICAL create() method
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
    def create(cls, config_type: str, **kwargs):  # CANONICAL create() method
        config_class = cls._registry.get(config_type.upper())
        if not config_class:
            raise ValueError(f"Unknown config type: {config_type}")
        print(f"[ConfigFactory] Creating config: {config_type}")  # Simple logging
        return config_class(**kwargs)
```

## 🏗️ **Configuration Hierarchy**

### **1. Global Configuration (ROOT/config/)**
```python
# config/game_constants.py
GRID_SIZE_DEFAULT = 10
MAX_GAMES_DEFAULT = 1
VISUALIZATION_DEFAULT = True

# config/llm_constants.py (Task-0 specific)
LLM_PROVIDERS = ["hunyuan", "deepseek", "mistral"]
MAX_TOKENS_DEFAULT = 1000

# config/network_constants.py
HOST_DEFAULT = "localhost"
PORT_DEFAULT = 8000
```

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
        print(f"[RuntimeConfig] Initialized for {extension_type}")  # Simple logging
    
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
        
        print(f"[RuntimeConfig] Configuration validated successfully")  # Simple logging
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
        print(f"[RuntimeConfig] Set {key} = {value}")  # Simple logging
    
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
        
        print(f"[HeuristicConfig] Initialized {algorithm} config")  # Simple logging
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid_algorithms = ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        
        if self.grid_size < 5 or self.grid_size > 50:
            raise ValueError(f"Invalid grid size: {self.grid_size}")
        
        if self.max_games < 1:
            raise ValueError(f"Invalid max games: {self.max_games}")
        
        print(f"[HeuristicConfig] Configuration validated")  # Simple logging
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

### **Supervised Learning Configuration**
```python
class SupervisedConfig:
    """
    Configuration for supervised learning models.
    
    Design Pattern: Strategy Pattern
    - Configures different model types
    - Provides training parameters
    - Supports hyperparameter tuning
    """
    
    def __init__(self, model_type: str = "MLP", **kwargs):
        self.model_type = model_type.upper()
        self.grid_size = kwargs.get('grid_size', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        self.dataset_path = kwargs.get('dataset_path')
        self.validation_split = kwargs.get('validation_split', 0.2)
        
        # Model-specific parameters
        if self.model_type in ["MLP", "CNN"]:
            self.hidden_layers = kwargs.get('hidden_layers', [64, 32])
            self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        elif self.model_type in ["XGBOOST", "LIGHTGBM"]:
            self.n_estimators = kwargs.get('n_estimators', 100)
            self.max_depth = kwargs.get('max_depth', 6)
        
        print(f"[SupervisedConfig] Initialized {model_type} config")  # Simple logging
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid_models = ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"]
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"Invalid learning rate: {self.learning_rate}")
        
        if self.batch_size < 1:
            raise ValueError(f"Invalid batch size: {self.batch_size}")
        
        if self.epochs < 1:
            raise ValueError(f"Invalid epochs: {self.epochs}")
        
        print(f"[SupervisedConfig] Configuration validated")  # Simple logging
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'model_type': self.model_type,
            'grid_size': self.grid_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'dataset_path': self.dataset_path,
            'validation_split': self.validation_split,
            'hidden_layers': getattr(self, 'hidden_layers', None),
            'dropout_rate': getattr(self, 'dropout_rate', None),
            'n_estimators': getattr(self, 'n_estimators', None),
            'max_depth': getattr(self, 'max_depth', None)
        }
```

### **Reinforcement Learning Configuration**
```python
class ReinforcementConfig:
    """
    Configuration for reinforcement learning algorithms.
    
    Design Pattern: Strategy Pattern
    - Configures different RL algorithms
    - Provides training parameters
    - Supports environment customization
    """
    
    def __init__(self, algorithm: str = "DQN", **kwargs):
        self.algorithm = algorithm.upper()
        self.grid_size = kwargs.get('grid_size', 10)
        self.num_episodes = kwargs.get('num_episodes', 1000)
        self.epsilon_start = kwargs.get('epsilon_start', 0.9)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.reward_apple = kwargs.get('reward_apple', 10)
        self.reward_death = kwargs.get('reward_death', -10)
        self.reward_move = kwargs.get('reward_move', -0.1)
        
        # Algorithm-specific parameters
        if self.algorithm == "DQN":
            self.learning_rate = kwargs.get('learning_rate', 0.001)
            self.memory_size = kwargs.get('memory_size', 10000)
            self.batch_size = kwargs.get('batch_size', 32)
        elif self.algorithm == "PPO":
            self.clip_ratio = kwargs.get('clip_ratio', 0.2)
            self.policy_lr = kwargs.get('policy_lr', 0.0003)
            self.value_lr = kwargs.get('value_lr', 0.0003)
        
        print(f"[ReinforcementConfig] Initialized {algorithm} config")  # Simple logging
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid_algorithms = ["DQN", "PPO", "A3C", "DDPG"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        
        if self.num_episodes < 1:
            raise ValueError(f"Invalid num episodes: {self.num_episodes}")
        
        if self.epsilon_start < 0 or self.epsilon_start > 1:
            raise ValueError(f"Invalid epsilon start: {self.epsilon_start}")
        
        if self.epsilon_decay < 0 or self.epsilon_decay > 1:
            raise ValueError(f"Invalid epsilon decay: {self.epsilon_decay}")
        
        print(f"[ReinforcementConfig] Configuration validated")  # Simple logging
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'num_episodes': self.num_episodes,
            'epsilon_start': self.epsilon_start,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'reward_apple': self.reward_apple,
            'reward_death': self.reward_death,
            'reward_move': self.reward_move,
            'learning_rate': getattr(self, 'learning_rate', None),
            'memory_size': getattr(self, 'memory_size', None),
            'batch_size': getattr(self, 'batch_size', None),
            'clip_ratio': getattr(self, 'clip_ratio', None),
            'policy_lr': getattr(self, 'policy_lr', None),
            'value_lr': getattr(self, 'value_lr', None)
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
        print(f"[ConfigLoader] Initialized for {extension_type}")  # Simple logging
    
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
        
        print(f"[ConfigLoader] Loaded {len(config)} env vars")  # Simple logging
        return config
    
    def _load_json(self, file_path: str) -> dict:
        """Load JSON configuration file"""
        with open(file_path, 'r') as f:
            config = json.load(f)
        print(f"[ConfigLoader] Loaded JSON config from {file_path}")  # Simple logging
        return config
    
    def _load_yaml(self, file_path: str) -> dict:
        """Load YAML configuration file"""
        import yaml
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[ConfigLoader] Loaded YAML config from {file_path}")  # Simple logging
        return config
    
    def _parse_env_value(self, value: str):
        """Parse environment variable value"""
        # Try to convert to appropriate type
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
```

### **Configuration Validation**
```python
class ConfigValidator:
    """
    Comprehensive configuration validation.
    
    Design Pattern: Strategy Pattern
    - Validates different configuration types
    - Provides detailed error messages
    - Supports custom validation rules
    """
    
    def __init__(self):
        self.errors = []
        print(f"[ConfigValidator] Initialized")  # Simple logging
    
    def validate(self, config: dict, config_type: str) -> bool:
        """Validate configuration based on type"""
        self.errors = []
        
        if config_type == 'heuristic':
            self._validate_heuristic_config(config)
        elif config_type == 'supervised':
            self._validate_supervised_config(config)
        elif config_type == 'reinforcement':
            self._validate_reinforcement_config(config)
        else:
            self.errors.append(f"Unknown config type: {config_type}")
        
        if self.errors:
            print(f"[ConfigValidator] Validation failed: {self.errors}")  # Simple logging
            return False
        
        print(f"[ConfigValidator] Validation passed")  # Simple logging
        return True
    
    def _validate_heuristic_config(self, config: dict):
        """Validate heuristic configuration"""
        required_fields = ['algorithm', 'grid_size', 'max_games']
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        if 'algorithm' in config:
            valid_algorithms = ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
            if config['algorithm'] not in valid_algorithms:
                self.errors.append(f"Invalid algorithm: {config['algorithm']}")
        
        if 'grid_size' in config:
            if not isinstance(config['grid_size'], int) or config['grid_size'] < 5 or config['grid_size'] > 50:
                self.errors.append(f"Invalid grid size: {config['grid_size']}")
    
    def _validate_supervised_config(self, config: dict):
        """Validate supervised learning configuration"""
        required_fields = ['model_type', 'grid_size', 'learning_rate', 'batch_size', 'epochs']
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        if 'model_type' in config:
            valid_models = ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"]
            if config['model_type'] not in valid_models:
                self.errors.append(f"Invalid model type: {config['model_type']}")
        
        if 'learning_rate' in config:
            if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
                self.errors.append(f"Invalid learning rate: {config['learning_rate']}")
    
    def _validate_reinforcement_config(self, config: dict):
        """Validate reinforcement learning configuration"""
        required_fields = ['algorithm', 'grid_size', 'num_episodes']
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        if 'algorithm' in config:
            valid_algorithms = ["DQN", "PPO", "A3C", "DDPG"]
            if config['algorithm'] not in valid_algorithms:
                self.errors.append(f"Invalid algorithm: {config['algorithm']}")
        
        if 'num_episodes' in config:
            if not isinstance(config['num_episodes'], int) or config['num_episodes'] < 1:
                self.errors.append(f"Invalid num episodes: {config['num_episodes']}")
    
    def get_errors(self) -> list:
        """Get validation errors"""
        return self.errors.copy()
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Global Configuration**: Constants in ROOT/config/
- [ ] **Extension Configuration**: Extension-specific settings
- [ ] **Runtime Configuration**: Dynamic parameter management
- [ ] **Configuration Factory**: Canonical `create()` method
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
- [ ] **Logging**: Uses simple print statements for debugging
- [ ] **Error Recovery**: Robust error handling and recovery
- [ ] **Serialization**: Support for configuration serialization

---

**Configuration architecture ensures flexible, maintainable, and extensible parameter management across all Snake Game AI extensions. By following these standards, developers can create robust configuration systems that support diverse algorithm requirements while maintaining educational value and technical excellence.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- [ ] **final-decision-10.md**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization