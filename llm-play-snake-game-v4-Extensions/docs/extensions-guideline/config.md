# Configuration Architecture for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for configuration architecture and supplements the _Final Decision Series_ (`` → `final-decision.md`).

> **See also:** `core.md`, `standalone.md`, `final-decision.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Flexible Configuration Management**

Configuration in the Snake Game AI project follows a hierarchical, extensible architecture that supports both simple parameter management and complex multi-extension configurations. The system is designed to be lightweight, educational, and maintainable while supporting the diverse needs of different algorithm types, strictly following SUPREME_RULES from `final-decision.md`.

### **Educational Value**
- **Configuration Patterns**: Demonstrates best practices for parameter management
- **Hierarchical Design**: Shows how to structure complex configurations
- **Extensibility**: Framework for adding new configuration types
- **Validation**: Proper parameter validation and error handling

## 🏗️ **Factory Pattern: Canonical Method is create()**

All configuration factories must use the canonical method name `create()` for instantiation, not `create_config()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision.md`.

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
print_info(f"Config name: {config.name}")  # SUPREME_RULES compliant logging
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
        print_info(f"[ConfigFactory] Creating config: {config_type}")  # SUPREME_RULES compliant logging
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

---

**Configuration architecture ensures flexible, maintainable, and extensible parameter management across all Snake Game AI extensions. By following these standards, developers can create robust configuration systems that support diverse algorithm requirements while maintaining educational value and technical excellence.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization