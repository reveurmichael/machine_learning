# Configuration Architecture for Snake Game AI Extensions

## 🎯 **Core Philosophy: Flexible Configuration Management**

Configuration in the Snake Game AI project follows a hierarchical, extensible architecture that supports both simple parameter management and complex multi-extension configurations. The system is designed to be lightweight, educational, and maintainable while supporting the diverse needs of different algorithm types.

## 🏗️ **Configuration Hierarchy**

### **1. Global Configuration (ROOT/config/)**

The project uses a centralized configuration system with clear separation of concerns:

```python
# config/game_constants.py - Core game rules
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
GRID_SIZES = [8, 10, 12, 16, 20]
DEFAULT_GRID_SIZE = 10

# config/llm_constants.py - LLM-specific settings (whitelisted extensions only)
DEFAULT_MODEL = "hunyuan"
DEFAULT_PROVIDER = "deepseek"

# config/network_constants.py - Network and communication settings
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3

# config/prompt_templates.py - Prompt templates for LLM interactions
SYSTEM_PROMPT = "You are a snake game AI assistant..."
```

### **2. Extension-Specific Configuration**
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

# extensions/reinforcement-v0.03/reinforcement_config.py
RL_ALGORITHMS = ["DQN", "PPO", "A3C"]
DEFAULT_EPSILON_START = 0.9
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_REWARD_APPLE = 10
DEFAULT_REWARD_DEATH = -10
```


## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization