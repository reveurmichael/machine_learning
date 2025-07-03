# Final Decision 2: Configuration, Validation, and Architectural Standards

> **SUPREME AUTHORITY**: This document establishes the definitive architectural standards for configuration organization, validation systems, and structural decisions across all Snake Game AI extensions.

> **See also:** `config.md` (Configuration standards), `validation.md` (Validation patterns), `naming-conventions.md` (Naming standards), `final-decision-10.md` (SUPREME_RULES).


## 🔧 **DECISION 1: Configuration Organization**

### **Finalized Structure**

```
ROOT/config/               # Task-0 specific (LLM-related configs)
├── game_constants.py      # ✅ Universal game rules (used by all tasks)
├── ui_constants.py        # ✅ Universal UI settings (used by all tasks) 
├── llm_constants.py       # 🚫 General extensions must not import; ✅ LLM-focused extensions (agentic-llms, vision-language-model, llm-finetune, etc.) may use
├── prompt_templates.py    # ❌ Task-0 only (LLM prompts); ✅ LLM-focused extensions (agentic-llms, vision-language-model, llm-finetune, etc.) may use
├── network_constants.py   # ✅ Universal HTTP/WebSocket settings (used by all tasks)
└── web_constants.py       # ✅ Universal Flask Web settings (used by all tasks)

extensions/common/config/  # Extension-specific configurations
├── __init__.py
├── dataset_formats.py     # Data format specifications
├── path_constants.py      # Directory path templates
└── validation_rules.py    # Validation thresholds and rules

# Note: Following SUPREME_RULE NO.3, we avoid patterns like:
# ml_constants.py, training_defaults.py, model_registry.py
# Instead, define extension-specific constants locally in each extension
```

### **Usage Patterns**

```python
# ✅ Universal constants (used by all tasks)
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH

# ✅ Extension-specific constants (SUPREME_RULE NO.3: define locally in extensions)
# Local constants in each extension instead of importing from common config
DEFAULT_LEARNING_RATE = 0.001
BATCH_SIZES = [16, 32, 64, 128]
EARLY_STOPPING_PATIENCE = 10

# ✅ Common utilities (lightweight, generic)
from extensions.common.config.dataset_formats import CSV_SCHEMA_VERSION

# ❌ Task-0 only (extensions should NOT import these)
# from config.llm_constants import AVAILABLE_PROVIDERS  # 🚫 Forbidden for non-LLM extensions
# from config.prompt_templates import SYSTEM_PROMPT     # 🚫 Forbidden for non-LLM extensions
```

## 🎯 **DECISION 4: File Naming Conventions**

### **Finalized Standards**

| Component Type | File Pattern | Class Pattern | Example |
|----------------|--------------|---------------|---------|
| **Agents** | `agent_*.py` | `*Agent` | `agent_bfs.py` → `BFSAgent` |
| **Game Logic** | `game_*.py` | `*GameLogic` | `game_heuristic.py` → `HeuristicGameLogic` |
| **Controllers** | `*_controller.py` | `*Controller` | `game_controller.py` → `GameController` |
| **Managers** | `*_manager.py` | `*Manager` | `game_manager.py` → `GameManager` |
| **Validators** | `*_validator.py` | `*Validator` | `dataset_validator.py` → `DatasetValidator` |
| **Factories** | `*_factory.py` | `*Factory` | `agent_factory.py` → `AgentFactory` |

### **Implementation Example**

```python
# ✅ CORRECT: Following naming conventions
# File: agent_bfs.py
class BFSAgent(BaseAgent):
    """Breadth-First Search agent implementation"""
    pass

# File: game_heuristic.py
class HeuristicGameLogic(BaseGameLogic):
    """Heuristic-based game logic implementation"""
    pass

# File: dataset_validator.py
class DatasetValidator:
    """Dataset validation implementation"""
    pass

# ❌ FORBIDDEN: Inconsistent naming
# File: bfs_agent.py (should be agent_bfs.py)
# File: heuristic_game.py (should be game_heuristic.py)
```