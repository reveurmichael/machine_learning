# Configuration Architecture

> **Important — Authoritative Reference:** This document supplements the Final Decision Series. Where conflicts exist, Final Decision 2 prevails.

## 🎯 **Configuration Philosophy**

The configuration architecture follows a **clear separation model** established in Final Decision 2:
- **Universal constants** (used by all tasks) in `ROOT/config/`
- **Extension-specific constants** in `extensions/common/config/`
- **Task-0 specific constants** remain isolated

## 📁 **Final Configuration Structure**

### **Universal Constants (ROOT/config/)**
```python
# ✅ Used by ALL tasks (Task-0 and all extensions)
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
```

### **Task-0 Specific Constants (ROOT/config/)**
```python
# ❌ FORBIDDEN in extensions - Task-0 only
from config.llm_constants import AVAILABLE_PROVIDERS
from config.prompt_templates import SYSTEM_PROMPT
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
```

### **Extension-Specific Constants**
```
extensions/common/config/
├── ml_constants.py        # ML-specific hyperparameters, thresholds
├── training_defaults.py   # Default training configurations
├── dataset_formats.py     # Data format specifications
├── path_constants.py      # Directory path templates
├── validation_rules.py    # Validation thresholds and rules
└── model_registry.py      # Model type definitions and metadata
```

## 🧠 **Design Benefits**

### **Clear Boundaries**
- **Universal constants** ensure consistency across all tasks
- **LLM constants** remain isolated to prevent extension pollution
- **Extension constants** enable shared functionality without coupling

### **Single Source of Truth**
- Each constant has exactly one authoritative location
- No duplication between Task-0 and extensions
- Clear import patterns prevent accidental dependencies

### **Scalability**
- Easy to add new extension-specific configurations
- Universal constants automatically available to new extensions
- Clean separation enables independent evolution

## 🔧 **Usage Patterns**

### **Extensions Should Use**
```python
# ✅ Universal game rules
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED

# ✅ Universal UI settings
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH

# ✅ Extension-specific settings
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
from extensions.common.config.training_defaults import EARLY_STOPPING_PATIENCE
```

### **Extensions Must NOT Use (General Rule)**
```python
# ❌ Task-0 specific - FORBIDDEN in most extensions
from config.llm_constants import AVAILABLE_PROVIDERS
from config.prompt_templates import SYSTEM_PROMPT
```

**Exception**: LLM-focused extensions (agentic-llms, vision-language-model, llm-finetune) MAY use LLM constants when implementing LLM functionality.

## 📊 **Benefits for Extension Types**

### **Heuristics Extensions**
- Access to universal movement rules and coordinate system
- Consistent visualization across all heuristic algorithms
- Extension-specific pathfinding constants and optimization settings

### **Supervised Learning Extensions**
- Universal game constants for feature engineering
- Shared ML hyperparameters and training configurations
- Consistent model evaluation metrics and thresholds

### **Reinforcement Learning Extensions**
- Universal game rules for environment definition
- Shared RL training parameters and exploration settings
- Consistent reward function definitions and normalization

---

**This configuration architecture ensures clean separation, prevents pollution, and enables scalable extension development.**

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




