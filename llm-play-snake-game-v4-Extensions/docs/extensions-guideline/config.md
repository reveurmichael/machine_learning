# Configuration Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`) and `final-decision-2.md`, establishing comprehensive configuration management standards.

## 🎯 **Configuration Philosophy: Single Source of Truth with Clear Separation**

The Snake Game AI configuration architecture implements a sophisticated separation model that ensures consistency, prevents pollution, and enables scalable extension development:

- **Universal Constants**: Core game rules and shared settings in `ROOT/config/`
- **Extension-Specific Constants**: ML, training, and extension utilities in `extensions/common/config/`
- **Task-0 Isolation**: LLM-specific constants properly isolated with controlled access
- **Access Control**: Explicit whitelisting prevents inappropriate cross-domain dependencies

## 📁 **Final Configuration Structure**

### **Universal Constants (ROOT/config/)**
```python
# ✅ Used by ALL tasks (Task-0 and all extensions)
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
```

### **Task-0 Specific Constants (ROOT/config/)**
```python
# 🚫 FORBIDDEN for general-purpose extensions
# ✅ ALLOWED ONLY for explicit LLM-focused extensions (see whitelist below)
from config.llm_constants import AVAILABLE_PROVIDERS
from config.prompt_templates import SYSTEM_PROMPT
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

## 🚫 **LLM Constants Access Control**

### **Explicit Whitelist: ONLY These Extensions May Use LLM Constants**

**✅ ALLOWED Extensions:**
- `agentic-llms-*` (any version)
- `llms-*` (any version) 
- `llm-*` (any version)
- `llm-finetune-*` (any version)
- `vision-language-model-*` (any version)

**❌ FORBIDDEN Extensions:**
- `heuristics-*` (all versions) - both v0.03 and v0.04 are widely used
- `supervised-*` (all versions)
- `reinforcement-*` (all versions)
- `evolutionary-*` (all versions)
- `distillation-*` (all versions)
- Any other extension not explicitly listed above

### **Clear Usage Patterns**

#### **Extensions That MUST Use Universal Constants Only**
```python
# ✅ CORRECT for: heuristics, supervised, reinforcement, evolutionary
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
from extensions.common.config.training_defaults import EARLY_STOPPING_PATIENCE

# 🚫 FORBIDDEN for these extensions
# from config.llm_constants import AVAILABLE_PROVIDERS  # ❌ NOT ALLOWED
# from config.prompt_templates import SYSTEM_PROMPT     # ❌ NOT ALLOWED
```

#### **Extensions That MAY Use LLM Constants**
```python
# ✅ ALLOWED for: agentic-llms, llms, llm, llm-finetune, vision-language-model
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
from config.llm_constants import AVAILABLE_PROVIDERS  # ✅ ALLOWED
from config.prompt_templates import SYSTEM_PROMPT     # ✅ ALLOWED
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

# ✅ Universal network settings
from config.network_constants import HTTP_TIMEOUT

# ✅ Universal web settings
from config.web_constants import FLASK_CONFIG

# ✅ Extension-specific settings
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
from extensions.common.config.training_defaults import EARLY_STOPPING_PATIENCE
```

### **Extensions Must NOT Use (General Rule – *except* explicit whitelist)**
```python
# 🚫 FORBIDDEN for: heuristics, supervised, reinforcement, evolutionary extensions
from config.llm_constants import AVAILABLE_PROVIDERS
from config.prompt_templates import SYSTEM_PROMPT

# ✅ ALLOWED ONLY for: agentic-llms-*, llms-*, llm-*, llm-finetune-*, vision-language-model-*
from config.llm_constants import AVAILABLE_PROVIDERS
from config.prompt_templates import SYSTEM_PROMPT
```

## 📊 **Benefits for Extension Types**

### **Heuristics Extensions (v0.03 and v0.04)**
- Access to universal movement rules and coordinate system
- Consistent visualization across all heuristic algorithms
- Extension-specific pathfinding constants and optimization settings
- Both v0.03 and v0.04 are widely used depending on use cases and scenarios

### **Supervised Learning Extensions**
- Universal game constants for feature engineering
- Shared ML hyperparameters and training configurations
- Consistent model evaluation metrics and thresholds

### **Reinforcement Learning Extensions**
- Universal game rules for environment definition
- Shared RL training parameters and exploration settings
- Consistent reward function definitions and normalization

### **Evolutionary Extensions**
- Universal game constants for fitness evaluation
- Shared evolutionary parameters and population settings
- Consistent genetic operator configurations

## 🔍 **Validation Requirements**

All extensions MUST validate configuration compliance:
```python
from extensions.common.validation import validate_config_access

def validate_extension_config(extension_type: str, imported_modules: List[str]):
    """Validate extension configuration access compliance"""
    validate_config_access(extension_type, imported_modules)
```

## 🔗 **See Also**

- **`final-decision-2.md`**: Authoritative reference for configuration architecture decisions
- **`extensions/common/config/`**: Extension-specific configuration constants
- **`config/`**: Universal configuration constants

---

**This configuration architecture ensures clean separation, prevents pollution, and enables scalable extension development.**

## **🏗️ Perfect Base Class Architecture Already in Place**

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

## 🏗️ **Configuration Hierarchy & Boundaries**

> **Educational Note:** A clear configuration hierarchy prevents "constant-sprawl" and makes it obvious where new settings belong. This section formalises the boundary rules already hinted at elsewhere.

### **1. Hierarchy Overview**
| Level | Path | Purpose |
|-------|------|---------|
| **Universal** | `config/` | Core game rules, UI, coordinate system, network, web – *used by every task & extension* |
| **Shared Extension** | `extensions/common/config/` | Settings reused by **multiple** extensions (e.g. CSV schema, training defaults) |
| **Type-Specific** | `extensions/{type}/config/` | Settings unique to one algorithm family (heuristics, supervised, rl, llm, …) |
| **Experiment / Script** | Local to script | Hyper-parameters that change per run (CLI flags, YAML, etc.) |

### **2. LLM Constants Rule**
* **Universal constant folder (`config/`)** must **NOT** contain LLM-specific settings.
* **LLM-focused extensions** (e.g. *agentic-llms*, *vision-language-model*) may import from:
  ```python
  from config.llm_constants import AVAILABLE_PROVIDERS
  from config.prompt_templates import SYSTEM_PROMPT
  ```
* Non-LLM extensions must never depend on LLM constants – this enforces loose coupling.

### **3. Examples**
**Heuristics extension** – needs only universal constants:
```python
from config.game_constants import DIRECTIONS, VALID_MOVES
from config.ui_constants import COLORS, GRID_SIZE
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
```

**Supervised extension** – universal + shared ML defaults:
```python
from config.game_constants import GRID_SIZE
from config.ui_constants import COLORS
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE
```

**Agentic LLM extension** – universal + shared + LLM-specific:
```python
from config.game_constants import VALID_MOVES
from config.ui_constants import COLORS
from config.network_constants import HTTP_TIMEOUT
from config.web_constants import FLASK_CONFIG
from config.llm_constants import AVAILABLE_PROVIDERS  # ✅ allowed here
from config.prompt_templates import SYSTEM_PROMPT     # ✅ allowed here
```

### **4. Validation Helpers**
```python
# extensions/common/validation/config_validator.py

def validate_constant_access(module_name: str, constant_name: str) -> None:
    """Raise if an extension imports forbidden constants."""
    forbidden = (
        module_name.startswith('config.llm_constants') and
        not __name__.startswith('extensions.llm')
    )
    if forbidden:
        raise ImportError(
            f"{constant_name} is Task-0 specific and cannot be imported outside LLM extensions")
```

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




