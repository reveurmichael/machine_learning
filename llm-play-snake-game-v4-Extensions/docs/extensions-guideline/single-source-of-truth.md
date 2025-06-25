# Single Source of Truth Principle

This document outlines the **Single Source of Truth (SSOT)** principle as applied throughout the Snake Game AI project, ensuring consistency, maintainability, and reliability across all components.

## 🎯 **Core Principle**

**Unless it's between different extensions** (each extension, plus the common folder, are regarded as standalone), we should go for **single source of truth**.

### **Definition**
Single Source of Truth means that every piece of data, configuration, or business logic should have exactly one authoritative representation within the system. This eliminates contradictions, reduces maintenance burden, and ensures consistency.

## 🏗️ **Application Areas**

### **1. Configuration Management**
All configuration constants are centralized in the `config/` directory:

```python
# ✅ GOOD: Single source in config/game_constants.py
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
DIRECTIONS = {
    "UP": (0, 1),
    "DOWN": (0, -1), 
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}
MAX_STEPS_ALLOWED = 1000

# ✅ All extensions import from this single source
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
```

```python
# ❌ BAD: Duplicated constants across files
# File 1: VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
# File 2: MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]  # Duplicate!
# File 3: DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]  # Another duplicate!
```

### **2. Coordinate System**
The entire codebase uses **one single coordinate system** defined in `docs/extensions-guideline/coordinate-system.md`:

```python
# ✅ GOOD: Consistent coordinate system everywhere
# Origin (0,0) at bottom-left, x grows right, y grows up
# Used by: core logic, agents, prompts, GUI adapters, extensions
```

### **3. Game State Schema**
Game state structure is defined once and reused everywhere:

```python
# ✅ GOOD: Single game state schema in core/game_data.py
class BaseGameData:
    def get_state_snapshot(self) -> Dict:
        return {
            'grid_size': self.grid_size,
            'snake_positions': self.snake_positions.copy(),
            'apple_position': self.apple_position,
            'current_direction': self.current_direction,
            'score': self.score,
            'steps': self.steps,
            'game_active': self.game_active
        }

# ✅ All extensions use this same schema
# Heuristics, RL, Supervised Learning all get consistent data
```

### **4. CSV Schema for ML**
Dataset schema is centralized in `extensions/common/csv_schema.py`:

```python
# ✅ GOOD: Single CSV schema definition
from extensions.common.csv_schema import generate_csv_schema, TabularFeatureExtractor

# Used by:
# - Heuristics v0.03 (dataset generation)
# - Supervised Learning v0.02+ (training)
# - All ML models (consistent feature extraction)
```

### **5. File Naming Conventions**
Naming rules are documented once and applied consistently:

```python
# ✅ GOOD: Single naming convention
# Files: game_*.py in core/
# Classes: PascalCase (GameManager, BFSAgent)
# Functions: snake_case (make_move, get_state)
# Constants: UPPER_SNAKE_CASE (MAX_STEPS, VALID_MOVES)
```

## 🚫 **Extension Boundaries**

### **When SSOT Does NOT Apply**
Extensions are designed to be **standalone** (extension + common folder), so they can have their own versions of certain components:

```python
# ✅ ALLOWED: Each extension can have its own GameManager
# ROOT/core/game_manager.py        (Task-0 LLM version)
# extensions/heuristics-v0.02/game_manager.py  (Heuristic version)
# extensions/supervised-v0.02/game_manager.py  (ML version)

# Each inherits from BaseGameManager but adds domain-specific logic
```

### **Extension Independence**
```python
# ✅ GOOD: Extensions don't share code between each other
# heuristics-v0.02 + common = standalone
# supervised-v0.02 + common = standalone
# reinforcement-v0.02 + common = standalone

# ❌ BAD: Cross-extension dependencies
# from heuristics_v0.02 import some_utility  # FORBIDDEN
# from supervised_v0.01 import helper_function  # FORBIDDEN
```

## 📚 **Authoritative SSOT Locations**

### 1. Configuration Constants (ROOT/config/)
These modules are *the* authority for project-wide constants.  Import; never redefine.

| File | Purpose |
|------|---------|
| `game_constants.py` | Game-rule numbers, movement maps, sentinel moves, step limits |
| `ui_constants.py`   | Universal colour palette, grid size default, window dimensions |
| `llm_constants.py`  | **Task-0 & LLM-focused extensions only** – provider names, model aliases, token limits |
| `prompt_templates.py` | System / user prompt skeletons (LLM tasks only) |
| `network_constants.py` | HTTP / WebSocket defaults for scripts & dashboards |
| `web_constants.py` | Flask / Streamlit specific settings |

> For an expanded rationale and hierarchy diagram see **`config.md`** and **`final-decision-2.md`**.

### 2. Universal Utilities (ROOT/utils/)
Stateless helper functions that *every* task and extension may (and should) reuse:

| Module | Key Responsibilities |
|--------|---------------------|
| `board_utils.py` | Apple placement, board generation, vacant-cell queries |
| `collision_utils.py` | Wall / body collision tests, reverse-move detection |
| `moves_utils.py` | Direction parsing, normalization, convenience enums |
| `json_utils.py` | Safe JSON read/write, pretty printing, schema sanity checks |
| `path_utils.py` | **Lightweight helpers only** (heavyweight project-root logic now lives in `extensions/common/path_utils.py`) |
| `seed_utils.py` | Project-wide RNG seed control for reproducible runs |
| `text_utils.py` | Markdown / console colouring, padding, wrapping |
| `web_utils.py` | Board-state → JSON for browser front-ends |

### 3. Path Management (Single Source of Truth)
All non-trivial path logic is consolidated in **`final-decision-6.md`** and its implementation `extensions/common/path_utils.py`.  *Do not* copy `ensure_project_root()` or sibling helpers into extension folders – just import them.

### 4. Shared Extension Utilities (extensions/common/)
Cross-extension helpers that don't belong in ROOT:

* `csv_schema.py`, `dataset_loader.py` – dataset definitions & loaders  
* `validation/` sub-package – reusable data / directory validators  
* `config/` sub-package – hyper-parameters used by multiple extension families

## 🔄 **How to Contribute Without Breaking SSOT**
1. **Look first** – search for an existing constant/function before adding a new one.
2. **Prefer import over copy-paste** – keep behaviour changes in one spot.
3. **If truly new:** place it in the *single* correct location (table above) and document it.

> 🛑 **Checklist** – Each time before writing the code, ask yourself:  
> • Am I duplicating any constant already defined in `ROOT/config/` or `ROOT/extensions/common/`?  
> • Could this helper live in `ROOT/utils/` or `ROOT/extensions/common/`?  
> • Does the documentation reference the new SSOT location?

## 🎯 **Benefits Achieved**

### **1. Consistency**
- All components use the same coordinate system
- All ML models use the same feature schema
- All extensions follow the same naming conventions

### **2. Maintainability**
- Change configuration in one place, affects entire system
- Update game state schema once, all components benefit
- Fix coordinate system bug in one location

### **3. Reliability**
- No contradictory information across codebase
- Reduced chance of integration bugs
- Clear ownership of each piece of data

### **4. Developer Experience**
- New developers learn one way of doing things
- Clear documentation hierarchy
- Predictable code organization

## 🚀 **Implementation Guidelines**

### **For Core Components**
1. **Identify authoritative source** for each piece of information
2. **Document the source location** clearly
3. **Import, don't duplicate** in other files
4. **Validate consistency** with automated tests

### **For Extensions**
1. **Inherit from base classes** rather than reimplementing
2. TODO: IMPORTANT: use ROOT/utils/ stuffs

2. **Use common utilities** from extensions/common/
3. **Don't share code** between different extensions
4. **Document extension-specific** deviations clearly

### **For Documentation**
1. **One authoritative document** per topic
2. **Cross-reference, don't duplicate** information
3. **Update single source** when changes occur
4. **Validate documentation** against implementation

## 📋 **Checklist for SSOT Compliance**

### **Before Adding New Code:**
- [ ] Is this information already defined elsewhere?
- [ ] Can I import/inherit instead of reimplementing?
- [ ] Am I creating a new authoritative source?
- [ ] Is the authoritative location documented?

### **Before Adding New Documentation:**
- [ ] Does this information exist in another document?
- [ ] Should I reference existing docs instead?
- [ ] Am I creating the authoritative source for this topic?
- [ ] Are cross-references updated?

---

**The Single Source of Truth principle ensures that the Snake Game AI project remains maintainable, consistent, and reliable as it grows in complexity across multiple extensions and algorithm types.**









