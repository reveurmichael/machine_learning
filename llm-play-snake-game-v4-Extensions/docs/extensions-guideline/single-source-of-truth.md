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

## 📚 **Common Utilities**

### ALL THINGS WITHIN THE FOLDER ROOT/config

TODO: HERE IS A LIST OF SSOT THINGS EXTENSIONS SHOULD ADOPT:
- TODO LIST
- TODO LIST
- TODO LIST
- TODO LIST
- TODO LIST
- TODO LIST
- TODO LIST
- TODO LIST

## TODO: IMPORTANT: use ROOT/utils/ stuffs
Here is a list of ROOT/utils/ stuffs (functions) that can and SHOULD be reused in extensions.
TODO: LIST
TODO: LIST
TODO: LIST
TODO: LIST
TODO: LIST
TODO: LIST

### **Shared Components in extensions/common/**
```python
# ✅ GOOD: Truly common utilities go in extensions/common/
extensions/common/
├── csv_schema.py          # CSV schema for all ML tasks
├── dataset_loader.py      # Dataset loading utilities
├── config.py             # Extension-specific configurations
└── validation.py         # Data validation utilities

# These are shared across extensions but NOT between extension versions
```

### **Usage Pattern**
```python
# ✅ Each extension version uses common utilities
# heuristics-v0.03 uses extensions/common/csv_schema.py
# supervised-v0.02 uses extensions/common/dataset_loader.py
# But heuristics-v0.03 does NOT import from supervised-v0.02
```

## 🔍 **Validation and Enforcement**

### **Automated Checks**
```python
# Example validation in extensions/common/validation.py # TODO: or maybe a lot valiations scripts in the folder common/validation
def validate_game_state_schema(game_state: Dict) -> bool:
    """Validate that game state follows SSOT schema"""
    required_fields = [
        'grid_size', 'snake_positions', 'apple_position',
        'current_direction', 'score', 'steps', 'game_active'
    ]
    
    for field in required_fields:
        if field not in game_state:
            raise ValueError(f"Missing required field: {field}")
    
    return True

def validate_coordinate_system(position: Tuple[int, int], grid_size: int) -> bool:
    """Validate position follows SSOT coordinate system"""
    x, y = position
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        raise ValueError(f"Position {position} outside valid range for grid {grid_size}")
    
    return True
```

# TODO: or maybe a lot valiations scripts in the folder common/validation

### **Documentation Consistency**
```python
# ✅ GOOD: Single source documentation
# docs/extensions-guideline/coordinate-system.md  (authoritative)
# All other docs reference this, don't duplicate

# ❌ BAD: Multiple coordinate system explanations
# Different docs saying different things about coordinates
```

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









