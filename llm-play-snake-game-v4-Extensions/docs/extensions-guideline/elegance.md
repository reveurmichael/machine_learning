# Code Elegance Guidelines for Extensions

> **Important — Authoritative Reference:** This document supplements the Final Decision Series for extension development standards.

## 🧹 **File Organization Philosophy**

### **File Length Standards**
- **Target**: ≤ 300–400 lines per Python file
- **Solution**: Split by responsibility when files grow beyond this limit
- **Principle**: One concept per file

### **Directory Organization**
Following Final Decision 5 directory structure:
```
extensions/{algorithm}-v0.0N/
├── agents/                 # Algorithm implementations (v0.02+)
├── dashboard/              # UI components (v0.03+)
├── scripts/                # CLI entry points (v0.03+)
├── utils/                  # Extension-specific utilities
└── core files              # game_logic.py, game_manager.py, etc.
```

### **Modular Boundaries**
- **Clear Separation**: Business logic, data access, UI components
- **Avoid Circular Imports**: Group interdependent code appropriately
- **Clean APIs**: Re-export through `__init__.py` for external consumption
- **Shared Utilities**: Use `extensions/common/` for cross-extension functionality

## 🎨 **Naming Conventions**

### **Standardized Patterns (Final Decision 4)**
```python
# ✅ File naming
agent_bfs.py              # Algorithm implementations
game_logic.py             # Core extension files
tab_main.py               # Dashboard components

# ✅ Class naming
class BFSAgent(BaseAgent)        # Algorithm agents
class HeuristicGameLogic         # Extension-specific classes
class ConfigurationManager      # Utility classes
```

### **Consistency Rules**
- **Classes**: `PascalCase` (e.g., `GameManager`, `BFSAgent`)
- **Functions & Variables**: `snake_case` (e.g., `compute_path`, `max_steps`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_STEPS_ALLOWED`)
- **No One-Letter Variables**: Except in very local loops

## 📚 **Documentation Standards**

### **Required Documentation**
- **Modules**: Purpose and usage summary at top
- **Classes**: Purpose, design patterns used, key methods
- **Functions**: Arguments, return values, side effects, exceptions
- **Design Patterns**: Explicit documentation of patterns used

### **Docstring Quality**
```python
class PathfindingAgent(BaseAgent):
    """
    Base class for pathfinding algorithms in Snake game.
    
    Design Pattern: Template Method Pattern
    - Defines common pathfinding structure
    - Subclasses implement specific algorithms
    - Ensures consistent interface across pathfinding variants
    
    Educational Note:
    Demonstrates how different pathfinding algorithms can share
    common infrastructure while implementing different strategies.
    """
```

## 🔧 **Configuration Management**

### **Follow Final Decision 2**
```python
# ✅ Universal constants (all tasks)
from config.game_constants import VALID_MOVES, DIRECTIONS

# ✅ Extension-specific constants
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE

# ❌ Task-0 specific (forbidden in extensions)
# from config.llm_constants import AVAILABLE_PROVIDERS
```

### **Centralized Settings**
- Use `extensions/common/config/` for extension-specific configurations
- Validate arguments early with clear error messages
- Provide sensible defaults for all parameters

## ⚙️ **Path Management**

### **Mandatory Pattern (Final Decision 6)**
```python
# Required for all extensions
from extensions.common.path_utils import (
    ensure_project_root,
    get_dataset_path,
    get_model_path
)

# Standard setup
project_root = ensure_project_root()
```

## 🚀 **Design Philosophy**

### **No Backward Compatibility Burden**
- **Future-Proof Mindset**: Fresh, modern approach
- **No Deprecation**: Remove outdated code completely
- **Clean Architecture**: No legacy considerations for extensions

### **Educational Excellence**
- **Pattern Documentation**: Explain why each design pattern is used
- **Clear Examples**: Provide concrete usage examples
- **Progressive Complexity**: From simple v0.01 to sophisticated v0.03

### **OOP and SOLID Principles**
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Open for extension, closed for modification
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

## 📊 **Type Hints and Validation**

### **Type Annotation Standards**
- **Public APIs**: Always type-hinted
- **Internal Functions**: Type-hint where beneficial
- **Avoid Over-Annotation**: Only where you're certain of types
- **Use Union Types**: For parameters accepting multiple types

### **Input Validation**
```python
def create_agent(algorithm: str, grid_size: int) -> BaseAgent:
    """Create agent with proper validation"""
    if grid_size < 8 or grid_size > 20:
        raise ValueError(f"Grid size must be 8-20, got {grid_size}")
    
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
```

## 🔍 **Code Quality Tools**

### **Linting and Formatting**
- **PEP8 Compliance**: Use linters like `flake8`, `pylint`
- **Line Length**: 88 characters (Black default)
- **Import Organization**: Standard library, third-party, local imports
- **Trailing Whitespace**: Remove consistently

---

**These elegance guidelines ensure maintainable, educational, and scalable extension development while following the architectural decisions established in the Final Decision Series.**

