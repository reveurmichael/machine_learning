# Code Elegance Guidelines for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines code elegance guidelines for extensions.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🧹 **File Organization Philosophy**

### **File Length Standards**
- **Target**: ≤ 300–400 lines per Python file
- **Solution**: Split by responsibility when files grow beyond this limit
- **Principle**: One concept per file

### **Directory Organization**
Following `final-decision-5.md` directory structure:
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

### **Standardized Patterns**
> **Authoritative Reference**: See `final-decision-4.md` for complete naming conventions.

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

### **Follow Configuration Standards**
> **Authoritative Reference**: See `config.md` for complete configuration architecture.

```python
# ✅ Universal constants (all tasks)
from config.game_constants import VALID_MOVES, DIRECTIONS

# ✅ Extension-specific constants - define locally
DEFAULT_LEARNING_RATE = 0.001  # Local extension constant

# 🚫 Not for heuristics/supervised/RL/evolutionary extensions
# ✅ Allowed in LLM-centric extensions (agentic-llms, vision-language-model, etc.)
# from config.llm_constants import AVAILABLE_PROVIDERS
```

### **Centralized Settings**
- Use `extensions/common/config/` for extension-specific configurations
- Validate arguments early with clear error messages
- Provide sensible defaults for all parameters

## ⚙️ **Path Management**

### **Mandatory Pattern**
> **Authoritative Reference**: See `unified-path-management-guide.md` for complete path management standards.

```python
# Required for all extensions
from extensions.common.utils.path_utils import (
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

### **OOP Extensibility in Common Utilities**

Following the principle: "The extensions/common/ folder should stay lightweight and generic. Whenever specialised behaviour is required, extensions can inherit from these simple OOP base classes without breaking the core."

**OOP Design for Extensibility:**
```python
# ✅ Base class with extension points
class BaseDatasetLoader(ABC):
    def _initialize_loader_specific_settings(self):
        """Override for specialized loaders"""
        pass
    
    def _generate_extension_specific_metadata(self, data, file_path):
        """Add custom metadata fields"""
        return {}

# ✅ Extension through inheritance
class RLDatasetLoader(BaseDatasetLoader):
    def _initialize_loader_specific_settings(self):
        self.rl_validator = RLValidator()
    
    def _generate_extension_specific_metadata(self, data, file_path):
        return {
            "episode_count": self._count_episodes(data),
            "reward_range": self._calculate_reward_range(data)
        }
```

**Key Benefits:**
- **Most extensions use base classes as-is** (no unnecessary complexity)
- **Specialized extensions can inherit and customize** when needed
- **Protected methods enable selective override** without breaking base functionality
- **Composition patterns support pluggable components** for maximum flexibility

## 📊 **Type Hints and Validation**

### **Type Annotation Standards**
- **Public APIs**: Always type-hinted
- **Internal Functions**: Type-hint where beneficial
- **Avoid Over-Annotation**: Only where you're certain of types
- **Use Union Types**: For parameters accepting multiple types

### **Input Validation**
```python
def create_agent(algorithm: str, grid_size: int) -> BaseAgent:
    """
    Create agent with flexible validation
    
    Educational Note:
    We should be able to add new extensions easily and try out new ideas.
    Therefore, validation is flexible to encourage experimentation.
    """
    # Basic grid size validation (flexible range)
    if grid_size < 5 or grid_size > 50:
        raise ValueError(f"Grid size should be reasonable (5-50), got {grid_size}")
    
    # Algorithm validation through factory pattern (no hard-coded lists)
    try:
        return AgentFactory.create(algorithm, grid_size)  # Canonical create() method
    except KeyError:
        available = AgentFactory.list_available_algorithms()
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
```

## 🎯 **Code Quality Standards**

### **Simplicity Over Complexity**
- **Clear Intent**: Code should be self-documenting
- **Minimal Dependencies**: Avoid unnecessary imports and dependencies
- **Consistent Patterns**: Use established patterns throughout the codebase
- **Educational Value**: Code should teach good practices

### **Performance Considerations**
- **Efficient Algorithms**: Choose appropriate algorithms for the task
- **Memory Management**: Be mindful of memory usage in large datasets
- **Lazy Loading**: Load resources only when needed
- **Caching**: Cache expensive computations when appropriate

### **Error Handling**
- **Graceful Degradation**: Handle errors without crashing
- **Clear Error Messages**: Provide helpful error information
- **Logging**: Use simple print statements for debugging
- **Validation**: Validate inputs early and clearly

---

**Code elegance in extensions is achieved through clear organization, consistent patterns, comprehensive documentation, and thoughtful design. The goal is to create code that is not only functional but also educational and maintainable.**

## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`factory-design-pattern.md`**: Factory pattern implementation


## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`factory-design-pattern.md`**: Factory pattern implementation
