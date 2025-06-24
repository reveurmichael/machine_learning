# AI-Friendly Development Guidelines

This document outlines how the Snake Game AI project is designed to be AI development assistant-friendly, making it easier for AI tools to understand, navigate, and contribute to the codebase.

## 🎯 **Core Philosophy: Explicit is Better Than Implicit**

The codebase follows principles that make it highly readable and navigable for AI development assistants:

### **1. Self-Documenting Code Structure**
- **Meaningful file names**: `game_controller.py`, `agent_bfs.py`, `replay_engine.py`
- **Clear directory hierarchy**: `core/`, `extensions/`, `gui/`, `web/`
- **Consistent naming patterns**: All core files use `game_*.py` pattern
- **Descriptive class names**: `BaseGameManager`, `LLMAgent`, `HeuristicGameLogic`

### **2. Comprehensive Documentation Strategy**
```python
# Every class includes purpose and usage examples
class ConsecutiveLimitsManager:
    """
    Centralizes tracking and enforcement of consecutive move limits in Snake game.
    
    Design Patterns Used:
    - Facade Pattern: Provides simple interface to complex limit tracking
    - Strategy Pattern: Pluggable enforcement policies
    - Template Method Pattern: Consistent move processing workflow
    
    Usage:
        manager = create_limits_manager(args)
        status = manager.check_and_update(LimitType.EMPTY_MOVES)
        if status.limit_exceeded:
            handle_limit_exceeded()
    """
```

### **3. Architecture Documentation**
- **Base class hierarchies clearly documented**
- **Design patterns explicitly named and explained**
- **Inheritance relationships with UML-style comments**
- **Extension points clearly marked**

## 🔍 **Navigation Aids for AI Assistants**

### **File Organization Patterns**
```
ROOT/
├── core/               # Base classes and core logic
│   ├── game_*.py      # All core files follow this pattern
│   └── __init__.py    # Exports for easy importing
├── extensions/        # Task-specific implementations
│   ├── common/        # Shared utilities across extensions
│   ├── heuristics-v0.*/  # Versioned algorithm implementations
│   └── supervised-v0.*/  # Versioned ML implementations
├── config/            # All configuration constants
├── docs/              # Comprehensive documentation
│   └── extensions-guideline/  # Extension-specific guides
└── scripts/           # Entry points and utilities
```

### **Import Structure Clarity**
```python
# Clear, hierarchical imports
from core.game_manager import BaseGameManager
from config.game_constants import VALID_MOVES, DIRECTIONS
from extensions.common.dataset_loader import load_dataset_for_training
```

### **Design Pattern Documentation**
Every design pattern used is explicitly documented:
```python
class GameAgentFactory:
    """
    Factory Pattern Implementation
    
    Purpose: Create appropriate agent instances based on configuration
    Benefits: 
    - Loose coupling between client code and concrete agents
    - Easy to add new agent types
    - Centralized agent creation logic
    
    Usage:
        agent = GameAgentFactory.create_agent("BFS", config)
    """
```

## 📚 **Documentation as Code**

### **Inline Architecture Diagrams**
```python
class BaseGameManager:
    """
    Template Method Pattern:
    
    BaseGameManager (abstract)
         ↓ (inherits)
    GameManager (Task-0, LLM-specific)
         ↓ (parallel inheritance)
    HeuristicGameManager (Task-1)
    RLGameManager (Task-2)
    SupervisedGameManager (Task-3)
    """
```

### **Comprehensive Type Hints**
```python
def create_limits_manager(
    args: argparse.Namespace
) -> ConsecutiveLimitsManager:
    """Type hints make function signatures crystal clear for AI tools"""
```

### **Rich Docstring Standards**
- **Purpose**: What the class/function does
- **Design Patterns**: Which patterns are implemented and why
- **Parameters**: Full type and description information
- **Returns**: Expected return types and meanings
- **Raises**: Possible exceptions and when they occur
- **Examples**: Concrete usage examples

## 🧭 **AI Assistant Navigation Features**

### **1. README Structure**
Each major directory has a README explaining:
- Purpose and scope
- Key files and their roles
- How to extend or modify
- Related documentation

### **2. Cross-Reference System**
```python
# Clear references to related components
class HeuristicGameLogic(BaseGameLogic):
    """
    See also:
    - BaseGameLogic: Parent class defining interface
    - HeuristicGameManager: Uses this logic class
    - agents/: Specific algorithm implementations
    - docs/extensions-guideline/heuristics.md: Detailed guide
    """
```

### **3. Extension Points Marked**
```python
class BaseGameManager:
    # 🔌 EXTENSION POINT: Override for task-specific game logic
    GAME_LOGIC_CLS = None  # Set in subclasses
    
    # 🔌 EXTENSION POINT: Override for task-specific initialization
    def initialize_task_specific_components(self):
        """Hook method for subclasses to add components"""
        pass
```

## 🎨 **Code Style for AI Readability**

### **Consistent Patterns**
- **File naming**: `agent_*.py`, `game_*.py`, `*_controller.py`
- **Class naming**: `Base*`, `*Manager`, `*Agent`, `*Controller`
- **Method naming**: `setup_*`, `handle_*`, `create_*`, `get_*`

### **Clear Separation of Concerns**
```python
# Each file has a single, clear responsibility
# game_controller.py - ONLY game state management
# game_logic.py - ONLY game rules and move planning
# game_manager.py - ONLY session/game lifecycle management
```

### **Explicit State Management**
```python
class GameStateAdapter:
    """
    Adapter Pattern: Converts between different state representations
    
    Purpose: Eliminate code duplication across multiple components
    that need game state access in different formats
    """
```

## 🚀 **AI Development Workflow Support**

### **1. Clear Entry Points**
```python
# scripts/main.py - Primary entry point for Task-0
# extensions/heuristics-v0.03/app.py - Streamlit app entry
# extensions/supervised-v0.02/scripts/train.py - ML training entry
```

### **2. Configuration Centralization**
All constants in `config/` directory with clear names:
- `game_constants.py` - Core game rules
- `ui_constants.py` - Display and interface settings
- `llm_constants.py` - LLM-specific configurations


## 🔧 **Tools and Utilities for AI Assistance**

### **1. Type Checking Support**
- Full mypy compatibility
- Type hints throughout codebase
- Clear interface definitions

### **2. Linting Configuration**
- pylint.sh, ruff.sh for code quality
- Consistent formatting standards
- Clear error messages

### **3. Development Scripts**
```bash
# Clear utility scripts
./mypy.sh         # Type checking
./pylint.sh       # Code quality
./rm_pycache.sh   # Cleanup
```

## 📖 **Documentation Strategy**

### **1. Layered Documentation**
- **README.md**: High-level project overview
- **docs/**: Detailed technical documentation
- **Inline comments**: Implementation-specific details
- **Docstrings**: API documentation

### **2. Extension Guidelines**
Comprehensive guides for each extension type:
- `extensions-v0.01.md` - Foundation patterns
- `extensions-v0.02.md` - Multi-algorithm expansion
- `extensions-v0.03.md` - Web interface integration
- `extensions-v0.04.md` - Language-rich datasets

### **3. Architecture Documentation**
- `core.md` - Base class architecture
- `mvc.md` - Model-View-Controller patterns
- `coordinate-system.md` - Consistent coordinate handling

## 🎯 **Best Practices for AI-Friendly Code**

### **1. Explicit Over Implicit**
```python
# ✅ Explicit and clear
class HeuristicGameManager(BaseGameManager):
    """Manages heuristic-based Snake gameplay using pathfinding algorithms"""
    GAME_LOGIC_CLS = HeuristicGameLogic  # Clear factory pattern

# ❌ Implicit and unclear
class Manager(Base):
    logic = Logic()
```

### **2. Rich Context in Comments**
```python
# ✅ Provides context and reasoning
def calculate_path(self, start: Position, goal: Position) -> List[Direction]:
    """
    Calculate optimal path using A* algorithm.
    
    The heuristic function uses Manhattan distance which is admissible
    for our grid-based movement system where diagonal moves are not allowed.
    
    Args:
        start: Current snake head position
        goal: Apple position to reach
        
    Returns:
        List of directions forming optimal path, or empty list if no path exists
        
    Design Note:
        Uses A* instead of Dijkstra because we have a good heuristic (Manhattan
        distance) that significantly reduces search space.
    """
```



## 📋 **AI Assistant Checklist**

When working with this codebase, AI assistants can quickly orient by checking:

- [ ] **File purpose**: Check docstring at top of file
- [ ] **Class responsibility**: Read class docstring for design patterns
- [ ] **Extension points**: Look for `# 🔌 EXTENSION POINT:` comments
- [ ] **Related documentation**: Check `docs/` for detailed guides
- [ ] **Configuration**: Check `config/` for relevant constants
- [ ] **Examples**: Look in `extensions/` for usage patterns

This structure ensures that AI development assistants can quickly understand the codebase architecture, locate relevant files, and contribute effectively to the project.
