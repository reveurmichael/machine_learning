# AI-Friendly Development Guidelines

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`` → `final-decision.md`) and defines AI-friendly development guidelines.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Explicit is Better Than Implicit**

The codebase follows principles that make it highly readable and navigable for AI development assistants, strictly adhering to SUPREME_RULES from `final-decision.md`.

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
from utils.factory_utils import SimpleFactory
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
        agent = GameAgentFactory.create("BFS", config)  # Canonical create() method per SUPREME_RULES
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
- `llm_constants.py` - LLM-specific configurations (whitelisted extensions only)

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

## 🎯 **AI-Friendly Design Principles**

### **1. Explicit Over Implicit**
- Clear naming conventions
- Explicit type hints
- Obvious control flow

### **2. Consistency Over Cleverness**
- Predictable patterns
- Standard library usage
- Familiar idioms

### **3. Documentation Over Code**
- Rich docstrings
- Clear comments
- Architecture explanations

### **4. Simplicity Over Complexity**
- Single responsibility principle
- Clear interfaces
- Minimal dependencies

---

**These AI-friendly development guidelines ensure that the codebase remains accessible, understandable, and maintainable for both human developers and AI development assistants. The focus on explicit patterns, comprehensive documentation, and consistent structure makes the project an excellent learning resource and development platform.**

## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Factory pattern implementation
