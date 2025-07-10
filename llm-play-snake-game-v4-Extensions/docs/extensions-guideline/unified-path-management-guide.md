# Unified Path Management Guide for Snake Game AI

## �� **Core Philosophy: Single Source of Truth**

All extensions **MUST** use the canonical `ensure_project_root()` function from `utils.path_utils` to ensure reliable cross-platform operation and eliminate path-related bugs. This function is the **ONLY** implementation that should be used across ALL extensions and scripts.

### **Educational Value**
- **Cross-Platform Compatibility**: Consistent behavior across Windows, macOS, and Linux
- **Development Workflow**: Works regardless of IDE working directory
- **Deployment Reliability**: Same code works in dev and production
- **Single Source of Truth**: One canonical implementation eliminates duplication

## 🛠️ **Mandatory Path Management Pattern**

### **Required Setup for All Extensions**
```python
# MANDATORY USAGE PATTERN FOR ALL EXTENSIONS
from utils.path_utils import ensure_project_root
from utils.print_utils import print_info, print_warning, print_error, print_success

# Ensure project root is set and properly configured
ensure_project_root()

# Now you can use absolute imports
from config.game_constants import DIRECTIONS
from core.game_logic import BaseGameLogic
from utils.moves_utils import position_to_direction
```

### **❌ FORBIDDEN: Custom _ensure_project_root() Implementations**
```python
# ❌ FORBIDDEN: Do NOT create custom implementations
def _ensure_project_root():
    """Custom implementation - FORBIDDEN"""
    # Any custom implementation violates single source of truth
    pass

# ❌ FORBIDDEN: Do NOT duplicate path management logic
import sys
import os
from pathlib import Path

current = Path(__file__).resolve()
# ... custom path finding logic ... FORBIDDEN
```

## 📁 **Core Path Utilities**

### **Project Root Management**
```python
def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    This is a pure function with no side effects, suitable for situations
    where you only need the path without changing the working directory.
    
    Project root is identified by the presence of three required directories:
    core/, llm/, and extensions/
    
    Returns:
        The absolute pathlib.Path to the project root directory.
        
    Raises:
        RuntimeError: If project root cannot be found within 10 levels
    """

def ensure_project_root() -> Path:
    """
    Ensures the current working directory is the project root and that the
    root directory is in sys.path for absolute imports.

    This function validates the project root by checking for required directories
    (core/, llm/, extensions/) and searches upward if needed.

    Single Source of Truth: This is the ONLY function that should be used
    for project root detection across ALL extensions and scripts.

    This function has intentional side effects:
    - Changes the current working directory (os.chdir)
    - Modifies sys.path to enable absolute imports
    - Prints a message if the directory is changed

    Returns:
        The absolute pathlib.Path to the project root directory.
    """
    project_root = get_project_root()
    current_dir = Path.cwd()
    
    if current_dir != project_root:
        print_info(f"[PathUtils] Changing working directory to project root: {project_root}")
        os.chdir(project_root)
    
    return project_root
```

## 🔧 **Extension Implementation Patterns**

### **Standard Extension Pattern**
```python
# extensions/heuristics-v0.04/agents/agent_bfs.py
from utils.path_utils import ensure_project_root
ensure_project_root()

# Now you can safely use absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error

class BFSAgent:
    def __init__(self):
        self.algorithm_name = "BFS"
    
    def get_move(self, game) -> str:
        # Agent implementation here
        pass
```

### **Extension Entry Point Pattern**
```python
# extensions/heuristics-v0.04/scripts/main.py
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from ..game_manager import HeuristicGameManager
from ..agents import AgentFactory

def main():
    """Main entry point for heuristics v0.04"""
    # Extension logic here
    pass
```

## 📊 **Simple Logging Standards for Path Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All path operations use simple print statements as established in `final-decision.md`:

```python
# ✅ CORRECT: Simple logging for path operations (SUPREME_RULES compliance)
def ensure_project_root() -> Path:
    project_root = get_project_root()
    current_dir = Path.cwd()
    
    if current_dir != project_root:
        print_info(f"[PathUtils] Changing working directory to project root: {project_root}")
        os.chdir(project_root)
    
    return project_root

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging  # FORBIDDEN
# logger = logging.getLogger(__name__)  # FORBIDDEN
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Path Management Benefits**
- **Cross-Platform Compatibility**: Consistent behavior across all operating systems
- **Development Workflow**: Works regardless of IDE working directory
- **Deployment Reliability**: Same code works in dev and production
- **Single Source of Truth**: One canonical implementation eliminates confusion

### **Pattern Consistency**
- **Canonical Method**: All extensions use identical path management
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Predictable patterns enable easier learning
- **SUPREME_RULES**: Path management follows same standards as other components

## 📋 **SUPREME_RULES Implementation Checklist for Path Management**

### **Mandatory Requirements**
- [ ] **Single Source of Truth**: Uses ONLY `utils.path_utils.ensure_project_root()` (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all path operations (final-decision.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision.md` in all path management documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all path implementations

### **Path-Specific Standards**
- [ ] **Cross-Platform**: Works on Windows, macOS, and Linux
- [ ] **Working Directory**: Proper working directory management
- [ ] **Path Validation**: Validates presence of core/, llm/, extensions/ directories
- [ ] **Error Handling**: Simple logging for all error conditions

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical patterns
- [ ] **Pattern Documentation**: Clear explanation of path management benefits
- [ ] **SUPREME_RULES Compliance**: All examples follow final-decision.md standards
- [ ] **Cross-Reference**: Links to related patterns and principles

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision.md`**: SUPREME_RULES for canonical path patterns
- **`core.md`**: Core architecture and path integration
- **`project-structure-plan.md`**: Project structure standards

### **Implementation Files**
- **`utils/path_utils.py`**: Canonical path utilities (SINGLE SOURCE OF TRUTH)
- **All extensions**: Use `from utils.path_utils import ensure_project_root`

### **Educational Resources**
- **Single Source of Truth**: Path management demonstrates importance of avoiding duplication
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **Cross-Platform Compatibility**: Path management works reliably everywhere

