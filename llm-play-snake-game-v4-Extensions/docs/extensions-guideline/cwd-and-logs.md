# Working Directory and Logging Standards

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for working directory and logging standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `standalone.md`, `final-decision-10.md`, `project-structure-plan.md`.

## �� **Core Philosophy: Single Source of Truth**

The Snake Game AI project uses a **unified path management system** centered around the canonical `ensure_project_root()` function in `utils.path_utils`. This system provides predictable file locations and simple logging mechanisms, strictly following SUPREME_RULES from `final-decision-10.md`.

**Critical Rule**: All extensions MUST use `from utils.path_utils import ensure_project_root` - NO custom implementations are allowed.

### **Educational Value**
- **Path Management**: Understanding consistent path handling
- **Logging Standards**: Learning simple, effective logging
- **Single Source of Truth**: Avoiding code duplication and inconsistency
- **Debugging Support**: Easy debugging with consistent paths

## 🏗️ **Working Directory Standards**

### **Canonical Project Root Detection**
```python
# utils/path_utils.py - SINGLE SOURCE OF TRUTH
def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    Project root is identified by the presence of three required directories:
    core/, llm/, and extensions/
    
    Returns:
        The absolute pathlib.Path to the project root directory.
        
    Raises:
        RuntimeError: If project root cannot be found within 10 levels
    """
    # First check if the static _PROJECT_ROOT is valid
    required_dirs = ["core", "llm", "extensions"]
    if all((_PROJECT_ROOT / dir_name).is_dir() for dir_name in required_dirs):
        return _PROJECT_ROOT
    
    # Search upward from current file location
    current = Path(__file__).resolve()
    for _ in range(10):
        if all((current / dir_name).is_dir() for dir_name in required_dirs):
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    raise RuntimeError(
        f"Could not locate project root containing 'core/', 'llm/', and 'extensions/' directories "
        f"within 10 levels from {Path(__file__).resolve()}"
    )

def ensure_project_root() -> Path:
    """
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
    
    # Ensure the project root is at the beginning of sys.path for import precedence
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    return project_root
```

### **✅ MANDATORY Usage Pattern**
```python
# ALL extensions MUST use this exact pattern
from utils.path_utils import ensure_project_root
ensure_project_root()

# Now you can safely use absolute imports
from config.game_constants import DIRECTIONS
from core.game_logic import BaseGameLogic
from utils.moves_utils import position_to_direction
```

### **❌ FORBIDDEN: Custom Implementations**
```python
# ❌ FORBIDDEN: Do NOT create custom _ensure_project_root() functions
def _ensure_project_root():
    """FORBIDDEN - violates single source of truth"""
    current = Path(__file__).resolve()
    # ... any custom path finding logic is FORBIDDEN
    pass

# ❌ FORBIDDEN: Manual path management
import sys
import os
from pathlib import Path

current = Path(__file__).resolve()
for _ in range(10):
    if (current / "config").is_dir():  # FORBIDDEN pattern
        os.chdir(str(current))
        break
    current = current.parent
```

## 📊 **Logging Standards**

### **Simple Print Logging (SUPREME_RULES)**
All logging must use simple print statements. No complex logging frameworks are allowed:

```python
# ✅ CORRECT: Simple print logging (SUPREME_RULES compliance)
from utils.print_utils import print_info, print_warning, print_error, print_success

print_info(f"[GameManager] Starting game {game_id}")
print_info(f"[Agent] Selected move: {move}")
print_info(f"[Game] Score: {score}")

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)
# logger.info("Starting game")
# logger.error("Game failed")
```


## 🚀 **File Organization Standards**

### **Logs Directory Structure**
```
logs/
├── {provider}_{model}_{timestamp}/
│   ├── game_1.json
│   ├── game_2.json
│   ├── ...
│   ├── prompts/
│   │   ├── game_1_round_1_prompt.txt
│   │   ├── game_1_round_2_prompt.txt
│   │   └── ...
│   ├── responses/
│   │   ├── game_1_round_1_raw_response.txt
│   │   ├── game_1_round_2_raw_response.txt
│   │   └── ...
│   └── summary.json
└── extensions/
    ├── datasets/
    │   └── grid-size-{N}/
    │       └── {extension}_{version}_{timestamp}/
    └── models/
        └── grid-size-{N}/
            └── {extension}_{version}_{timestamp}/
```

### **Extension Logs Structure**
```
logs/extensions/datasets/grid-size-10/heuristics_v0.04_20240101_120000/
├── bfs/
│   ├── game_1.json
│   ├── game_2.json
│   ├── summary.json
│   ├── bfs_dataset.csv
│   └── bfs_dataset.jsonl
├── astar/
│   ├── game_1.json
│   ├── summary.json
│   ├── astar_dataset.csv
│   └── astar_dataset.jsonl
└── metadata.json
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Path Management Benefits**
- **Consistency**: Same path handling across all extensions
- **Reliability**: Predictable file locations
- **Single Source of Truth**: One canonical implementation
- **Educational Value**: Learn path management through consistent patterns

### **Logging Benefits**
- **Simplicity**: Simple print statements for all logging
- **Clarity**: Clear, readable log messages
- **Consistency**: Standardized logging format
- **Educational Value**: Learn logging through consistent patterns

## 📋 **SUPREME_RULES Implementation Checklist**

### **Mandatory Requirements**
- [ ] **Single Source of Truth**: Uses ONLY `utils.path_utils.ensure_project_root()` (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all operations (SUPREME_RULES from final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References SUPREME_RULES from final-decision-10.md in all documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all implementations

### **Path-Specific Standards**
- [ ] **No Custom Implementations**: NO custom _ensure_project_root() functions anywhere
- [ ] **Canonical Import**: ALL files use `from utils.path_utils import ensure_project_root`
- [ ] **Project Root Validation**: Validates presence of core/, llm/, extensions/ directories
- [ ] **Cross-Platform**: Works on Windows, macOS, and Linux

---

**Working directory and logging standards ensure consistent path management and simple logging while maintaining SUPREME_RULES compliance and single source of truth across all Snake Game AI extensions.**

## 🔗 **See Also**

- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure standards 