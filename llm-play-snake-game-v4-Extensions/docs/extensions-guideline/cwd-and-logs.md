# Working Directory and Path Management

> **Important — Authoritative Reference:** This document supplements Final Decision 6. The path management utilities are mandatory for all extensions.

This document explains the working directory management strategy and logging directory structure used throughout the project to ensure consistent behavior across different execution contexts.

All extensions **MUST** use standardized path management from `extensions/common/path_utils.py` as established in Final Decision 6. This ensures reliable cross-platform operation and eliminates path-related bugs.

## 📁 **Mandatory Path Management Pattern**

## Working Directory Management

### Repository Root as Working Directory

All scripts in the project (both Task-0 and extensions) now use a consistent pattern to ensure they run from the repository root directory:

1. **ROOT/scripts/** files (Task-0): Use `ensure_project_root()` utility
2. **Extensions**: Use inline repo root finder + `os.chdir()`

This ensures that relative paths (like `logs/`) behave consistently regardless of where the user launches the script from.

### Implementation Pattern

#### Task-0 Scripts (ROOT/scripts/)

```python
# MANDATORY for all extensions
from extensions.common.path_utils import (
    ensure_project_root,
    get_extension_path,
    get_dataset_path,
    get_model_path
)

from utils.path_utils import ensure_project_root

# ------------------
# Ensure current working directory == repository root
# ------------------
REPO_ROOT = ensure_project_root()
```

The `ensure_project_root()` utility:
- Changes working directory to repo root (`os.chdir()`)
- Ensures repo root is in `sys.path`
- Prints a message if directory was changed
- Returns the repo root path

#### Extension Scripts

```python
import sys
import os
import pathlib

# Find repo root and add to sys.path
def find_repo_root():
    current = pathlib.Path(__file__).resolve()
    while current.parent != current:
        if (current / "README.md").exists() and (current / "core").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")

project_root = find_repo_root()
sys.path.insert(0, str(project_root))

# Change working directory to repo root
os.chdir(project_root)
```

### **Benefits of Standardized Path Management**
- **Consistent Behavior**: Relative paths work the same way everywhere
- **Cross-Platform**: Works on Windows, macOS, and Linux reliably
- **Subprocess Safety**: Child processes inherit correct working directory
- **IDE Independence**: Works regardless of IDE working directory settings

## 🗂️ **Logging Directory Structure**

### **Task-0 Logs (First-Class)**
```
logs/
├── hunyuan-t1-latest_20250617_223807/
│   ├── game_1.json
│   ├── game_2.json
│   ├── summary.json
│   ├── prompts/
│   └── responses/
├── deepseek-reasoner_20250618_003933/
│   ├── game_1.json
│   ├── summary.json
│   ├── prompts/
│   └── responses/
└── ...
```

### **Extension Logs (Organized)**
```
logs/extensions/
├── datasets/                      # Per Final Decision 1
│   └── grid-size-N/
│       ├── heuristics_v0.03_{timestamp}/
│       ├── supervised_v0.02_{timestamp}/
│       └── reinforcement_v0.02_{timestamp}/
├── models/                        # Per Final Decision 1
│   └── grid-size-N/
│       ├── supervised_v0.02_{timestamp}/
│       └── reinforcement_v0.02_{timestamp}/
└── [execution logs by extension]
```



## 🚀 **Benefits**

### **For Development**
- **Predictable Paths**: All relative paths resolve consistently
- **Easy Debugging**: Clear working directory management
- **Reliable Testing**: Tests run consistently across environments

### **For Deployment**
- **Container Compatibility**: Works in Docker and other containers
- **CI/CD Reliability**: Consistent behavior in automation pipelines
- **Multi-Environment**: Same code works in development and production

### **For Users**
- **No Setup Required**: Works regardless of launch directory
- **Consistent Experience**: Same behavior across different platforms
- **Error Prevention**: Reduces path-related user errors

---

**This path management strategy ensures reliable, consistent, and maintainable directory operations across all extensions.** 