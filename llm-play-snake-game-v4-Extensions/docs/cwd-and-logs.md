# Working Directory and Logging Management

## Overview

This document explains the working directory management strategy and logging directory structure used throughout the project to ensure consistent behavior across different execution contexts.

## Working Directory Management

### Repository Root as Working Directory

All scripts in the project (both Task-0 and extensions) now use a consistent pattern to ensure they run from the repository root directory:

1. **ROOT/scripts/** files (Task-0): Use `ensure_project_root()` utility
2. **Extensions**: Use inline repo root finder + `os.chdir()`

This ensures that relative paths (like `logs/`) behave consistently regardless of where the user launches the script from.

### Implementation Pattern

#### Task-0 Scripts (ROOT/scripts/)

```python
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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

### Benefits

1. **Consistent Path Resolution**: All relative paths work the same way
2. **Subprocess Compatibility**: Child processes inherit the correct working directory
3. **Streamlit Integration**: Web apps can reliably find assets and logs
4. **Cross-Platform**: Works on Windows, macOS, and Linux
5. **IDE Independence**: Works regardless of IDE working directory settings

## Logging Directory Structure

### Task-0 (First-Class Citizen)

Task-0 logs are stored directly under `logs/` as the primary, production-ready implementation:

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

### Extensions (Second-Class Citizens)

Extension logs are isolated under `logs/extensions/` to separate experimental implementations:

```
logs/
├── extensions/
│   ├── heuristics-bfs_20250623_090525/
│   │   ├── game_1.json
│   │   └── summary.json
│   ├── heuristics-astar_20250623_091234/
│   │   ├── game_1.json
│   │   └── summary.json
│   └── ...
└── hunyuan-t1-latest_20250617_223807/  # Task-0 logs
    ├── game_1.json
    └── summary.json
```

### Extension Logging Implementation

Each extension's `GameManager._setup_logging()` method creates logs under `logs/extensions/`:

```python
def _setup_logging(self):
    """Set up logging for the heuristic extension.
    
    Logs are stored under logs/extensions/ to separate experimental
    implementations from the primary Task-0 logs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"heuristics-{self.algorithm}_{timestamp}"
    
    # Create logs/extensions/ directory structure
    logs_dir = pathlib.Path("logs") / "extensions" / session_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    self.log_dir = str(logs_dir)
    # ... rest of logging setup
```

## Rationale

### Why Change Working Directory?

1. **Path Consistency**: Relative paths like `logs/` always resolve correctly
2. **Subprocess Safety**: Child processes inherit the correct working directory
3. **Streamlit Integration**: Web apps can reliably find templates and static files
4. **IDE Independence**: Works regardless of how the IDE sets the working directory
5. **Cross-Platform**: Consistent behavior on all operating systems

### Why Separate Extension Logs?

1. **Clear Separation**: Task-0 is the primary implementation, extensions are experimental
2. **No Pollution**: Extension experiments don't clutter the main logs directory
3. **Easy Cleanup**: Can easily remove all extension logs without affecting Task-0
4. **Future-Proof**: Room for more extension types (RL, etc.) without confusion
5. **Backup Strategy**: Can backup Task-0 logs separately from experimental logs

## Migration Notes

- **Task-0 scripts**: Already use `ensure_project_root()` utility
- **Extensions**: Use inline repo root finder + `os.chdir()` pattern
- **Log locations**: Task-0 → `logs/`, Extensions → `logs/extensions/`
- **Backward compatibility**: All existing functionality preserved
- **Documentation**: Updated to reflect new patterns and structure 