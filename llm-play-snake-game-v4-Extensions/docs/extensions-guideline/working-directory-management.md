# Working Directory Management Strategy

TODO: chdir() a lot ! That's what we want to have. With chdir(), we can make things easier to manage.

## Overview

This document explains the critical importance of working directory management in our project and how the `os.chdir(project_root)` pattern solves multiple challenges related to data management, subprocess integration, and cross-platform compatibility.

## The Problem

When scripts can be launched from arbitrary locations (e.g., from IDE, terminal, subprocess), several issues arise:

1. **Inconsistent Path Resolution**: Relative paths like `logs/` resolve differently based on launch location
2. **Subprocess Failures**: Child processes inherit the wrong working directory
3. **Streamlit Integration Issues**: Web apps can't reliably find assets and logs
4. **Cross-Platform Problems**: Different OS behaviors with relative paths
5. **IDE Dependencies**: Scripts only work when launched from specific directories

## The Solution: `os.chdir(project_root)`

### Core Strategy

All scripts in the project (both Task-0 and extensions) ensure they run from the repository root directory:

```python
# Find repository root and change working directory
project_root = find_repo_root()
os.chdir(project_root)
```

This guarantees that relative paths always resolve consistently, regardless of where the script is launched from.

## Implementation Patterns

### Task-0 Scripts (ROOT/scripts/)

Task-0 scripts use the `ensure_project_root()` utility:

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

### Extension Scripts

Extension scripts use inline repo root finder:

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

## Benefits for Data Management

### 1. Consistent Log Output Locations

**Before**: Logs could end up anywhere depending on launch location
```
# Launched from extensions/heuristics-v0.01/
logs/heuristics-bfs_20250623_123456/  # ✅ Correct

# Launched from project root
logs/heuristics-bfs_20250623_123456/  # ✅ Correct

# Launched from arbitrary directory
/Users/user/Desktop/logs/heuristics-bfs_20250623_123456/  # ❌ Wrong!
```

**After**: Logs always go to the correct location
```
# Any launch location → logs always in ROOT/logs/
ROOT/logs/extensions/heuristics-bfs_20250623_123456/  # ✅ Always correct
```

### 2. Predictable File Paths

All relative paths now resolve consistently:

```python
# These always work, regardless of launch location:
logs_dir = "logs/extensions/"  # ✅ Always ROOT/logs/extensions/
config_file = "config/game_constants.py"  # ✅ Always ROOT/config/game_constants.py
templates_dir = "web/templates/"  # ✅ Always ROOT/web/templates/
```

### 3. Simplified Backup and Analytics

With consistent log locations, backup scripts become trivial:

```bash
# Backup all Task-0 logs
rsync -av logs/ backup/task0_logs/

# Backup all extension logs
rsync -av logs/extensions/ backup/extension_logs/

# Run analytics on all heuristics experiments
python analyze_heuristics.py logs/extensions/
```

## Benefits for Subprocess Integration

### 1. Streamlit/Flask Subprocess Calls

Web applications can easily spawn any extension without worrying about working directories:

```python
# Streamlit app can call any extension reliably
import subprocess

# Task-0
subprocess.run(["python", "scripts/main.py", "--max-games", "1"])

# Heuristics v0.01
subprocess.run(["python", "extensions/heuristics-v0.01/main.py", "--max-games", "1"])

# Heuristics v0.02
subprocess.run(["python", "extensions/heuristics-v0.02/main.py", "--algorithm", "astar", "--max-games", "1"])

# Heuristics v0.03
subprocess.run(["python", "extensions/heuristics-v0.03/scripts/main.py", "--max-games", "1"])
```

**All scripts work identically** regardless of their location in the file tree.

### 2. Child Process Inheritance

Child processes automatically inherit the correct working directory:

```python
# Parent process changes to repo root
os.chdir(project_root)

# Child process inherits correct working directory
subprocess.run(["python", "some_script.py"])  # ✅ Runs from repo root
```

### 3. Cross-Platform Compatibility

The pattern works consistently across all platforms:

```python
# Windows, macOS, Linux - all work the same
project_root = find_repo_root()
os.chdir(project_root)  # ✅ Works everywhere
```

## Benefits for Web Integration

### 1. Flask/Streamlit Asset Discovery

Web applications can reliably find templates and static files:

```python
# Flask app can find templates regardless of launch location
app = Flask(__name__, 
            template_folder="web/templates",  # ✅ Always ROOT/web/templates/
            static_folder="web/static")       # ✅ Always ROOT/web/static/
```

### 2. Log File Access

Web interfaces can easily access log files for replay and analysis:

```python
# Web app can list and access logs reliably
logs_dir = "logs/extensions/"  # ✅ Always correct location
log_files = os.listdir(logs_dir)
```

### 3. Configuration File Access

Web apps can load configuration without path issues:

```python
# Load config from predictable location
config_path = "config/game_constants.py"  # ✅ Always ROOT/config/game_constants.py
```

## Log Directory Structure

### Task-0 (First-Class Citizen)

Task-0 logs are stored directly under `logs/` as the primary, production-ready implementation:

```
ROOT/logs/
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
ROOT/logs/
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

## Real-World Examples

### Example 1: IDE Development

**Scenario**: Developer launches script from IDE with different working directory

```python
# IDE working directory: /Users/dev/projects/snake-game/extensions/heuristics-v0.01/
# Script location: /Users/dev/projects/snake-game/extensions/heuristics-v0.01/main.py

# Without os.chdir():
logs_dir = "logs/extensions/"  # ❌ Creates /Users/dev/projects/snake-game/extensions/heuristics-v0.01/logs/extensions/

# With os.chdir():
os.chdir(project_root)  # Changes to /Users/dev/projects/snake-game/
logs_dir = "logs/extensions/"  # ✅ Creates /Users/dev/projects/snake-game/logs/extensions/
```

### Example 2: Streamlit Integration

**Scenario**: Streamlit app spawns multiple extensions

```python
# Streamlit app working directory: /Users/dev/projects/snake-game/
# Extension location: /Users/dev/projects/snake-game/extensions/heuristics-v0.02/main.py

# Without os.chdir():
subprocess.run(["python", "extensions/heuristics-v0.02/main.py"])
# ❌ Extension creates logs in wrong location

# With os.chdir():
subprocess.run(["python", "extensions/heuristics-v0.02/main.py"])
# ✅ Extension automatically changes to repo root and creates logs in correct location
```

### Example 3: Cross-Platform Deployment

**Scenario**: Deploying on different operating systems

```python
# Windows: C:\projects\snake-game\
# macOS: /Users/dev/projects/snake-game/
# Linux: /home/dev/projects/snake-game/

# Without os.chdir():
# ❌ Different behavior on each platform

# With os.chdir():
project_root = find_repo_root()  # ✅ Finds correct root on any platform
os.chdir(project_root)           # ✅ Changes to correct directory on any platform
```

## Migration and Maintenance

### Adding New Scripts

When adding new scripts, always include the working directory setup:

```python
# For Task-0 scripts (ROOT/scripts/)
from utils.path_utils import ensure_project_root
REPO_ROOT = ensure_project_root()

# For extension scripts
def find_repo_root():
    # ... implementation ...
project_root = find_repo_root()
os.chdir(project_root)
```

### Testing the Pattern

Verify the pattern works by testing from different locations:

```bash
# Test from project root
cd /path/to/project
python scripts/main.py

# Test from subdirectory
cd /path/to/project/extensions/heuristics-v0.01
python ../../scripts/main.py

# Test from arbitrary location
cd /tmp
python /path/to/project/scripts/main.py
```

All should produce logs in the same location: `ROOT/logs/`

## Conclusion

The `os.chdir(project_root)` pattern is essential for:

1. **Consistent Data Management**: All outputs go to predictable locations
2. **Robust Subprocess Integration**: Web apps can spawn any script reliably
3. **Cross-Platform Compatibility**: Works consistently across all operating systems
4. **IDE Independence**: Scripts work regardless of IDE working directory settings
5. **Simplified Deployment**: No need to manage complex path configurations

This pattern makes the entire project more maintainable, testable, and deployable while ensuring that Task-0 remains the first-class citizen with its logs in `ROOT/logs/` and extensions are properly isolated under `ROOT/logs/extensions/`. 