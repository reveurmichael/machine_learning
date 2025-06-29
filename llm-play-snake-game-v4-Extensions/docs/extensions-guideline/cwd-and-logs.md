# Working Directory and Logging Standards

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for working directory and logging standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `standalone.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Consistent Path Management**

The Snake Game AI project uses a **unified path management system** that ensures consistent working directories and logging across all extensions. This system provides predictable file locations and simple logging mechanisms, strictly following SUPREME_RULES from final-decision-10.md.

### **Educational Value**
- **Path Management**: Understanding consistent path handling
- **Logging Standards**: Learning simple, effective logging
- **File Organization**: Clear file organization patterns
- **Debugging Support**: Easy debugging with consistent paths

## 🏗️ **Working Directory Standards**

### **Project Root Detection**
```python
# extensions/common/utils/path_utils.py
import os
from pathlib import Path

def ensure_project_root():
    """Ensure working directory is set to project root"""
    current_dir = Path.cwd()
    
    # Look for project root indicators
    root_indicators = ['app.py', 'README.md', 'config/']
    
    # Navigate up until we find project root
    while current_dir != current_dir.parent:
        if any((current_dir / indicator).exists() for indicator in root_indicators):
            os.chdir(current_dir)
            print(f"[PathUtils] Set working directory to: {current_dir}")  # SUPREME_RULES compliant logging
            return current_dir
        current_dir = current_dir.parent
    
    # If not found, stay in current directory
    print(f"[PathUtils] Project root not found, staying in: {Path.cwd()}")  # SUPREME_RULES compliant logging
    return Path.cwd()

def get_project_root() -> Path:
    """Get project root path"""
    current_dir = Path.cwd()
    
    # Look for project root indicators
    root_indicators = ['app.py', 'README.md', 'config/']
    
    # Navigate up until we find project root
    while current_dir != current_dir.parent:
        if any((current_dir / indicator).exists() for indicator in root_indicators):
            return current_dir
        current_dir = current_dir.parent
    
    return Path.cwd()
```

### **Extension Path Management**
```python
def get_extension_path(extension_type: str, version: str) -> Path:
    """Get path for specific extension"""
    project_root = get_project_root()
    extension_path = project_root / "extensions" / f"{extension_type}-{version}"
    
    if not extension_path.exists():
        raise ValueError(f"Extension path not found: {extension_path}")
    
    return extension_path

def get_logs_path() -> Path:
    """Get logs directory path"""
    project_root = get_project_root()
    return project_root / "logs"

def get_datasets_path() -> Path:
    """Get datasets directory path"""
    project_root = get_project_root()
    return project_root / "extensions" / "datasets"

def get_models_path() -> Path:
    """Get models directory path"""
    project_root = get_project_root()
    return project_root / "extensions" / "models"
```

## 📊 **Logging Standards**

### **Simple Print Logging (SUPREME_RULES)**
All logging must use simple print statements. No complex logging frameworks are allowed:

```python
# ✅ CORRECT: Simple print logging (SUPREME_RULES compliance)
print(f"[GameManager] Starting game {game_id}")
print(f"[Agent] Selected move: {move}")
print(f"[Game] Score: {score}")

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)
# logger.info("Starting game")
# logger.error("Game failed")
```

### **Logging Format Standards**
```python
def log_info(component: str, message: str):
    """Standard info logging format"""
    print(f"[{component}] {message}")

def log_error(component: str, message: str):
    """Standard error logging format"""
    print(f"[{component}] ERROR: {message}")

def log_debug(component: str, message: str):
    """Standard debug logging format"""
    print(f"[{component}] DEBUG: {message}")

# Usage examples
log_info("GameManager", "Starting new game")
log_error("Agent", "Invalid move detected")
log_debug("Pathfinding", "Calculating route to apple")
```

### **Component-Specific Logging**
```python
class GameManager:
    def __init__(self):
        self.component_name = "GameManager"
        print(f"[{self.component_name}] Initialized")  # SUPREME_RULES compliant logging
    
    def start_game(self):
        print(f"[{self.component_name}] Starting new game")  # SUPREME_RULES compliant logging
        # Game logic here
        print(f"[{self.component_name}] Game completed")  # SUPREME_RULES compliant logging
    
    def log_error(self, message: str):
        print(f"[{self.component_name}] ERROR: {message}")  # SUPREME_RULES compliant logging

class Agent:
    def __init__(self, name: str):
        self.component_name = f"Agent_{name}"
        print(f"[{self.component_name}] Initialized")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state: dict) -> str:
        print(f"[{self.component_name}] Planning move")  # SUPREME_RULES compliant logging
        # Move planning logic here
        move = "UP"  # Example
        print(f"[{self.component_name}] Selected move: {move}")  # SUPREME_RULES compliant logging
        return move
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
logs/extensions/datasets/grid-size-10/heuristics_v0.03_20240101_120000/
├── bfs_data.csv
├── astar_data.csv
├── dfs_data.csv
└── metadata.json

logs/extensions/models/grid-size-10/supervised_v0.03_20240101_120000/
├── mlp_model.pth
├── xgboost_model.pkl
├── training_results.json
└── evaluation_results.json
```

## 📋 **Implementation Examples**

### **Extension Path Management**
```python
# In any extension
from extensions.common.utils.path_utils import ensure_project_root, get_extension_path

def setup_extension():
    """Setup extension with proper paths"""
    # Ensure we're in project root
    project_root = ensure_project_root()
    
    # Get extension-specific paths
    extension_path = get_extension_path("heuristics", "v0.03")
    logs_path = project_root / "logs"
    
    print(f"[Extension] Project root: {project_root}")  # SUPREME_RULES compliant logging
    print(f"[Extension] Extension path: {extension_path}")  # SUPREME_RULES compliant logging
    print(f"[Extension] Logs path: {logs_path}")  # SUPREME_RULES compliant logging
    
    return project_root, extension_path, logs_path
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Path Management Benefits**
- **Consistency**: Same path handling across all extensions
- **Reliability**: Predictable file locations
- **Debugging**: Easy debugging with consistent paths
- **Educational Value**: Learn path management through consistent patterns

### **Logging Benefits**
- **Simplicity**: Simple print statements for all logging
- **Clarity**: Clear, readable log messages
- **Consistency**: Standardized logging format
- **Educational Value**: Learn logging through consistent patterns

## 📋 **SUPREME_RULES Implementation Checklist**

### **Mandatory Requirements**
- [ ] **Simple Logging**: Uses print() statements only for all operations (SUPREME_RULES from final-decision-10.md compliance)
- [ ] **Path Consistency**: All extensions use same path management utilities
- [ ] **GOOD_RULES Reference**: References SUPREME_RULES from final-decision-10.md in all documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all implementations

### **Path-Specific Standards**
- [ ] **Project Root Detection**: Automatic project root detection
- [ ] **Extension Paths**: Standardized extension path management
- [ ] **Logs Organization**: Consistent logs directory structure
- [ ] **File Naming**: Standardized file naming conventions

---

**Working directory and logging standards ensure consistent path management and simple logging while maintaining SUPREME_RULES compliance and educational value across all Snake Game AI extensions.**

## 🔗 **See Also**

- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure standards 