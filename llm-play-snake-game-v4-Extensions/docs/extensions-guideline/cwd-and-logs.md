# Working Directory and Logging Standards

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for working directory and logging standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `standalone.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Consistent Path Management**

The Snake Game AI project uses a **unified path management system** that ensures consistent working directories and logging across all extensions. This system provides predictable file locations and simple logging mechanisms, strictly following `final-decision-10.md` SUPREME_RULES.

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
            print(f"[PathUtils] Set working directory to: {current_dir}")  # Simple logging
            return current_dir
        current_dir = current_dir.parent
    
    # If not found, stay in current directory
    print(f"[PathUtils] Project root not found, staying in: {Path.cwd()}")  # Simple logging
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
        print(f"[{self.component_name}] Initialized")  # Simple logging
    
    def start_game(self):
        print(f"[{self.component_name}] Starting new game")  # Simple logging
        # Game logic here
        print(f"[{self.component_name}] Game completed")  # Simple logging
    
    def log_error(self, message: str):
        print(f"[{self.component_name}] ERROR: {message}")  # Simple logging

class Agent:
    def __init__(self, name: str):
        self.component_name = f"Agent_{name}"
        print(f"[{self.component_name}] Initialized")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        print(f"[{self.component_name}] Planning move")  # Simple logging
        # Move planning logic here
        move = "UP"  # Example
        print(f"[{self.component_name}] Selected move: {move}")  # Simple logging
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
from extensions.common.utils.path_utils import ensure_project_root, get_logs_path

def main():
    """Main function with proper path management"""
    # Ensure we're in project root
    ensure_project_root()
    
    # Get paths
    logs_path = get_logs_path()
    print(f"[Main] Logs path: {logs_path}")  # Simple logging
    
    # Create timestamp for this run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = logs_path / "extensions" / "datasets" / f"grid-size-10" / f"heuristics_v0.03_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Main] Output directory: {output_dir}")  # Simple logging
    
    # Run extension logic
    run_extension(output_dir)

def run_extension(output_dir: Path):
    """Run extension with proper logging"""
    print(f"[Extension] Starting execution")  # Simple logging
    
    try:
        # Extension logic here
        results = generate_data()
        
        # Save results
        save_results(results, output_dir)
        
        print(f"[Extension] Execution completed successfully")  # Simple logging
        
    except Exception as e:
        print(f"[Extension] ERROR: {e}")  # Simple logging
        raise
```

### **Logging in Extensions**
```python
class HeuristicGameManager:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.component_name = f"HeuristicGameManager_{algorithm}"
        print(f"[{self.component_name}] Initialized")  # Simple logging
    
    def run_games(self, num_games: int):
        print(f"[{self.component_name}] Running {num_games} games")  # Simple logging
        
        results = []
        for game_id in range(num_games):
            print(f"[{self.component_name}] Starting game {game_id + 1}")  # Simple logging
            
            try:
                game_result = self.run_single_game()
                results.append(game_result)
                
                print(f"[{self.component_name}] Game {game_id + 1} completed, score: {game_result['score']}")  # Simple logging
                
            except Exception as e:
                print(f"[{self.component_name}] ERROR in game {game_id + 1}: {e}")  # Simple logging
                continue
        
        print(f"[{self.component_name}] All games completed, {len(results)} successful")  # Simple logging
        return results
```

## 📋 **Implementation Checklist**

### **Path Management**
- [ ] **Project Root**: Proper project root detection
- [ ] **Working Directory**: Consistent working directory setting
- [ ] **Path Utilities**: Use of common path utilities
- [ ] **Directory Creation**: Proper directory creation and management

### **Logging Standards**
- [ ] **Simple Logging**: Use of print statements only (SUPREME_RULES compliance)
- [ ] **Component Names**: Clear component identification
- [ ] **Log Levels**: Appropriate log level usage
- [ ] **Error Handling**: Proper error logging

### **File Organization**
- [ ] **Directory Structure**: Follow standard directory structure
- [ ] **File Naming**: Consistent file naming conventions
- [ ] **Timestamp Usage**: Proper timestamp usage for unique directories
- [ ] **Metadata**: Proper metadata file creation

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Path Management**: Understanding consistent path handling
- **Logging Standards**: Learning simple, effective logging
- **File Organization**: Understanding file organization patterns
- **Debugging**: Using consistent paths and logging for debugging

### **Best Practices**
- **Consistency**: Consistent path and logging patterns
- **Simplicity**: Simple, effective logging without over-engineering
- **Organization**: Clear file organization and naming
- **Maintainability**: Easy to maintain and debug

---

**Working directory and logging standards ensure consistent, predictable behavior across all Snake Game AI extensions, providing clear paths and simple logging for educational value and technical excellence.**

## 🔗 **See Also**

- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization 