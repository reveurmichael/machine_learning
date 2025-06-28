# Unified Path Management Guide for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines unified path management patterns.

> **See also:** `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

# Unified Path Management Guide

> **Authoritative Reference**: This document serves as a **GOOD_RULES** authoritative reference for path management standards and establishes the definitive path management standards for all Snake Game AI extensions, following final-decision-6.md.

## 🎯 **Core Philosophy: Consistent Path Resolution**

All extensions **MUST** use standardized path utilities from `extensions/common/path_utils.py` to ensure reliable cross-platform operation and eliminate path-related bugs.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Enforces reading all GOOD_RULES before making path management architectural changes to ensure comprehensive understanding
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and path management patterns
- **simple logging**: Enables lightweight common utilities with OOP extensibility while maintaining path management patterns through inheritance rather than tight coupling

## 🛠️ **Mandatory Path Management Pattern**

### **Required Setup for All Extensions**
```python
# MANDATORY USAGE PATTERN FOR ALL EXTENSIONS
from extensions.common.path_utils import (
    ensure_project_root,
    get_extension_path,
    get_dataset_path,
    get_model_path,
    validate_path_structure
)

def setup_extension_environment():
    """Standard setup for all extensions"""
    # Ensure we're working from project root
    project_root = ensure_project_root()
    
    # Get extension-specific paths
    extension_path = get_extension_path(__file__)
    
    # Validate path structure
    validate_path_structure(project_root, extension_path)
    
    return project_root, extension_path
```

## 📁 **Core Path Utilities**

### **Project Root Management**
```python
def ensure_project_root() -> Path:
    """
    Ensure current working directory is project root
    
    Design Pattern: Facade Pattern
    - Provides simple interface to complex path management
    - Handles cross-platform compatibility
    - Manages Python path and working directory
    """
    current_file = Path(__file__).resolve()
    
    # Find project root (contains README.md and core/ folder)
    project_root = current_file
    while project_root.parent != project_root:
        if (project_root / "README.md").exists() and (project_root / "core").exists():
            break
        project_root = project_root.parent
    else:
        raise RuntimeError("Could not find project root directory")
    
    # Change working directory to project root
    if os.getcwd() != str(project_root):
        os.chdir(str(project_root))
        print(f"Changed working directory to: {project_root}")
    
    # Ensure project root is in Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root
```

### **Extension Path Management**
```python
def get_extension_path(current_file: str) -> Path:
    """Get the extension directory path from current file"""
    return Path(current_file).resolve().parent

def get_dataset_path(extension_type: str, version: str, grid_size: int, 
                    algorithm: str, timestamp: str) -> Path:
    """Get standardized dataset path following final-decision-1.md structure"""
    session_name = f"{extension_type}_v{version}_{timestamp}"
    return Path("logs/extensions/datasets") / f"grid-size-{grid_size}" / session_name / algorithm

def get_model_path(extension_type: str, version: str, grid_size: int,
                  model_name: str, timestamp: str) -> Path:
    """Get standardized model path following final-decision-1.md structure"""
    session_name = f"{extension_type}_v{version}_{timestamp}"
    return Path("logs/extensions/models") / f"grid-size-{grid_size}" / session_name / model_name
```

### **Path Validation**
```python
def validate_path_structure(project_root: Path, extension_path: Path) -> None:
    """Validate that path structure follows required patterns"""
    # Validate project root
    if not (project_root / "README.md").exists():
        raise ValueError(f"Invalid project root: {project_root}")
    
    if not (project_root / "core").exists():
        raise ValueError(f"Missing core/ directory in project root: {project_root}")
    
    # Validate extension path
    if not extension_path.exists():
        raise ValueError(f"Extension path does not exist: {extension_path}")
    
    # Validate extension structure
    required_files = ["__init__.py", "game_logic.py", "game_manager.py"]
    for file in required_files:
        if not (extension_path / file).exists():
            raise ValueError(f"Missing required file {file} in extension: {extension_path}")
```

## 🔧 **Extension Implementation Patterns**

### **v0.01 Extension Pattern**
```python
# extensions/heuristics-v0.01/main.py
from extensions.common.path_utils import ensure_project_root, get_extension_path

def main():
    """Main entry point for heuristics v0.01"""
    # Standard setup
    project_root, extension_path = setup_extension_environment()
    
    # Extension-specific logic
    from game_manager import HeuristicGameManager
    manager = HeuristicGameManager()
    manager.run()
```

### **v0.02 Extension Pattern**
```python
# extensions/heuristics-v0.02/main.py
from extensions.common.path_utils import ensure_project_root, get_extension_path, get_dataset_path

def main():
    """Main entry point for heuristics v0.02"""
    project_root, extension_path = setup_extension_environment()
    
    # Use standardized dataset paths
    dataset_path = get_dataset_path(
        extension_type="heuristics",
        version="0.02",
        grid_size=args.grid_size,
        algorithm=args.algorithm,
        timestamp=timestamp
    )
    
    # Extension logic
    from game_manager import HeuristicGameManager
    manager = HeuristicGameManager(dataset_path=dataset_path)
    manager.run()
```

### **v0.03 Extension Pattern**
```python
# extensions/heuristics-v0.03/app.py
from extensions.common.path_utils import ensure_project_root, get_extension_path

class HeuristicStreamlitApp(BaseExtensionApp):
    def __init__(self):
        # Standard setup
        self.project_root, self.extension_path = setup_extension_environment()
        super().__init__()
    
    def launch_script(self, script_name: str, params: dict):
        """Launch script with proper path management"""
        script_path = self.extension_path / "scripts" / f"{script_name}.py"
        
        # Use subprocess with proper working directory
        subprocess.run([
            "python", str(script_path),
            *[f"--{k}", str(v) for k, v in params.items()]
        ], cwd=self.project_root)
```

## 🎯 **Benefits of Standardized Path Management**

### **Cross-Platform Compatibility**
- **Windows**: Handles backslashes and drive letters
- **macOS/Linux**: Handles forward slashes and permissions
- **Docker**: Works in containerized environments

### **Development Workflow**
- **IDE Independence**: Works regardless of IDE working directory
- **Subprocess Safety**: Child processes inherit correct working directory
- **Error Prevention**: Eliminates common path-related user errors

### **Deployment Reliability**
- **Container Compatibility**: Works seamlessly in Docker
- **CI/CD Reliability**: Consistent behavior in automation
- **Multi-Environment**: Same code works in dev and production

## 🚫 **Anti-Patterns to Avoid**

### **Manual Path Construction**
```python
# ❌ WRONG: Manual path construction
import os
dataset_path = os.path.join("logs", "extensions", "datasets", f"grid-size-{grid_size}")

# ✅ CORRECT: Use standardized utilities
from extensions.common.path_utils import get_dataset_path
dataset_path = get_dataset_path(extension_type, version, grid_size, algorithm, timestamp)
```

### **Working Directory Assumptions**
```python
# ❌ WRONG: Assuming working directory
with open("config.json", "r") as f:  # May fail if run from wrong directory

# ✅ CORRECT: Use project root
project_root = ensure_project_root()
config_path = project_root / "config.json"
with open(config_path, "r") as f:
```

### **Platform-Specific Paths**
```python
# ❌ WRONG: Platform-specific paths
if os.name == "nt":  # Windows
    path = "logs\\extensions\\datasets"
else:  # Unix
    path = "logs/extensions/datasets"

# ✅ CORRECT: Use pathlib
from pathlib import Path
path = Path("logs/extensions/datasets")
```

## 🔍 **Validation and Testing**

### **Path Validation Script**
```python
# extensions/common/validation/path_validator.py
def validate_extension_paths(extension_path: Path):
    """Validate extension follows path management standards"""
    
    # Check required files exist
    required_files = ["__init__.py", "game_logic.py", "game_manager.py"]
    for file in required_files:
        if not (extension_path / file).exists():
            raise ValidationError(f"Missing required file: {file}")
    
    # Check path utilities are used
    main_file = extension_path / "main.py"
    if main_file.exists():
        with open(main_file) as f:
            content = f.read()
            if "ensure_project_root" not in content:
                raise ValidationError("main.py must use ensure_project_root()")
```

### **Testing Path Management**
```python
def test_path_management():
    """Test path management utilities"""
    
    # Test project root detection
    project_root = ensure_project_root()
    assert (project_root / "README.md").exists()
    assert (project_root / "core").exists()
    
    # Test dataset path generation
    dataset_path = get_dataset_path("heuristics", "0.02", 10, "bfs", "20240101_120000")
    assert "logs/extensions/datasets/grid-size-10/heuristics_v0.02_20240101_120000/bfs" in str(dataset_path)
```

---

**This unified path management ensures reliable, consistent, and maintainable directory operations across all extensions while eliminating platform-specific path issues.** 