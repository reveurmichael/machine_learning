# Unified Path Management Guide for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines unified path management patterns.

> **See also:** `final-decision-10.md`, `core.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Consistent Path Resolution**

All extensions **MUST** use standardized path utilities from `extensions/common/utils/path_utils.py` to ensure reliable cross-platform operation and eliminate path-related bugs.

### **Educational Value**
- **Cross-Platform Compatibility**: Consistent behavior across Windows, macOS, and Linux
- **Development Workflow**: Works regardless of IDE working directory
- **Deployment Reliability**: Same code works in dev and production
- **Canonical Patterns**: Demonstrates factory patterns and simple logging throughout

## 🛠️ **Mandatory Path Management Pattern**

### **Required Setup for All Extensions**
```python
# MANDATORY USAGE PATTERN FOR ALL EXTENSIONS
from extensions.common.utils.path_utils import (
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
    Purpose: Provides simple interface to complex path management
    Educational Value: Shows how canonical patterns work with path management
    while maintaining simple logging throughout.
    
    Reference: final-decision-10.md for simple logging standards
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
        print(f"[PathUtils] Changed working directory to: {project_root}")  # Simple logging - SUPREME_RULES
    
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
    
    print(f"[PathUtils] Path structure validated successfully")  # Simple logging - SUPREME_RULES
```

## 🔧 **Extension Implementation Patterns**

### **v0.01 Extension Pattern**
```python
# extensions/heuristics-v0.01/main.py
from extensions.common.utils.path_utils import ensure_project_root, get_extension_path

def main():
    """Main entry point for heuristics v0.01"""
    # Standard setup
    project_root, extension_path = setup_extension_environment()
    
    # Extension-specific logic using canonical factory patterns
    from extensions.common.utils.factory_utils import SimpleFactory
    from game_manager import HeuristicGameManager
    
    # Use canonical factory pattern
    factory = SimpleFactory()
    factory.register("heuristic", HeuristicGameManager)
    
    manager = factory.create("heuristic")  # CANONICAL create() method - SUPREME_RULES
    manager.run()
```

### **v0.02 Extension Pattern**
```python
# extensions/heuristics-v0.02/main.py
from extensions.common.utils.path_utils import ensure_project_root, get_extension_path, get_dataset_path

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
    
    # Extension logic using canonical factory patterns
    from extensions.common.utils.factory_utils import SimpleFactory
    from game_manager import HeuristicGameManager
    
    factory = SimpleFactory()
    factory.register("heuristic", HeuristicGameManager)
    
    manager = factory.create("heuristic", dataset_path=dataset_path)  # Canonical
    manager.run()
```

## 📊 **Simple Logging Standards for Path Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All path operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for path operations (SUPREME_RULES compliance)
def setup_paths(extension_type: str):
    print(f"[PathUtils] Setting up paths for {extension_type}")  # Simple logging - REQUIRED
    
    # Path setup logic
    project_root = ensure_project_root()
    
    print(f"[PathUtils] Paths configured successfully")  # Simple logging
    return project_root

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def setup_paths(extension_type: str):
#     logger.info(f"Setting up paths for {extension_type}")  # FORBIDDEN - complex logging
#     # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Path Management Benefits**
- **Cross-Platform Compatibility**: Consistent behavior across all operating systems
- **Development Workflow**: Works regardless of IDE working directory
- **Deployment Reliability**: Same code works in dev and production
- **Canonical Patterns**: Factory patterns ensure consistent path management

### **Pattern Consistency**
- **Canonical Method**: All path utilities use consistent patterns
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: Path management follows same standards as other components

## 📋 **SUPREME_RULES Implementation Checklist for Path Management**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All path utilities use consistent patterns (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all path operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all path management documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all path implementations

### **Path-Specific Standards**
- [ ] **Cross-Platform**: Works on Windows, macOS, and Linux
- [ ] **Working Directory**: Proper working directory management
- [ ] **Path Validation**: Consistent validation patterns
- [ ] **Error Handling**: Simple logging for all error conditions

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical patterns
- [ ] **Pattern Documentation**: Clear explanation of path management benefits
- [ ] **SUPREME_RULES Compliance**: All examples follow final-decision-10.md standards
- [ ] **Cross-Reference**: Links to related patterns and principles

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for canonical path patterns
- **`core.md`**: Core architecture and path integration
- **`project-structure-plan.md`**: Project structure standards

### **Implementation Files**
- **`extensions/common/utils/path_utils.py`**: Canonical path utilities
- **`extensions/common/utils/factory_utils.py`**: Canonical factory utilities
- **`extensions/common/utils/csv_schema_utils.py`**: Schema utilities with factory patterns

### **Educational Resources**
- **Design Patterns**: Path management as foundation for cross-platform compatibility
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **OOP Principles**: Path management demonstrates effective abstraction 