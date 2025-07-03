# Final Decision 6: Path Management Standardization

> **SUPREME AUTHORITY**: This document establishes the definitive path management standards for all Snake Game AI extensions.

> **See also:** `unified-path-management-guide.md` (Path standards), `cwd-and-logs.md` (Working directory), `standalone.md` (Extension independence), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Executive Summary**

This document establishes **mandatory path management standards** for all Snake Game AI extensions, requiring consistent use of `extensions/common/utils/path_utils.py` utilities. This ensures reliable cross-platform operation, consistent working directory management, and eliminates path-related bugs across all extension types, strictly following `final-decision-10.md` SUPREME_RULES.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making path management changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining path patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`unified-path-management-guide.md`**: Authoritative reference for path management standards
- **`cwd-and-logs.md`**: Authoritative reference for working directory and log organization
- **`single-source-of-truth.md`**: Ensures path consistency across all extensions
- **`standalone.md`**: Maintains extension independence through proper path management

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using ROOT/utils/print_utils.py functions rather than complex logging mechanisms:

```python
from utils.print_utils import print_info, print_warning, print_error, print_success

# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def ensure_project_root() -> Path:
    """Ensure current working directory is project root"""
    current_file = Path(__file__).resolve()
    project_root = find_project_root(current_file)
    
    if os.getcwd() != str(project_root):
        os.chdir(str(project_root))
        print_info(f"[PathUtils] Changed working directory to: {project_root}")  # SUPREME_RULE NO.3
    
    return project_root

def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str) -> Path:
    """Get standardized dataset path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs/extensions/datasets") / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}" / algorithm
    print_info(f"[PathUtils] Generated dataset path: {path}")  # SUPREME_RULE NO.3
    return path
```

## 🛠️ **Mandatory Path Management Pattern**

### **Core Requirement: Use Common Path Utilities**

All extensions **MUST** use the standardized path management utilities from `extensions/common/utils/path_utils.py` instead of manual path construction or working directory manipulation.

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
    """
    Standard setup for all extensions
    
    Design Patterns:
    - Template Method Pattern: Consistent setup across all extensions
    - Facade Pattern: Simplified path management interface
    - Strategy Pattern: Different path strategies for different contexts
    """
    # Ensure we're working from project root
    project_root = ensure_project_root()
    
    # Get extension-specific paths
    extension_path = get_extension_path(__file__)
    
    # Validate path structure
    validate_path_structure(extension_path)
    
    return project_root, extension_path
```

## 📁 **Required Path Utilities**

### **1. Project Root Management**

```python
# extensions/common/utils/path_utils.py
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def ensure_project_root() -> Path:
    """
    Ensure current working directory is project root
    
    Design Pattern: Facade Pattern
    - Provides simple interface to complex path management
    - Handles cross-platform compatibility
    - Manages Python path and working directory
    
    Returns:
        Path: Project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
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
        print_info(f"[PathUtils] Changed working directory to: {project_root}")  # Simple logging
    
    # Ensure project root is in Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

def get_extension_path(current_file: str) -> Path:
    """
    Get the extension directory path from current file
    
    Args:
        current_file: __file__ from the calling module
        
    Returns:
        Path: Extension directory path
    """
    extension_path = Path(current_file).resolve().parent
    print_info(f"[PathUtils] Extension path: {extension_path}")  # Simple logging
    return extension_path
```

### **2. Dataset and Model Path Management**

```python
def get_dataset_path(extension_type: str, version: str, grid_size: int, 
                    algorithm: str, timestamp: str = None) -> Path:
    """
    Get standardized dataset path following final-decision-1.md structure
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        version: Extension version (0.01, 0.02, 0.03, etc.)
        grid_size: Grid size used for training/generation
        algorithm: Algorithm name (bfs, astar, mlp, etc.)
        timestamp: Session timestamp (auto-generated if None)
        
    Returns:
        Path: Dataset directory path
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_name = f"{extension_type}_v{version}_{timestamp}"
    path = Path("logs/extensions/datasets") / f"grid-size-{grid_size}" / session_name / algorithm
    print_info(f"[PathUtils] Generated dataset path: {path}")  # Simple logging
    return path

def get_model_path(extension_type: str, version: str, grid_size: int,
                  model_name: str, timestamp: str = None) -> Path:
    """
    Get standardized model path following final-decision-1.md structure
    
    Args:
        extension_type: Type of extension (supervised, reinforcement, etc.)
        version: Extension version
        grid_size: Grid size used for training
        model_name: Model name (mlp, dqn, lora, etc.)
        timestamp: Session timestamp (auto-generated if None)
        
    Returns:
        Path: Model directory path
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_name = f"{extension_type}_v{version}_{timestamp}"
    path = Path("logs/extensions/models") / f"grid-size-{grid_size}" / session_name / model_name
    print_info(f"[PathUtils] Generated model path: {path}")  # Simple logging
    return path

def validate_path_structure(project_root: Path, extension_path: Path) -> None:
    """
    Validate that path structure follows required patterns
    
    Args:
        project_root: Project root directory
        extension_path: Extension directory
        
    Raises:
        ValueError: If path structure is invalid
    """
    print_info(f"[PathUtils] Validating path structure")  # Simple logging
    
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
    
    print_success(f"[PathUtils] Path structure validation passed")  # Simple logging
```

## 🔧 **Extension Implementation Patterns**

### **v0.01 Extension Pattern**

```python
# extensions/heuristics-v0.01/main.py
import sys
from pathlib import Path

# MANDATORY: Use common path utilities
from extensions.common.utils.path_utils import ensure_project_root, get_extension_path

def main():
    """Main entry point for heuristics v0.01"""
    # Standard setup
    project_root, extension_path = setup_extension_environment()
    
    # Extension-specific logic
    from game_manager import HeuristicGameManager
    from game_logic import HeuristicGameLogic
    
    # Initialize and run
    manager = HeuristicGameManager()
    manager.run()

def setup_extension_environment():
    """Standard extension environment setup"""
    print_info(f"[HeuristicsV001] Setting up extension environment")  # Simple logging
    
    # Ensure project root
    project_root = ensure_project_root()
    
    # Get extension path
    extension_path = get_extension_path(__file__)
    
    # Validate structure
    validate_path_structure(extension_path)
    
    print_success(f"[HeuristicsV001] Environment setup complete")  # Simple logging
    return project_root, extension_path
```

### **v0.02 Extension Pattern**

```python
# extensions/heuristics-v0.02/main.py
import argparse
from extensions.common.utils.path_utils import (
    ensure_project_root, 
    get_extension_path, 
    get_dataset_path,
    validate_path_structure
)

def main():
    """Main entry point for heuristics v0.02"""
    parser = argparse.ArgumentParser(description="Heuristics Snake Game AI v0.02")
    parser.add_argument("--algorithm", default="bfs", help="Algorithm to use")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    args = parser.parse_args()
    
    # Standard setup
    project_root, extension_path = setup_extension_environment()
    
    # Get dataset path for this run
    dataset_path = get_dataset_path(
        extension_type="heuristics",
        version="0.02",
        grid_size=args.grid_size,
        algorithm=args.algorithm
    )
    
    # Extension-specific logic
    from game_manager import HeuristicGameManager
    from agents import AgentFactory
    
    # Create agent using factory pattern
    agent = AgentFactory.create(args.algorithm, grid_size=args.grid_size)
    
    # Initialize and run
    manager = HeuristicGameManager(agent=agent, dataset_path=dataset_path)
    manager.run()

def setup_extension_environment():
    """Standard extension environment setup"""
    print_info(f"[HeuristicsV002] Setting up extension environment")  # Simple logging
    
    # Ensure project root
    project_root = ensure_project_root()
    
    # Get extension path
    extension_path = get_extension_path(__file__)
    
    # Validate structure
    validate_path_structure(extension_path)
    
    print_success(f"[HeuristicsV002] Environment setup complete")  # Simple logging
    return project_root, extension_path
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Path Management**: Understanding cross-platform path handling
- **Working Directory**: Learning to manage working directories properly
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Extension Independence**: Learning to create standalone extensions

### **Implementation Examples**
- **Extension Setup**: How to set up extensions with proper path management
- **Path Generation**: How to generate consistent paths across extensions
- **Validation**: How to validate path structures
- **Factory Integration**: How to integrate path management with factory patterns

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`unified-path-management-guide.md`**: Detailed path management standards
- **`cwd-and-logs.md`**: Working directory and log organization
- **`standalone.md`**: Extension independence principles
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This path management system supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All extension versions (v0.01, v0.02, v0.03)
- Cross-platform compatibility
- Consistent path organization across all extensions

---

**This path management standardization ensures reliable, consistent, and cross-platform path handling across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project. 