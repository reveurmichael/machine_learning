# Unified Path Management Guide

> **Authoritative Reference**: This document provides the **single canonical path management implementation** for all extensions. It replaces all scattered path management patterns in other guideline files.

## 🎯 **Core Path Management Philosophy**

Consistent, reliable path management is critical for:
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Consistent behavior** regardless of launch directory
- **Subprocess safety** for script launching
- **IDE independence** from working directory settings
- **Container compatibility** for deployment

## 🛠️ **Required Path Management Implementation**

All extensions **MUST** use this standardized implementation:

### **Core Path Utilities (extensions/common/path_utils.py)**
```python
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

def ensure_project_root() -> Path:
    """
    Ensure current working directory is project root and add to Python path.
    
    This MUST be called before any other imports in all extension entry points.
    
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Find project root by looking for key marker files
    current = Path.cwd()
    
    # Try current directory and parents
    for path in [current] + list(current.parents):
        if all((path / marker).exists() for marker in ['core', 'extensions', 'config']):
            project_root = path
            break
    else:
        # Fallback: look for __file__ based detection
        script_path = Path(__file__).resolve()
        for path in [script_path.parent] + list(script_path.parents):
            if all((path / marker).exists() for marker in ['core', 'extensions', 'config']):
                project_root = path
                break
        else:
            raise RuntimeError(
                "Cannot determine project root. Ensure you're running from within the project directory."
            )
    
    # Change working directory to project root
    os.chdir(project_root)
    
    # Add project root to Python path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

def get_extension_path(file_path: str) -> Path:
    """
    Get the extension directory path from a file within the extension.
    
    Args:
        file_path: Usually __file__ from the calling module
        
    Returns:
        Path to the extension directory
    """
    return Path(file_path).resolve().parent

def get_dataset_path(
    extension_type: str,
    version: str,
    grid_size: int,
    algorithm: str,
    timestamp: str
) -> Path:
    """
    Generate standardized dataset path following grid-size hierarchy.
    
    Returns:
        Standardized dataset path: logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/
    """
    project_root = Path.cwd()  # Should be project root after ensure_project_root()
    
    return project_root / "logs" / "extensions" / "datasets" / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}" / algorithm

def get_model_path(
    extension_type: str,
    version: str,
    grid_size: int,
    algorithm: str,
    timestamp: str
) -> Path:
    """
    Generate standardized model path following grid-size hierarchy.
    
    Args:
        extension_type: Type of extension (supervised, reinforcement)
        version: Extension version (0.02, 0.03, etc.)
        grid_size: Grid size for the model
        algorithm: Algorithm name
        timestamp: Timestamp in YYYYMMDD_HHMMSS format
        
    Returns:
        Standardized model path
    """
    project_root = Path.cwd()  # Should be project root after ensure_project_root()
    
    return project_root / "logs" / "extensions" / "models" / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}" / algorithm

def validate_path_structure(project_root: Path, extension_path: Path) -> None:
    """
    Validate that paths follow expected structure.
    
    Args:
        project_root: Project root directory
        extension_path: Extension directory path
        
    Raises:
        ValueError: If path structure is invalid
    """
    # Validate project root has expected directories
    required_dirs = ['core', 'extensions', 'config', 'logs']
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            raise ValueError(f"Invalid project root: missing {dir_name} directory")
    
    # Validate extension path is within extensions directory
    extensions_dir = project_root / "extensions"
    try:
        extension_path.relative_to(extensions_dir)
    except ValueError:
        raise ValueError(f"Extension path {extension_path} is not within extensions directory")

def create_timestamp() -> str:
    """
    Create standardized timestamp for path generation.
    
    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
```

## 📋 **Mandatory Usage Patterns**

### **Extension Entry Points (ALL files)**
```python
# extensions/{algorithm}-v0.0N/app.py
# extensions/{algorithm}-v0.0N/scripts/main.py
# extensions/{algorithm}-v0.0N/scripts/generate_dataset.py

# MANDATORY: First lines in every extension entry point
from extensions.common.path_utils import ensure_project_root
ensure_project_root()  # MUST be called before any other imports

# Now safe to import project modules
import streamlit as st  # or other project imports
from config.game_constants import VALID_MOVES
from core.game_manager import BaseGameManager
```

### **Dataset Generation Pattern**
```python
# extensions/{algorithm}-v0.0N/scripts/generate_dataset.py

from extensions.common.path_utils import (
    ensure_project_root,
    get_dataset_path,
    create_timestamp
)

def generate_dataset(algorithm: str, grid_size: int = 10):
    # MANDATORY: Ensure proper working directory
    ensure_project_root()
    
    # Generate standardized path
    timestamp = create_timestamp()
    dataset_path = get_dataset_path(
        extension_type="heuristics",  # or supervised, reinforcement
        version="0.03",
        grid_size=grid_size,
        algorithm=algorithm,
        timestamp=timestamp
    )
    
    # Create directories if needed
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset with proper path
    output_file = dataset_path / "processed_data" / "tabular_data.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset.to_csv(output_file, index=False)
    
    return dataset_path
```

### **Streamlit App Pattern**
```python
# extensions/{algorithm}-v0.03/app.py

# MANDATORY: Path setup first
from extensions.common.path_utils import ensure_project_root
ensure_project_root()

import streamlit as st
import subprocess
from pathlib import Path

def launch_script(script_name: str, **params):
    """Launch script with proper path management"""
    script_path = Path("extensions") / "heuristics-v0.03" / "scripts" / script_name
    
    cmd = [sys.executable, str(script_path)]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    subprocess.run(cmd)
```

### **Script Launching Pattern**
```python
# dashboard/tab_main.py or similar

def launch_algorithm_script(algorithm: str, grid_size: int):
    """Launch algorithm script with proper path management"""
    
    # Working directory is already project root (set by ensure_project_root)
    # Scripts use relative paths from project root
    script_path = Path("extensions") / "heuristics-v0.03" / "scripts" / "main.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--algorithm", algorithm,
        "--grid-size", str(grid_size)
    ]
    
    subprocess.run(cmd)
```

## 🔧 **Extension Setup Helper**

```python
# extensions/common/setup_utils.py

from .path_utils import ensure_project_root, get_extension_path, validate_path_structure

def setup_extension_environment(file_path: str) -> Tuple[Path, Path]:
    """
    Standard setup for all extensions.
    
    Args:
        file_path: Usually __file__ from the calling module
        
    Returns:
        Tuple of (project_root, extension_path)
    """
    # Ensure proper working directory and Python path
    project_root = ensure_project_root()
    
    # Get extension directory
    extension_path = get_extension_path(file_path)
    
    # Validate structure
    validate_path_structure(project_root, extension_path)
    
    return project_root, extension_path

# Convenience function for common pattern
def setup_extension(file_path: str) -> Tuple[Path, Path]:
    """Alias for setup_extension_environment for backward compatibility"""
    return setup_extension_environment(file_path)
```

## 📁 **Standard Directory Structure Enforcement**

```python
# extensions/common/validation/path_validator.py

def validate_dataset_path_format(path: str) -> bool:
    """Validate dataset path follows standardized format"""
    import re
    
    pattern = r"logs/extensions/datasets/grid-size-\d+/\w+_v\d+\.\d+_\d{8}_\d{6}/"
    return bool(re.match(pattern, path))

def validate_model_path_format(path: str) -> bool:
    """Validate model path follows standardized format"""
    import re
    
    pattern = r"logs/extensions/models/grid-size-\d+/\w+_v\d+\.\d+_\d{8}_\d{6}/"
    return bool(re.match(pattern, path))

def enforce_path_compliance(extension_type: str, version: str, grid_size: int, timestamp: str) -> dict:
    """Enforce and return all standardized paths for an extension"""
    
    paths = {
        'dataset_base': get_dataset_path(extension_type, version, grid_size, "base", timestamp).parent,
        'model_base': get_model_path(extension_type, version, grid_size, "base", timestamp).parent,
    }
    
    # Validate all paths
    for path_type, path in paths.items():
        if not validate_dataset_path_format(str(path)) and not validate_model_path_format(str(path)):
            raise ValueError(f"Invalid {path_type} path format: {path}")
    
    return paths
```

## 🚫 **Anti-Patterns to Avoid**

### **Don't Use Manual Path Construction**
```python
# ❌ WRONG: Manual path construction
dataset_path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension}_v{version}_{timestamp}/"

# ✅ CORRECT: Use standardized utilities
dataset_path = get_dataset_path(extension_type, version, grid_size, algorithm, timestamp)
```

### **Don't Assume Working Directory**
```python
# ❌ WRONG: Assuming current directory
with open("config/game_constants.py") as f:  # Fails if not in project root

# ✅ CORRECT: Ensure project root first
ensure_project_root()
with open("config/game_constants.py") as f:  # Always works
```

### **Don't Use Hardcoded Paths**
```python
# ❌ WRONG: Hardcoded absolute paths
script_path = "/home/user/project/extensions/heuristics-v0.03/scripts/main.py"

# ✅ CORRECT: Relative from project root
script_path = Path("extensions") / "heuristics-v0.03" / "scripts" / "main.py"
```

## 🧪 **Path Management Testing**

```python
# tests/test_path_management.py

import tempfile
import pytest
from pathlib import Path
from extensions.common.path_utils import (
    ensure_project_root,
    get_dataset_path,
    get_model_path,
    validate_path_structure
)

def test_ensure_project_root():
    """Test project root detection and setup"""
    original_cwd = Path.cwd()
    
    try:
        # Should work from project root
        project_root = ensure_project_root()
        assert (project_root / "core").exists()
        assert (project_root / "extensions").exists()
        
    finally:
        os.chdir(original_cwd)

def test_dataset_path_generation():
    """Test standardized dataset path generation"""
    path = get_dataset_path("heuristics", "0.03", 10, "bfs", "20241225_120000")
    
    expected = Path("logs/extensions/datasets/grid-size-10/heuristics_v0.03_20241225_120000/bfs")
    assert path == expected

def test_path_validation():
    """Test path structure validation"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Should fail without required directories
        with pytest.raises(ValueError):
            validate_path_structure(temp_path, temp_path / "extensions" / "test")
        
        # Should pass with proper structure
        for dir_name in ['core', 'extensions', 'config', 'logs']:
            (temp_path / dir_name).mkdir()
        
        validate_path_structure(temp_path, temp_path / "extensions" / "test")
```

## 📋 **Implementation Checklist**

For every extension, ensure:

- [ ] **ensure_project_root()** called first in all entry points
- [ ] **Standardized path generation** using provided utilities
- [ ] **No hardcoded paths** or manual path construction
- [ ] **Proper subprocess launching** from project root
- [ ] **Path validation** for generated datasets and models
- [ ] **Cross-platform compatibility** using Path objects
- [ ] **Error handling** for path-related operations

---

**This unified path management ensures reliable, consistent, and maintainable directory operations across all extensions while eliminating platform-specific path issues.** 