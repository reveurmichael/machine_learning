# Working Directory and Path Management

> **📢 NOTE — Authoritative Reference**: Final Decision 6 ( `final-decision-6.md` ) is now the **single source of truth** for all path-management code patterns and utilities.  
> This document focuses **only** on directory-structure conventions and log locations.

## 🎯 **Why This Document Still Exists**
`cwd-and-logs.md` captures *where* generated artifacts live (datasets, models, execution logs) and *why* the grid-size hierarchy matters.  The *how* (code needed to set paths) has been centralized in Final Decision 6 to avoid duplication.

If you need to know **how to call** `ensure_project_root()` or any related helper, **stop reading now and open `final-decision-6.md`**.

---

## 🎯 **Mandatory Path Management Philosophy**

All extensions **MUST** use standardized path management from `extensions/common/path_utils.py` as established in Final Decision 6. This eliminates path-related bugs and ensures reliable cross-platform operation.

## 📁 **Path Management Requirements**

### **Core Requirement: Use Common Path Utilities**

All extensions are **required** to use the standardized path management utilities instead of manual path construction or working directory manipulation:

```python
# MANDATORY for all extensions
from extensions.common.path_utils import (
    ensure_project_root,
    get_extension_path,
    get_dataset_path,
    get_model_path,
    validate_path_structure
)

# Standard setup pattern for all extensions
def setup_extension_environment():
    """Required setup for all extensions"""
    project_root = ensure_project_root()  # Ensures working directory = repo root
    extension_path = get_extension_path(__file__)
    validate_path_structure(project_root, extension_path)
    return project_root, extension_path
```

### **Benefits of Standardized Path Management**
- **Consistent Behavior**: Relative paths work identically across all platforms
- **Cross-Platform Compatibility**: Works reliably on Windows, macOS, and Linux  
- **Subprocess Safety**: Child processes inherit correct working directory
- **IDE Independence**: Works regardless of IDE working directory settings
- **Error Prevention**: Eliminates path-related user errors

## 🗂️ **Directory Structure Overview**

### **Task-0 Logs (Primary)**
```
logs/
├── {model}_{timestamp}/          # LLM session logs
│   ├── game_N.json
│   ├── summary.json
│   ├── prompts/
│   └── responses/
└── ...
```

### **Extension Logs (Organized)**
Following Final Decision 1 structure:
```
logs/extensions/
├── datasets/                     # Grid-size organized datasets
│   └── grid-size-N/
│       ├── heuristics_v0.03_{timestamp}/
│       ├── supervised_v0.02_{timestamp}/
│       └── reinforcement_v0.02_{timestamp}/
├── models/                       # Grid-size organized models
│   └── grid-size-N/
│       ├── supervised_v0.02_{timestamp}/
│       └── reinforcement_v0.02_{timestamp}/
└── [execution logs by extension]
```

## 🚀 **Implementation Benefits**

### **For Development**
- **Predictable Paths**: All relative paths resolve consistently across environments
- **Easy Debugging**: Clear, standardized working directory management
- **Reliable Testing**: Consistent behavior across different test environments

### **For Deployment**
- **Container Compatibility**: Works seamlessly in Docker and other containers
- **CI/CD Reliability**: Consistent behavior in automation pipelines
- **Multi-Environment**: Same code works identically in development and production

### **For Users**
- **No Setup Required**: Works regardless of launch directory or platform
- **Consistent Experience**: Same behavior across different operating systems
- **Error Prevention**: Reduces common path-related configuration errors

## 🔧 **Path Utility Integration**

### **Required Pattern for All Extensions**
```python
# All extensions must start with this pattern
project_root, extension_path = setup_extension_environment()

# Use utilities for all path operations
dataset_path = get_dataset_path(
    extension_type="heuristics",
    version="0.03", 
    grid_size=grid_size,
    algorithm="bfs",
    timestamp=timestamp
)
```

### **Path Validation**
The path utilities automatically:
- Change working directory to repository root
- Add repository root to Python path
- Validate directory structure compliance
- Ensure cross-platform path compatibility

---

**This standardized path management ensures reliable, consistent, and maintainable directory operations across all extensions while eliminating platform-specific path issues.** 