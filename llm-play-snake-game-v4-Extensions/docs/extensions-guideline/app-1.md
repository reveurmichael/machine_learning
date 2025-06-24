# Streamlit App Path Management

> **Important — Authoritative Reference:** This document supplements Final Decision 6 on mandatory path management standards.

## 🎯 **Path Management for Streamlit Apps**

All Streamlit applications in extensions **MUST** use the standardized path utilities from `extensions/common/path_utils.py` as established in Final Decision 6.

## 🚫 **Common Path Issues**

### **Problem: Module Import Failures**
```
ModuleNotFoundError: No module named 'extensions'
```

**Root Cause**: Streamlit changes working directory and Python path when launching applications.

## ✅ **Mandatory Solution Pattern**

### **Standard Path Setup**
```python
# REQUIRED at the top of all extension app.py files
import sys
import os
from pathlib import Path

# Use common path utilities (Final Decision 6)
from extensions.common.path_utils import ensure_project_root, get_extension_path

def setup_streamlit_environment():
    """
    Standard setup for all Streamlit apps in extensions
    
    This function ensures:
    - Working directory is project root
    - Python path includes project root
    - Extension paths are properly resolved
    """
    project_root = ensure_project_root()
    extension_path = get_extension_path(__file__)
    return project_root, extension_path

# Call setup before any other imports
project_root, extension_path = setup_streamlit_environment()

# Now safe to import project modules
import streamlit as st
from extensions.common.app_utils import BaseExtensionApp
```

### **Benefits of Standardized Setup**

#### **Reliability**
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Environment Independent**: Same behavior in IDE, terminal, and containers
- **Streamlit Compatible**: Handles Streamlit's working directory changes

#### **Maintainability**
- **Consistent Pattern**: Same setup across all extension apps
- **Single Source**: Changes to path logic centralized in one location
- **Error Prevention**: Eliminates path-related import failures

#### **Development Experience**
- **Predictable Behavior**: Apps launch reliably from any directory
- **Easy Debugging**: Clear working directory and path resolution
- **IDE Integration**: Works with different IDE configurations

## 🔧 **Implementation Guidelines**

### **App Entry Point**
```python
# extensions/{algorithm}-v0.03/app.py

# MANDATORY: Path setup before imports
from extensions.common.path_utils import ensure_project_root
ensure_project_root()

import streamlit as st
from extensions.common.app_utils import BaseExtensionApp

class AlgorithmApp(BaseExtensionApp):
    def __init__(self):
        super().__init__()
        self.setup_page()
        self.main()
    
    def setup_page(self):
        st.set_page_config(page_title=f"{self.get_extension_name()} Dashboard")
    
    def main(self):
        # App logic here
        pass

if __name__ == "__main__":
    AlgorithmApp()
```

### **Script Launching**
```python
# Use proper path resolution for subprocess calls
import subprocess
from extensions.common.path_utils import get_extension_path

extension_path = get_extension_path(__file__)
script_path = extension_path / "scripts" / "main.py"

subprocess.run([
    "python", str(script_path),
    "--algorithm", algorithm_name,
    # ... other parameters
])
```

## 📋 **Compliance Checklist**

- [ ] Does the app use `ensure_project_root()` before other imports?
- [ ] Are all path operations using utilities from `extensions/common/path_utils.py`?
- [ ] Does the app work when launched from any directory?
- [ ] Are subprocess calls using properly resolved script paths?

---

**This standardized path management ensures reliable Streamlit app operation across all extensions and environments.**
