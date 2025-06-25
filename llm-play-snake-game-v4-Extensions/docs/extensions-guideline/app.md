# Streamlit App Path Management

> **Important — Authoritative Reference:** This document supplements Final Decision 6 (Path Management). **See `cwd-and-logs.md` for complete path management documentation.**

## 🎯 **Streamlit-Specific Path Requirements**

All Streamlit applications **MUST** use the standardized path utilities from Final Decision 6 to handle Streamlit's unique execution environment:

```python
# REQUIRED pattern for all extension app.py files
from extensions.common.path_utils import ensure_project_root

# MANDATORY: Call before any other imports
ensure_project_root()

# Now safe to import project modules
import streamlit as st
from extensions.common.app_utils import BaseExtensionApp
```

## 🧠 **Benefits of Standardized Path Management**

### **Reliability**
- **Cross-Platform**: Works consistently on Windows, macOS, and Linux
- **Environment Independent**: Same behavior in IDE, terminal, containers
- **Streamlit Compatible**: Handles Streamlit's working directory changes

### **Development Experience**
- **Predictable Behavior**: Apps launch reliably from any directory
- **Error Prevention**: Eliminates path-related import failures
- **Easy Debugging**: Clear working directory and path resolution

## 🔧 **Standard Implementation Pattern**

### **App Entry Point**
```python
# extensions/{algorithm}-v0.03/app.py

# MANDATORY: Path setup first
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

### **Script Launching with Proper Paths**
```python
from extensions.common.path_utils import get_extension_path

extension_path = get_extension_path(__file__)
script_path = extension_path / "scripts" / "main.py"

subprocess.run([
    "python", str(script_path),
    "--algorithm", algorithm_name,
    # ... other parameters
])
```

## 📋 **Compliance Requirements**

All extension Streamlit apps must:
- [ ] Use `ensure_project_root()` before other imports
- [ ] Use path utilities from `extensions/common/path_utils.py`
- [ ] Work when launched from any directory
- [ ] Use properly resolved paths for subprocess calls

---

**This standardized path management ensures reliable Streamlit app operation across all extensions and environments.**
