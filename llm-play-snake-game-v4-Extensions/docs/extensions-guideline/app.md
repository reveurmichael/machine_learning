# Streamlit App Path Management

> **Important — Reference Redirect:** Streamlit apps have unique needs, but the **authoritative path-management guide is `final-decision-6.md`**.  
> This file shows only the *Streamlit-specific* snippet; all generic patterns and rationales live in the Final Decision document.

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

## 🔧 **Standard Implementation Pattern (Streamlit Only)**

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

For any **non-Streamlit** scripts (training, dataset generation, etc.) follow the generic template in `final-decision-6.md`.
