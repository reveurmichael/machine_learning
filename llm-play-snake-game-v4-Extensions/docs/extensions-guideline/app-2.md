# Advanced Streamlit Architecture

> **Important — Authoritative Reference:** This document supplements Final Decision 9 (Streamlit OOP Architecture).

## 🎯 **Script-Runner Philosophy**

Following Final Decision 9, all v0.03+ extensions use **Object-Oriented Streamlit architecture** with the **script-runner philosophy**: launch scripts from the `scripts/` directory with user-configured parameters via `subprocess`.

## 🧠 **What Streamlit Apps Are For**

### **✅ Streamlit Apps ARE for:**
- User-friendly parameter configuration
- Script launching with `subprocess`  
- Results display and analysis
- Dashboard organization

### **❌ Streamlit Apps are NOT for:**
- Real-time game visualization
- Complex game state management
- Direct algorithm implementation

## 🏗️ **OOP Architecture Requirements**

### **Base Class Pattern**
```python
from extensions.common.app_utils import BaseExtensionApp

class AlgorithmApp(BaseExtensionApp):
    def setup_page(self) -> None:
        """Configure page settings and layout"""
        pass
    
    def main(self) -> None:
        """Main application logic - script launching interface"""
        pass
```

### **Standard Tab Structure**
```python
def main(self):
    tab1, tab2, tab3 = st.tabs([
        "Run Algorithm",     # Primary functionality
        "Generate Dataset",  # Data generation  
        "Replay & Analysis"  # Results analysis
    ])
```

## 🔧 **Script Integration**

### **Standard Launching Pattern**
```python
import subprocess
from extensions.common.path_utils import get_extension_path

def launch_script(self, script_name: str, params: dict):
    extension_path = get_extension_path(__file__)
    script_path = extension_path / "scripts" / f"{script_name}.py"
    
    cmd = ["python", str(script_path)]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    return subprocess.run(cmd, capture_output=True, text=True)
```

## 🎨 **Benefits**

### **Consistency**
- Same interface patterns across all algorithm types
- Predictable user experience
- Standardized error handling

### **Maintainability**  
- Clear separation of concerns
- Modular dashboard components
- Reusable base class functionality

---

**This architecture ensures consistent, maintainable interfaces while maintaining the script-runner philosophy that keeps core functionality accessible via command line.**