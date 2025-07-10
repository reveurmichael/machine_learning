# Streamlit App Pattern for Snake Game AI Extensions

> **Reference:** See `final-decision.md` (SUPREME_RULES), `scripts.md`, `standalone.md`.

## 🎯 **Core Philosophy: Script Launcher Interface (SUPREME_RULE NO.5)**

All v0.03+ extensions must provide a `Streamlit app.py` whose **sole purpose** is to launch scripts with adjustable parameters. The app is **NOT** for real-time visualization, game state display, or complex UI features - it serves exclusively as a user-friendly frontend for backend script execution.

### **Educational Value**
- **Script Integration**: Understanding how to bridge UI and backend execution
- **Parameter Management**: Learning to handle user input and script parameters
- **Subprocess Usage**: Demonstrating proper subprocess execution patterns
- **SUPREME_RULES Compliance**: Following the script launcher mandate from `final-decision.md`

## 🏗️ **Factory Pattern: Canonical Method is create()**

All app factories must use the canonical method name `create()` for instantiation, not `create_app()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision.md`.

### **Reference Implementation**

```python
from utils.factory_utils import SimpleFactory

class StreamlitApp:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("streamlit", StreamlitApp)
app = factory.create("streamlit", name="TestApp")  # CANONICAL create() method per SUPREME_RULES
print_info(f"App name: {app.name}")  # SUPREME_RULES compliant logging
```

## ✅ **Key Requirements**

### **Script Launcher Focus**
- **Primary Purpose**: Launch backend scripts with adjustable parameters
- **No Visualization**: Do not display game state or real-time progress
- **No Statistics**: Do not show complex statistics or analytics
- **Simple Interface**: Keep UI clean and focused on parameter adjustment

### **Logging Standards**
- **Simple Logging**: Use `utils/print_utils.py` functions only
- **No .log Files**: Never produce `.log` files (SUPREME_RULE NO.3)
- **Consistent Format**: Use `print_info`, `print_warning`, `print_error`, `print_success`

### **Subprocess Integration**
```python
import subprocess
from utils.print_utils import print_info

def launch_script(script_path: str, parameters: dict):
    """Launch backend script with parameters"""
    cmd = [script_path] + [f"--{k}={v}" for k, v in parameters.items()]
    print_info(f"[App] Launching script: {' '.join(cmd)}")  # SUPREME_RULES compliant logging
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result
```

## 🎯 **Implementation Example**

```python
import streamlit as st
import subprocess
from utils.print_utils import print_info

class ExtensionApp:
    """Streamlit app for launching extension scripts"""
    
    def __init__(self):
        print_info(f"[ExtensionApp] Initialized")  # SUPREME_RULES compliant logging
    
    def run(self):
        st.title("Extension Script Launcher")
        
        # Parameter inputs
        algorithm = st.selectbox("Algorithm", ["BFS", "ASTAR", "DFS"])
        grid_size = st.slider("Grid Size", 8, 20, 10)
        max_games = st.slider("Max Games", 1, 100, 10)
        
        if st.button("Launch Script"):
            self._launch_script(algorithm, grid_size, max_games)
    
    def _launch_script(self, algorithm: str, grid_size: int, max_games: int):
        """Launch backend script with parameters"""
        cmd = [
            "python", "scripts/main.py",
            "--algorithm", algorithm,
            "--grid-size", str(grid_size),
            "--max-games", str(max_games)
        ]
        
        print_info(f"[ExtensionApp] Launching: {' '.join(cmd)}")  # SUPREME_RULES compliant logging
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Script completed successfully")
            st.text(result.stdout)
        else:
            st.error("Script failed")
            st.text(result.stderr)
```

## 🔗 **See Also**

- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`scripts.md`**: Backend script organization and execution
- **`standalone.md`**: Standalone principle and extension independence


