# Unified Streamlit Architecture Guide

> **Authoritative Reference**: This document provides the **single canonical Streamlit OOP architecture** for all v0.03+ extensions.

## 🎯 **Core Streamlit Philosophy**

Streamlit applications in v0.03+ extensions follow:
- **Object-Oriented Design** with clear class hierarchies
- **Script-Runner Pattern** - UI launches scripts via subprocess
- **Modular Dashboard Components** organized in dashboard/ folder
- **Consistent User Experience** across all extensions

## 🏗️ **Base Architecture Pattern**

### **Core Base Class**
```python
import streamlit as st
import subprocess
from abc import ABC, abstractmethod

class BaseExtensionApp(ABC):
    """Base class for all extension Streamlit applications"""
    
    def __init__(self):
        self.setup_page_config()
        self.main()
    
    @abstractmethod
    def get_extension_name(self) -> str:
        """Return the human-readable extension name"""
        pass
    
    @abstractmethod
    def get_available_algorithms(self) -> list[str]:
        """Return list of available algorithms"""
        pass
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{self.get_extension_name()} Dashboard",
            layout="wide"
        )
    
    def main(self) -> None:
        """Main application flow"""
        st.title(f"🐍 {self.get_extension_name()} Dashboard")
        self.render_sidebar()
        self.render_main_content()
    
    def render_sidebar(self) -> None:
        """Render sidebar with common controls"""
        with st.sidebar:
            st.selectbox("Algorithm", self.get_available_algorithms())
            st.selectbox("Grid Size", [8, 10, 12, 16, 20])
    
    @abstractmethod
    def render_main_content(self) -> None:
        """Render main content - implemented by subclasses"""
        pass
    
    def launch_script(self, script_name: str, **params):
        """Launch script via subprocess"""
        cmd = ["python", f"scripts/{script_name}"]
        subprocess.run(cmd)
```

## 🔧 **Extension Implementation**

```python
# extensions/heuristics-v0.03/app.py

from extensions.common.path_utils import ensure_project_root
ensure_project_root()

from extensions.common.app_utils import BaseExtensionApp

class HeuristicStreamlitApp(BaseExtensionApp):
    def get_extension_name(self) -> str:
        return "Heuristic Pathfinding"
    
    def get_available_algorithms(self) -> list[str]:
        return ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
    
    def render_main_content(self) -> None:
        tab1, tab2 = st.tabs(["Run", "Evaluate"])
        
        with tab1:
            if st.button("Run Algorithm"):
                self.launch_script("main.py")

if __name__ == "__main__":
    HeuristicStreamlitApp()
```

---

**This unified architecture ensures consistent interfaces across all extensions.** 