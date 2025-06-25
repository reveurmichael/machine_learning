# Dashboard Architecture for v0.03 Extensions

> **Important — Authoritative Reference:** This document supplements `unified-streamlit-architecture-guide.md` (Streamlit OOP) and `final-decision-5.md` (Directory Structure).

## 🎯 **Dashboard Philosophy**

The `dashboard/` directory is **mandatory for all v0.03+ extensions** and serves as the organizational hub for modular UI components following the Object-Oriented Streamlit architecture established in `unified-streamlit-architecture-guide.md`.

## 🧠 **Separation of UI Concerns**

### **Core Architecture Pattern**
- **`app.py`**: Main Streamlit application class (the conductor)
- **`dashboard/`**: Modular UI components (the instruments)
- **`scripts/`**: Command-line functionality (the workers)

This separation ensures **script-runner philosophy**: the UI primarily launches scripts from the `scripts/` directory using `subprocess` with user-configured parameters.

## 🏗️ **Mandatory Directory Structure**

### **v0.03 Template**
```
extensions/{algorithm}-v0.03/
├── app.py                         # OOP Streamlit application
├── dashboard/                     # Modular UI components
│   ├── __init__.py
│   ├── base_tab.py               # Common tab functionality
│   ├── tab_main.py               # Algorithm execution
│   ├── tab_evaluation.py         # Performance analysis
│   ├── tab_replay.py             # Replay interface
│   └── tab_visualization.py      # Results display
├── scripts/                       # CLI entry points
│   ├── main.py                   # Core functionality
│   ├── generate_dataset.py       # Dataset generation
│   └── replay.py                 # Replay functionality
└── agents/                        # Copied exactly from v0.02
    └── [agent files unchanged]
```

**Standard Dashboard Template**: All v0.03 extensions MUST use this exact dashboard structure.

## 🎨 **UI Component Organization**

> **Authoritative Reference**: See `unified-streamlit-architecture-guide.md` for complete OOP Streamlit implementation patterns.

### **Tab-Based Architecture**
Each dashboard component represents a distinct functionality:

```python
# dashboard/tab_main.py - Primary algorithm interface
def render(session_state):
    """Render main algorithm execution interface"""
    # Parameter controls
    # Script launch interface
    # Results display

# dashboard/tab_evaluation.py - Performance analysis
def render(session_state):
    """Render evaluation and benchmarking interface"""
    # Performance metrics
    # Comparison tools
    # Export functionality
```

### **Script-Runner Integration**
```python
# Example from dashboard component
import subprocess

def launch_algorithm(algorithm: str, params: dict):
    """Launch algorithm script with user parameters"""
    cmd = f"python scripts/main.py --algorithm {algorithm}"
    for key, value in params.items():
        cmd += f" --{key} {value}"
    
    # Launch with subprocess
    subprocess.run(cmd, shell=True)
```

## 🚀 **Benefits of Dashboard Architecture**

### **Modularity**
- **Isolated Components**: Each tab handles one specific function
- **Reusable Design**: Components can be shared across extensions
- **Easy Maintenance**: Updates isolated to specific components

### **Scalability**
- **Plugin Architecture**: Easy to add new dashboard components
- **Extension Points**: Clear interfaces for custom functionality
- **Performance**: Lazy loading of complex components

### **User Experience**
- **Consistent Interface**: Same tab structure across all extensions
- **Intuitive Navigation**: Clear separation of different functions
- **Responsive Design**: Adapts to different screen sizes

## 📋 **Implementation Guidelines**

### **Dashboard Component Standards**
```python
# Required pattern for all dashboard components
def render(session_state: dict) -> None:
    """
    Render dashboard component
    
    Args:
        session_state: Streamlit session state for state management
    """
    # Component-specific UI logic
    # Parameter collection
    # Script launching
    # Results display
```

### **State Management**
- Use Streamlit `session_state` for persistence
- Share common state between dashboard components
- Isolate component-specific state appropriately

### **Error Handling**
- Consistent error display across all components
- User-friendly error messages
- Graceful degradation for missing functionality

## 🔧 **Compliance Checklist**

- [ ] Does the v0.03 extension have a `dashboard/` directory?
- [ ] Are UI components separated into individual files?
- [ ] Does each component follow the `render(session_state)` pattern?
- [ ] Does the main `app.py` use OOP architecture from `unified-streamlit-architecture-guide.md`?
- [ ] Do dashboard components primarily launch scripts via subprocess?

## 🔗 **See Also**

- **`unified-streamlit-architecture-guide.md`**: Authoritative reference for OOP Streamlit patterns
- **`final-decision-9.md`**: Streamlit architectural decisions
- **`final-decision-5.md`**: Directory structure standards
- **`app.md`**: Streamlit-specific path management

---

**The dashboard architecture ensures maintainable, scalable, and user-friendly interfaces while maintaining the script-runner philosophy that keeps core functionality accessible via command line.**
