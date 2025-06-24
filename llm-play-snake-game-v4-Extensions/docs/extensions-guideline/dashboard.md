# The Dashboard Architecture

## 🎯 **Executive Summary**

This document outlines the purpose and architecture of the `dashboard/` directory, a mandatory component for all `v0.03` and later extensions. The `dashboard/` directory serves as the dedicated location for organizing all modular UI components, such as tabs and views, that are used by the extension's main `app.py` Streamlit application.

## 🧠 **Core Philosophy: Separation of UI Concerns**

As a Streamlit application grows, placing all UI code into a single `app.py` file becomes unmanageable. The `dashboard/` directory enforces a **separation of concerns** for the user interface.

*   **`app.py` (The Conductor):** The main `app.py` file is responsible for the overall application structure and lifecycle, as defined by the `BaseExtensionApp` architecture. It acts as a "conductor," deciding which UI components to render and when.
*   **`dashboard/` (The UI Components):** This directory contains the individual, modular pieces of the UI. Each file in this directory should ideally correspond to a distinct part of the interface, such as a Streamlit tab or a complex, reusable component.

This separation makes the UI code easier to navigate, develop, and test.

## 🏗️ **Architectural Integration**

The `dashboard/` directory works in direct concert with the mandatory OOP Streamlit architecture. The main `app.py` class will import and instantiate UI components from the `dashboard/` directory to build the final user interface.

### **Example Structure for `heuristics-v0.03`**

```
extensions/heuristics-v0.03/
├── app.py              # The main application class (HeuristicsApp)
└── dashboard/
    ├── __init__.py
    ├── run_agent_tab.py  # A class or function that renders the "Run Agent" tab
    ├── dataset_tab.py    # A class or function for the "Generate Dataset" tab
    └── replay_tab.py     # A class or function for the "Replay Game" tab
```

### **How `app.py` Uses the Dashboard Components**

The main `HeuristicsApp` class would then use these components to render its body.

```python
# extensions/heuristics-v0.03/app.py

import streamlit as st
from extensions.common.app_utils import BaseExtensionApp
# Import the UI components from the dashboard directory
from .dashboard import run_agent_tab, dataset_tab, replay_tab

class HeuristicsApp(BaseExtensionApp):
    # ... (get_extension_name and render_sidebar methods) ...

    def render_body(self):
        """Renders the main body using tabs defined in the dashboard."""
        st.header(f"Algorithm: `{st.session_state.selected_algorithm}`")

        # Create tabs in the UI
        tab1, tab2, tab3 = st.tabs(["Run Agent", "Generate Dataset", "Replay Game"])

        with tab1:
            # Delegate the rendering of this tab's content to the imported component
            run_agent_tab.render(st.session_state)

        with tab2:
            dataset_tab.render(st.session_state)

        with tab3:
            replay_tab.render(st.session_state)

if __name__ == "__main__":
    HeuristicsApp()
```

## 🚀 **The "Script-Runner" Philosophy**

It is critical to remember that the primary goal of the v0.03 Streamlit UI is **not** to re-implement core logic. Instead, the UI components in the `dashboard/` should be designed to:
1.  Provide user-friendly controls (sliders, selectors, etc.) to configure a task.
2.  Use those user-defined configurations to build a command-line string.
3.  Execute the appropriate script from the `scripts/` directory using `subprocess`.

This ensures that all core functionality remains accessible via the command line, and the Streamlit app serves as a convenient, interactive "frontend" for these powerful scripts.

## 📋 **Compliance Checklist**

- [ ] Does the `v0.03` extension contain a `dashboard/` directory?
- [ ] Are modular UI components (like tabs) separated into their own files within `dashboard/`?
- [ ] Does the main `app.py` import and use the components from the `dashboard/` directory to construct its UI?
- [ ] Does the UI logic in the dashboard components focus on configuring and launching scripts from the `scripts/` folder?

---

> **The `dashboard/` architecture is key to building scalable and maintainable Streamlit applications. By properly separating UI components, we keep our `app.py` clean and our overall interface modular.**
