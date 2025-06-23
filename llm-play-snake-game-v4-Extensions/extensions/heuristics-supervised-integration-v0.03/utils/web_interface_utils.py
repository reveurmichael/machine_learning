"""web_interface_utils.py - Web Interface Utilities

Utilities for managing Streamlit web interface components and state management.
"""

import streamlit as st
from typing import Dict, Any


class WebInterfaceUtils:
    """Utilities for web interface management."""
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check availability of web dependencies."""
        dependencies = {}
        
        try:
            import streamlit
            dependencies['streamlit'] = True
        except ImportError:
            dependencies['streamlit'] = False
        
        try:
            import plotly
            dependencies['plotly'] = True
        except ImportError:
            dependencies['plotly'] = False
        
        try:
            import pandas
            dependencies['pandas'] = True
        except ImportError:
            dependencies['pandas'] = False
        
        return dependencies
    
    @staticmethod
    def show_dependency_status(dependencies: Dict[str, bool]):
        """Show dependency status in sidebar."""
        st.sidebar.markdown("## 🔧 System Status")
        
        for dep, available in dependencies.items():
            if available:
                st.sidebar.success(f"✅ {dep}")
            else:
                st.sidebar.error(f"❌ {dep}")
        
        readiness = sum(dependencies.values()) / len(dependencies)
        st.sidebar.metric("Readiness", f"{readiness:.0%}")
    
    @staticmethod
    def create_progress_indicator(current: int, total: int, label: str = "Progress"):
        """Create a progress indicator with label."""
        progress = current / total if total > 0 else 0
        st.progress(progress)
        st.text(f"{label}: {current}/{total} ({progress:.1%})")
    
    @staticmethod
    def display_metrics_grid(metrics: Dict[str, Any], columns: int = 4):
        """Display metrics in a grid layout."""
        cols = st.columns(columns)
        
        for idx, (key, value) in enumerate(metrics.items()):
            with cols[idx % columns]:
                if isinstance(value, (int, float)):
                    st.metric(key, f"{value:.3f}" if isinstance(value, float) else str(value))
                else:
                    st.metric(key, str(value))
