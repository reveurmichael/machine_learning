"""
Supervised Learning v0.03 - Dashboard Module
--------------------

Dashboard components for Streamlit web interface.
Modular, focused components for different dashboard sections.

Design Pattern: Component Pattern
- Focused, single-responsibility components
- Clean interface for Streamlit integration
- Modular dashboard architecture
"""

from .training_dashboard import (
    TrainingDashboard,
    create_training_dashboard
)

__all__ = [
    "TrainingDashboard",
    "create_training_dashboard"
] 