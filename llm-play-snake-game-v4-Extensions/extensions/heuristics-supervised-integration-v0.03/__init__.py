"""heuristics-supervised-integration-v0.03 - Interactive Web Interface Extension

This extension represents the evolution from v0.02's CLI-only system to a comprehensive
web-based interface with Streamlit, interactive training, real-time monitoring, and
advanced visualization capabilities.

Evolution Overview:
- v0.01: Single MLP model proof-of-concept
- v0.02: Multi-framework production CLI with advanced configuration
- v0.03: Interactive web interface with real-time training and visualization

Key Features (v0.03):
- Streamlit web application for interactive training
- Real-time training progress monitoring
- Interactive model comparison dashboard
- Advanced visualization and analytics
- Dataset management and exploration
- Web-based hyperparameter tuning

Usage:
    streamlit run extensions/heuristics_supervised_integration_v0_03/app.py
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "0.03"
__author__ = "Extension Development Team"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_web_dependencies() -> Dict[str, bool]:
    """Check availability of web interface dependencies."""
    dependencies = {}
    
    try:
        import streamlit
        dependencies['streamlit'] = True
    except ImportError:
        dependencies['streamlit'] = False
        logger.warning("Streamlit not available - web interface disabled")
    
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

def launch_web_interface(port: int = 8501):
    """Launch the Streamlit web interface."""
    dependencies = check_web_dependencies()
    
    if not dependencies.get('streamlit', False):
        print("❌ Streamlit not available. Please install with:")
        print("   pip install streamlit plotly pandas")
        return
    
    import subprocess
    import sys
    from pathlib import Path
    
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    
    logger.info(f"Launching web interface on port {port}")
    subprocess.run(cmd)

# Initialize module
capabilities = check_web_dependencies()
if capabilities['streamlit']:
    logger.info("✅ Web interface available")
else:
    logger.warning("⚠️ Install streamlit for web interface: pip install streamlit plotly pandas")
