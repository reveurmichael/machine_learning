"""Heuristics LLM Fine-tuning Integration v0.03 - Web Interface & Interactive Training

Evolution from v0.02: Production-ready web interface with Streamlit app, interactive
training monitoring, model comparison dashboards, and enhanced user experience.

Key Features v0.03:
- 🌐 Interactive Streamlit web application with multiple tabs
- 📊 Real-time training progress monitoring and visualization
- 🔄 Model comparison dashboard with statistical analysis
- 🎮 Web-based replay system for fine-tuned model evaluation
- 📈 Interactive charts and performance metrics
- 🛠️ User-friendly parameter configuration interface
- 💾 Dataset management and preprocessing tools
- 🚀 One-click training pipeline execution
- 📋 Comprehensive evaluation suite with visual results

Design Patterns Implemented:
- Model-View-Controller (MVC): Separates UI from business logic
- Observer Pattern: Real-time updates and progress notifications
- Repository Pattern: Centralized dataset access and management
- Strategy Pattern: Different training and preprocessing strategies
- Factory Pattern: Configuration builders and dataset creation
- Facade Pattern: Simplified web interface over complex operations

Architecture:
- Frontend: Streamlit multi-tab application + Flask dashboard
- Backend: Reuses v0.02 training pipeline and evaluation suite
- Data Layer: Enhanced dataset manager with preprocessing
- Communication: WebSocket for real-time updates

Usage:
    # Launch web interface (recommended)
    python cli.py web --port 8501
    
    # Launch Flask dashboard
    python cli.py dashboard --port 5000
    
    # CLI operations
    python cli.py datasets --list
    python cli.py train --strategy LoRA --model gpt2
    
    # System information
    python cli.py info

Dependencies:
- v0.02 components for core training functionality
- Streamlit for the main web interface
- Flask + SocketIO for real-time dashboard
- Common utilities for shared functionality

Evolution Path:
- v0.01: Proof of concept with basic pipeline
- v0.02: Production CLI with advanced features
- v0.03: Interactive web interface with real-time monitoring
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

__version__ = "0.03"
__title__ = "Heuristics LLM Fine-tuning Integration v0.03"
__description__ = "Interactive web interface for LLM fine-tuning with heuristic datasets"
__evolution__ = "v0.02 CLI to v0.03 Web Interface"

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure working directory is project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

# Lazy imports to avoid circular dependencies and missing dependencies
_app = None
_web_interface = None
_dataset_manager = None
_cli = None


def get_app():
    """Get the Streamlit application module."""
    global _app
    if _app is None:
        try:
            from . import app
            _app = app
        except ImportError as e:
            raise ImportError(f"Failed to import Streamlit app: {e}")
    return _app


def get_web_interface():
    """Get the Flask web interface module."""
    global _web_interface
    if _web_interface is None:
        try:
            from . import web_interface
            _web_interface = web_interface
        except ImportError as e:
            raise ImportError(f"Failed to import Flask web interface: {e}")
    return _web_interface


def get_dataset_manager():
    """Get the dataset manager module."""
    global _dataset_manager
    if _dataset_manager is None:
        try:
            from . import dataset_manager
            _dataset_manager = dataset_manager
        except ImportError as e:
            raise ImportError(f"Failed to import dataset manager: {e}")
    return _dataset_manager


def get_cli():
    """Get the CLI module."""
    global _cli
    if _cli is None:
        try:
            from . import cli
            _cli = cli
        except ImportError as e:
            raise ImportError(f"Failed to import CLI: {e}")
    return _cli


def launch_web_interface(port: int = 8501, host: str = "localhost") -> None:
    """Launch the Streamlit web interface.
    
    Args:
        port: Port number for the web interface
        host: Host address for the web interface
    """
    cli_module = get_cli()
    cli_module.launch_streamlit_app(port=port, host=host)


def launch_dashboard(port: int = 5000, debug: bool = False) -> None:
    """Launch the Flask dashboard.
    
    Args:
        port: Port number for the dashboard
        debug: Enable debug mode
    """
    cli_module = get_cli()
    cli_module.launch_flask_dashboard(port=port, debug=debug)


def create_dataset_manager(base_path: Optional[str] = None):
    """Create a dataset manager instance.
    
    Args:
        base_path: Base path for datasets (default: logs/extensions/datasets)
        
    Returns:
        DatasetManager instance
    """
    dm_module = get_dataset_manager()
    return dm_module.create_dataset_manager(base_path)


def get_system_info() -> Dict[str, Any]:
    """Get system information and extension status.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'version': __version__,
        'title': __title__,
        'description': __description__,
        'evolution': __evolution__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'working_directory': os.getcwd(),
        'dependencies': {}
    }
    
    # Check dependencies
    dependencies = {
        'streamlit': 'Streamlit web interface',
        'flask': 'Flask dashboard',
        'pandas': 'Data processing',
        'numpy': 'Numerical computations',
        'transformers': 'Model training',
        'torch': 'PyTorch backend'
    }
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            info['dependencies'][dep] = {'available': True, 'description': description}
        except ImportError:
            info['dependencies'][dep] = {'available': False, 'description': description}
    
    # Check v0.02 availability
    try:
        from extensions.heuristics_llm_fine_tuning_integration_v0_02 import pipeline
        info['v0_02_available'] = True
    except ImportError:
        info['v0_02_available'] = False
    
    return info


# Export main components for direct import
__all__ = [
    'launch_web_interface',
    'launch_dashboard', 
    'create_dataset_manager',
    'get_system_info',
    'get_app',
    'get_web_interface',
    'get_dataset_manager',
    'get_cli',
    '__version__',
    '__title__',
    '__description__',
    '__evolution__'
] 