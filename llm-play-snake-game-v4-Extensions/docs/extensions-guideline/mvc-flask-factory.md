# MVC Flask Factory Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document defines the comprehensive MVC Flask factory architecture used throughout the Snake Game AI project and all extensions (Tasks 1-5).

> **See also:** `task0.md`, `flask.md`, `network.md`, `kiss.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Layered MVC with Universal Factory Patterns**

The Snake Game AI project implements a sophisticated **layered MVC architecture** combined with **universal factory patterns** to provide a scalable, educational, and maintainable web infrastructure. This architecture serves as the foundation for Task-0 and the template for all future extensions.

### **Educational Value**
- **Design Pattern Mastery**: Comprehensive demonstration of MVC, Factory, Template Method, and Facade patterns
- **Layered Architecture**: Clear inheritance hierarchy with educational benefits
- **Universal Utilities**: SSOT compliance with factory_utils from ROOT/utils/
- **Enhanced Naming**: Self-documenting code with explicit domain indication

## 🏗️ **Layered MVC Architecture Overview**

### **Three-Layer Web Infrastructure**
```
Layer 1: BaseWebApp (Universal Foundation)
├── Universal Flask configuration
├── Common routing patterns
├── Shared utilities and middleware
└── Base application lifecycle

Layer 2: SimpleFlaskApp (Game-Specific)
├── Game-oriented conveniences
├── Common game API patterns
├── Standardized response formats
└── Game state management

Layer 3: BaseReplayApp (Replay-Specific)
├── Replay infrastructure patterns
├── Universal replay utilities
├── ReplayEngine integration
└── Replay-specific lifecycle
```

### **Enhanced Naming Convention**
```python
# Enhanced Naming Pattern: {Domain}{Type}{Purpose}{Layer}
BaseWebApp              # Universal base for all web applications
SimpleFlaskApp          # Game-specific Flask application layer
BaseReplayApp           # Universal replay application layer

HumanWebGameApp         # Human + Web + Game + App
LLMWebGameApp           # LLM + Web + Game + App  
ReplayWebGameApp        # Replay + Web + Game + App

HumanWebAppLauncher     # Human + Web + App + Launcher
LLMWebAppLauncher       # LLM + Web + App + Launcher
ReplayWebAppLauncher    # Replay + Web + App + Launcher
```

## 🎨 **MVC Architecture Components**

### **Model Layer (Enhanced)**
```python
# Base Model with Universal Utilities
class BaseWebModel:
    """
    Universal base model for all web applications.
    
    Design Pattern: Template Method Pattern (Model Layer)
    Purpose: Provides consistent data management across all applications
    Educational Value: Shows universal model patterns with enhanced naming
    """
    
    def __init__(self, **config):
        self.config = config
        self.state = {}
        print_log = create_logger("BaseWebModel")
        print_log("Base web model initialized with universal utilities")
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get application data for template rendering."""
        return {
            'name': self.__class__.__name__,
            'type': self._get_app_type(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_app_type(self) -> str:
        """Get application type with enhanced naming."""
        class_name = self.__class__.__name__
        if 'Human' in class_name:
            return 'human_web_game_app'
        elif 'LLM' in class_name:
            return 'llm_web_game_app'
        elif 'Replay' in class_name:
            return 'replay_web_game_app'
        return 'base_web_app'
```


### **View Layer (Enhanced Templates)**
```python
# Base View Renderer with Universal Templates
class BaseWebViewRenderer:
    """Universal base view renderer for all web applications."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = template_dir
        self.template_env = self._setup_template_environment()
    
    def render_app_template(self, template_name: str, **context) -> str:
        """Render application template with enhanced context."""
        enhanced_context = self._enhance_template_context(context)
        template = self.template_env.get_template(template_name)
        return template.render(**enhanced_context)
```

### **Controller Layer (Enhanced API)**
```python
# Base Controller with Universal API Patterns
class BaseWebController:
    """Universal base controller for all web applications."""
    
    def __init__(self, model: BaseWebModel, view: BaseWebViewRenderer):
        self.model = model
        self.view = view
    
    def handle_index_request(self) -> str:
        """Handle index page request with enhanced rendering."""
        app_data = self.model.get_app_data()
        return self.view.render_app_template('index.html', **app_data)
    
    def handle_api_state_request(self) -> Dict[str, Any]:
        """Handle API state request with enhanced response."""
        state = self.model.get_app_data()
        return build_success_response(state)
```

## 🏭 **Universal Factory Architecture**

### **Multi-Layer Factory Pattern**
```python
# Universal Factory Utilities (ROOT/utils/factory_utils.py)
class WebAppFactory(SimpleFactory):
    """Universal web application factory following SSOT principles."""
    
    _registry = {}
    
    @classmethod
    def create(cls, app_type: str, **kwargs) -> BaseWebApp:
        """Create web application using canonical create() method."""
        app_class = cls._registry.get(app_type.lower())
        if not app_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown app type: {app_type}. Available: {available}")
        
        print_log = create_logger("WebAppFactory")
        print_log(f"Creating web application: {app_type}")
        
        return app_class(**kwargs)

# Enhanced Factory Functions
def create_human_web_game_app(grid_size: int = 10, port: Optional[int] = None, **kwargs):
    """Create human web game application using enhanced factory function."""
    print_log = create_logger("GameWebFactory")
    print_log(f"Creating human web game app with grid_size={grid_size}")
    
    return GameWebAppFactory.create('human', grid_size=grid_size, port=port, **kwargs)
```

## 🚀 **Extension Template Architecture**

### **Universal Extension Pattern**
```python
# Extension Template: extensions/{extension-type}-v0.03/web/
# Example: extensions/heuristics-v0.03/web/heuristic_web_app.py

from web.base_app import SimpleFlaskApp
from utils.factory_utils import SimpleFactory
from utils.text_utils import create_logger

class HeuristicWebGameApp(SimpleFlaskApp):
    """Heuristic web game application with enhanced naming."""
    
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10, **config):
        super().__init__(f"Heuristic Snake Web Game ({algorithm})")
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.config = config
        
        # Initialize heuristic-specific components
        self.pathfinder = self._create_pathfinder()
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get heuristic application data with enhanced naming."""
        return {
            'name': self.name,
            'mode': 'heuristic_web',
            'app_type': 'heuristic_web_game_app',
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'status': 'ready'
        }
```

## 📝 **Script Template Architecture**

### **Enhanced Script Pattern**
```python
# Extension Script Template: extensions/{extension-type}-v0.03/scripts/web_interface.py

"""
{Extension} Web Interface Script (Enhanced Architecture)
======================================================

Flask-based web application for {extension} algorithms using the enhanced layered web
infrastructure. This script demonstrates the extension's modern web architecture
and follows the canonical template established by Task-0.

Extension Template: Copy from scripts/human_play_web.py and customize for {extension}
"""

import sys
import argparse
from pathlib import Path

# Ensure project root for consistent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import enhanced web infrastructure (Extension-Specific)
from web import {Extension}WebGameApp, {Extension}WebAppLauncher
from web.factories import create_{extension}_web_game_app

# Import universal utilities following SSOT principles
from utils.validation_utils import validate_grid_size, validate_port
from utils.text_utils import create_logger

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for {extension} web game interface."""
    parser = argparse.ArgumentParser(
        description="Snake Game - {Extension} Web Game Interface (Enhanced Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  python scripts/web_interface.py                    # Default {extension} settings
  python scripts/web_interface.py --algorithm BFS    # Specific algorithm
  python scripts/web_interface.py --grid-size 15     # Larger grid with validation
  python scripts/web_interface.py --debug            # Debug mode with enhanced logging

Extension Architecture:
  - Layered inheritance: BaseWebApp → SimpleFlaskApp → {Extension}WebGameApp
  - Enhanced naming: Clear domain indication ({Extension} + Web + Game + App)
  - Universal utilities: Validation, logging, factory patterns from SSOT
  - Algorithm integration: {Extension}-specific algorithms and strategies
        """
    )
    
    # Extension-specific arguments
    parser.add_argument("--algorithm", type=str, default="BFS",
                       help="{Extension} algorithm to use (default: BFS)")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Size of the game grid with validation (default: 10)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host address for the web server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None,
                       help="Port number with conflict detection (default: auto-detect)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable Flask debug mode with enhanced logging")
    
    return parser

def main() -> int:
    """Main entry point for {extension} web interface script."""
    try:
        print_log = create_logger("{Extension}WebScript")
        print_log("🚀 Starting Snake Game - {Extension} Web Interface (Enhanced)")
        print_log("📊 Architecture: Enhanced Layered Web Infrastructure")
        print_log("🎯 Mode: {Extension} Algorithms with Enhanced Naming")
        
        # Parse and validate arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Validate using universal utilities
        args.grid_size = validate_grid_size(args.grid_size)
        args.port = validate_port(args.port)
        
        # Create enhanced {extension} web application
        app = create_{extension}_web_game_app(
            algorithm=args.algorithm,
            grid_size=args.grid_size,
            port=args.port
        )
        
        # Display application information
        print_log(f"✅ {Extension} web app created: {app.name}")
        print_log(f"   Algorithm: {app.algorithm}")
        print_log(f"   Grid Size: {app.grid_size}x{app.grid_size}")
        print_log(f"   URL: http://{args.host}:{app.port}")
        
        # Start enhanced web application
        app.run(host=args.host, port=app.port, debug=args.debug)
        return 0
        
    except Exception as e:
        print_log(f"❌ Failed to start {extension} web interface: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

## 📋 **Implementation Checklist**

### **Universal Requirements**
- [ ] **Enhanced Naming**: All classes use {Domain}{Type}{Purpose}{Layer} pattern
- [ ] **Layered Architecture**: Proper inheritance from BaseWebApp → SimpleFlaskApp → SpecificApp
- [ ] **Universal Utilities**: Uses factory_utils, validation_utils, text_utils from SSOT
- [ ] **Canonical Methods**: All factories use canonical create() method
- [ ] **Educational Documentation**: Comprehensive docstrings with design patterns

### **MVC Component Standards**
- [ ] **Model Layer**: BaseWebModel with enhanced naming and universal utilities
- [ ] **View Layer**: BaseWebViewRenderer with enhanced templates and context
- [ ] **Controller Layer**: BaseWebController with enhanced API patterns
- [ ] **Integration**: Proper integration between MVC components

### **Factory Pattern Standards**
- [ ] **Universal Factory**: WebAppFactory in ROOT/utils/factory_utils.py
- [ ] **Game Factory**: GameWebAppFactory with enhanced naming
- [ ] **Extension Factories**: Extension-specific factories with canonical create() method
- [ ] **Convenience Functions**: Factory functions with enhanced naming

### **Extension Template Standards**
- [ ] **Script Templates**: Enhanced script patterns in scripts/ folder
- [ ] **Application Classes**: Enhanced web app classes with layered inheritance
- [ ] **Factory Integration**: Proper factory pattern usage with canonical methods
- [ ] **Documentation**: Comprehensive documentation with extension guidance

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`task0.md`**: Task-0 foundational architecture and base class patterns
- **`flask.md`**: Flask integration patterns for extensions
- **`network.md`**: Dynamic port allocation and networking architecture
- **`kiss.md`**: KISS principles and canonical method naming
- **`factory-design-pattern.md`**: Universal factory pattern implementation

### **Implementation Files**
- **`ROOT/utils/factory_utils.py`**: Universal factory utilities and patterns
- **`ROOT/web/base_app.py`**: Layered web infrastructure foundation
- **`ROOT/web/factories.py`**: Game-specific factory implementations
- **`ROOT/scripts/{type}_web.py`**: Enhanced script templates

### **Extension Integration**
- **Extensions Web Structure**: `extensions/{type}-v0.03/web/` for extension web apps
- **Extension Scripts**: `extensions/{type}-v0.03/scripts/` for extension web scripts
- **Extension Factories**: Extension-specific factory implementations
- **Extension Documentation**: Extension-specific MVC documentation and patterns

---

**This MVC Flask factory architecture provides a comprehensive, educational, and scalable foundation for web interfaces across all Snake Game AI tasks and extensions, following enhanced naming conventions and universal design patterns.**
