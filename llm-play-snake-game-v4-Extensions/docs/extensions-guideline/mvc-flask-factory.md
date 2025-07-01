# MVC Flask Factory Architecture Guide

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and defines the enhanced MVC Flask architecture with universal factory patterns.

> **See also:** `final-decision-10.md`, `flask.md`, `network.md`, `kiss.md`.

## 🎯 **Core Philosophy: Enhanced Layered Architecture**

The Snake Game AI project implements a **sophisticated MVC Flask architecture** with **universal factory patterns** that demonstrates educational excellence while providing a **canonical template** for all future extensions (Task 1-5). This architecture follows KISS principles while delivering enterprise-grade extensibility.

### **Educational Value**
- **Layered Inheritance**: Demonstrates sophisticated OOP patterns with clear separation of concerns
- **Factory Patterns**: Universal factory utilities with canonical `create()` methods
- **Single Source of Truth**: Centralized configuration and utilities following SSOT principles
- **Future-Proof Design**: Template for all extension web interfaces
- **Enhanced Naming**: Self-documenting architecture with clear domain indication

## 🏗️ **Architecture Overview**

### **Layered Inheritance Hierarchy**
```
BaseWebApp (Abstract Base)
    ↓
SimpleFlaskApp (Concrete Base)
    ↓
Task-Specific Apps (HumanWebGameApp, LLMWebGameApp, ReplayWebGameApp)
```

### **Universal Factory Pattern**
```python
# Universal factory utilities in ROOT/utils/factory_utils.py
class SimpleFactory:
    """Universal factory with canonical create() method (SUPREME_RULES compliance)"""
    
    def create(self, app_type: str, **kwargs):
        """Canonical create() method - single entry point for all app creation"""
        # Implementation follows KISS principles
```

## 📁 **File Organization**

### **Core Web Infrastructure (`ROOT/web/`)**
```
web/
├── __init__.py                 # Enhanced web module exports
├── base_app.py                # Abstract base classes with layered architecture
├── applications.py            # Task-specific Flask applications
├── factories.py               # Universal factory functions
├── static/                    # CSS, JavaScript, images
├── templates/                 # HTML templates with debug mode integration
└── controllers/               # MVC controllers (future extension)
```

### **Universal Utilities (`ROOT/utils/`)**
```
utils/
├── factory_utils.py           # Universal factory patterns
├── network_utils.py           # Dynamic port allocation
├── validation_utils.py        # Argument validation
├── print_utils.py             # Simple logging (SUPREME_RULES)
└── web_utils.py               # Web state utilities
```

### **Configuration (`ROOT/config/`)**
```
config/
├── web_constants.py           # Flask/JS/HTML/CSS constants
├── network_constants.py       # Network/host/port constants
└── ui_constants.py            # UI/grid/visualization constants
```

## 🎨 **Design Patterns Implementation**

### **1. Template Method Pattern (Layered Architecture)**
```python
class BaseWebApp(ABC):
    """
    Abstract base class for all web applications.
    
    Design Pattern: Template Method Pattern
    Purpose: Defines the skeleton of web application lifecycle
    Educational Value: Shows how to create extensible base classes
    """
    
    def __init__(self, name: str, port: int | None = None):
        self.name = name
        self.port = port or random_free_port()
        self.app = Flask(name)
        self._setup_routes()
    
    @abstractmethod
    def _setup_routes(self) -> None:
        """Subclasses must implement route configuration."""
        pass
    
    def run(self, host: str | None = None, debug: bool = FLASK_DEBUG_MODE) -> None:
        """Template method - consistent startup across all apps."""
        resolved_host, resolved_port = self._get_server_host_port(host)
        print(f"[{self.name}] Starting on http://{resolved_host}:{resolved_port}")
        self.app.run(host=resolved_host, port=resolved_port, debug=debug)
```

### **2. Factory Pattern (Universal Creation)**
```python
# Universal factory utilities
def create_human_web_game_app(grid_size: int = 10, port: int | None = None) -> HumanWebGameApp:
    """
    Factory function for human web game applications.
    
    Design Pattern: Factory Pattern (Universal Implementation)
    Purpose: Centralized app creation with validation
    Educational Value: Shows canonical factory pattern usage
    """
    validate_human_web_arguments(grid_size, port)
    return HumanWebGameApp(grid_size=grid_size, port=port)

def create_llm_web_game_app(grid_size: int = 10, port: int | None = None) -> LLMWebGameApp:
    """Factory function for LLM web game applications."""
    validate_llm_web_arguments(grid_size, port)
    return LLMWebGameApp(grid_size=grid_size, port=port)
```

### **3. Strategy Pattern (Pluggable Game Modes)**
```python
class HumanWebGameApp(SimpleFlaskApp):
    """
    Human-controlled web game application.
    
    Design Pattern: Strategy Pattern
    Purpose: Pluggable game mode implementation
    Educational Value: Shows how to specialize base classes
    """
    
    def _setup_routes(self) -> None:
        """Configure human-specific routes."""
        self.app.add_url_rule('/', 'index', self._human_game_index)
        self.app.add_url_rule('/game', 'game', self._human_game_play, methods=['GET', 'POST'])
        self.app.add_url_rule('/api/move', 'move', self._handle_human_move, methods=['POST'])
```

### **4. Facade Pattern (Simplified Interface)**
```python
class HumanWebAppLauncher:
    """
    Simplified launcher for human web applications.
    
    Design Pattern: Facade Pattern
    Purpose: Provides simple interface to complex web setup
    Educational Value: Shows how to hide complexity behind simple interfaces
    """
    
    @staticmethod
    def launch(grid_size: int = 10, port: int | None = None) -> None:
        """Launch human web game with minimal configuration."""
        app = create_human_web_game_app(grid_size=grid_size, port=port)
        app.run()
```

## 🔧 **Configuration Architecture**

### **Single Source of Truth (SSOT)**
```python
# config/web_constants.py - Flask/JS/HTML/CSS constants
FLASK_DEBUG_MODE: Final[bool] = True  # Controls both server and client debug behavior

# config/network_constants.py - Network/host/port constants  
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT_RANGE_START: Final[int] = 8000
DEFAULT_PORT_RANGE_END: Final[int] = 16000

# config/ui_constants.py - UI/grid/visualization constants
GRID_SIZE: Final[int] = 10
WINDOW_WIDTH: Final[int] = 800
WINDOW_HEIGHT: Final[int] = 600
```

### **Environment Variable Support**
```python
# Environment variables for deployment flexibility
HOST_ENV_VAR: Final[str] = "HOST"      # Override default host
PORT_ENV_VAR: Final[str] = "PORT"      # Override random port allocation
```

## 🚀 **Dynamic Port Allocation**

### **Random Port Strategy**
```python
def random_free_port(min_port: int = DEFAULT_PORT_RANGE_START, 
                    max_port: int = DEFAULT_PORT_RANGE_END) -> int:
    """
    Return a random free port within specified range.
    
    Design Pattern: Strategy Pattern (Port Allocation)
    Purpose: Provides conflict-free port allocation
    Educational Value: Shows robust resource management
    """
    for _ in range(MAX_PORT_ATTEMPTS):
        candidate = random.randint(min_port, max_port)
        if is_port_free(candidate):
            return candidate
    
    # Fallback to sequential search
    return find_free_port(min_port)
```

### **Benefits of Random Ports**
- ✅ **Parallel Development**: Multiple developers can work simultaneously
- ✅ **No Port Conflicts**: Automatic conflict resolution
- ✅ **CI/CD Friendly**: Works seamlessly in automated environments
- ✅ **Container Compatible**: Perfect for Docker/Kubernetes deployment

## 🎮 **Task-Specific Implementations**

### **Task-0 (LLM Snake Game)**
```python
# scripts/main_web.py
def main() -> int:
    """Main entry point for LLM web game interface."""
    try:
        print_log("🐍 Starting Snake Game - LLM Web Interface (Enhanced)")
        
        # Parse and validate arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_llm_web_arguments(args)
        
        # Create application using factory pattern
        app = create_llm_web_game_app(
            grid_size=args.grid_size,
            port=args.port
        )
        
        # Display application information
        display_application_info(app)
        
        # Launch application
        app.run(host=args.host)
        return 0
        
    except Exception as e:
        print_log(f"❌ Error: {e}")
        return 1
```

### **Task-1 (Heuristics) Extension Template**
```python
# extensions/heuristics-v0.03/scripts/replay_web.py
def main() -> int:
    """Main entry point for heuristic replay web interface."""
    try:
        print_log("🔍 Starting Heuristic Replay Web Interface (Enhanced)")
        
        # Parse and validate arguments
        parser = create_heuristic_argument_parser()
        args = parser.parse_args()
        validate_heuristic_web_arguments(args)
        
        # Create application using factory pattern
        app = create_heuristic_replay_app(
            algorithm=args.algorithm,
            port=args.port
        )
        
        # Display application information
        display_heuristic_application_info(app)
        
        # Launch application
        app.run(host=args.host)
        return 0
        
    except Exception as e:
        print_log(f"❌ Error: {e}")
        return 1
```

## 🎯 **Enhanced Naming Conventions**

### **Domain-Indicated Naming**
```python
# ✅ CORRECT: Clear domain indication
HumanWebGameApp      # Human + Web + Game + App
LLMWebGameApp        # LLM + Web + Game + App  
ReplayWebGameApp     # Replay + Web + Game + App
HeuristicReplayApp   # Heuristic + Replay + App

# ❌ AVOID: Generic naming
WebApp               # Too generic
GameApp              # Unclear domain
App                  # No domain indication
```

### **Factory Function Naming**
```python
# ✅ CORRECT: Consistent factory naming
create_human_web_game_app()
create_llm_web_game_app()
create_replay_web_game_app()
create_heuristic_replay_app()

# ❌ AVOID: Inconsistent naming
make_human_app()
build_llm_game()
new_replay_interface()
```

## 🔍 **Debug Mode Integration**

### **Server-Side Debug Mode**
```python
# config/web_constants.py - Single source of truth
FLASK_DEBUG_MODE: Final[bool] = True

# web/base_app.py - Server-side integration
def run(self, host: str | None = None, debug: bool = FLASK_DEBUG_MODE) -> None:
    """Run Flask application with centralized debug mode."""
    resolved_host, resolved_port = self._get_server_host_port(host)
    self.app.run(host=resolved_host, port=resolved_port, debug=debug)
```

### **Client-Side Debug Mode**
```python
# web/base_app.py - Template integration
def render_template(self, template_name: str, **app_data) -> str:
    """Render template with debug mode injection."""
    return render_template(template_name, debug_mode=FLASK_DEBUG_MODE, **app_data)

# templates/base.html - Client-side integration
<script>
    const DEBUG_MODE = {{ debug_mode|tojson }};
    
    if (DEBUG_MODE) {
        console.log('[Debug] Application initialized');
        console.log('[Debug] Game state:', gameState);
    }
</script>
```

## 📊 **Validation Architecture**

### **Universal Validation Utilities**
```python
# utils/validation_utils.py
def validate_human_web_arguments(args) -> None:
    """Validate human web game arguments."""
    if args.grid_size < 5 or args.grid_size > 50:
        raise ValueError("Grid size must be between 5 and 50")
    
    if args.port is not None and (args.port < 1024 or args.port > 65535):
        raise ValueError("Port must be between 1024 and 65535")

def validate_llm_web_arguments(args) -> None:
    """Validate LLM web game arguments."""
    validate_human_web_arguments(args)  # Reuse validation logic
    # Add LLM-specific validation if needed
```

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Design Patterns**: Comprehensive demonstration of OOP patterns
- **Architecture**: Layered inheritance and separation of concerns
- **Factory Patterns**: Universal creation patterns with canonical methods
- **Configuration**: Single source of truth principles
- **Debugging**: Integrated debug mode across full stack

### **Extension Development**
- **Template Reuse**: Copy structure for new extensions
- **Consistent Patterns**: Same architecture across all tasks
- **Minimal Effort**: Leverage existing infrastructure
- **Guaranteed Compatibility**: All extensions interoperate seamlessly

## 🔮 **Future Extensibility**

### **Extension Points**
```python
# Easy extension for new tasks
class SupervisedWebGameApp(SimpleFlaskApp):
    """Supervised learning web game application."""
    
    def _setup_routes(self) -> None:
        """Configure supervised learning routes."""
        self.app.add_url_rule('/', 'index', self._supervised_index)
        self.app.add_url_rule('/training', 'training', self._training_interface)
        self.app.add_url_rule('/api/predict', 'predict', self._model_prediction)

# Factory function for new extensions
def create_supervised_web_game_app(grid_size: int = 10, port: int | None = None) -> SupervisedWebGameApp:
    """Factory function for supervised learning web applications."""
    validate_supervised_web_arguments(grid_size, port)
    return SupervisedWebGameApp(grid_size=grid_size, port=port)
```

### **Integration Guidelines**
1. **Inherit from SimpleFlaskApp**: Use existing infrastructure
2. **Implement _setup_routes()**: Define task-specific routes
3. **Create factory function**: Follow canonical factory pattern
4. **Add validation**: Use universal validation utilities
5. **Follow naming conventions**: Use domain-indicated naming

## 📋 **Implementation Checklist**

### **For All Web Applications**
- [ ] **Inherit from SimpleFlaskApp**: Leverage existing infrastructure
- [ ] **Use factory functions**: Follow canonical creation patterns
- [ ] **Implement validation**: Use universal validation utilities
- [ ] **Follow naming conventions**: Use domain-indicated naming
- [ ] **Support debug mode**: Integrate with centralized debug configuration
- [ ] **Use random ports**: Leverage dynamic port allocation
- [ ] **Include simple logging**: Follow SUPREME_RULES for logging

### **For New Extensions**
- [ ] **Copy template structure**: Use existing scripts as templates
- [ ] **Create task-specific app class**: Inherit from SimpleFlaskApp
- [ ] **Implement factory function**: Follow canonical factory pattern
- [ ] **Add validation functions**: Use universal validation utilities
- [ ] **Update documentation**: Document extension-specific features
- [ ] **Test integration**: Ensure compatibility with existing infrastructure

---

**The MVC Flask Factory architecture demonstrates the power of careful design patterns and layered inheritance. By providing a sophisticated yet simple foundation, it enables rapid extension development while maintaining consistency and educational value across all AI approaches.**

## 🔗 **Cross-References**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for canonical patterns
- **`flask.md`**: Flask integration patterns for extensions
- **`network.md`**: Network architecture and port allocation
- **`kiss.md`**: KISS principles and simple design

### **Implementation Files**
- **`web/base_app.py`**: Abstract base classes and layered architecture
- **`web/applications.py`**: Task-specific Flask applications
- **`web/factories.py`**: Universal factory functions
- **`utils/factory_utils.py`**: Universal factory utilities
- **`utils/network_utils.py`**: Dynamic port allocation
- **`config/web_constants.py`**: Flask/JS/HTML/CSS constants
- **`config/network_constants.py`**: Network/host/port constants

### **Educational Resources**
- **Design Patterns**: Template Method, Factory, Strategy, Facade patterns
- **OOP Principles**: Inheritance, abstraction, encapsulation, polymorphism
- **Architecture**: Layered design and separation of concerns
- **Configuration**: Single source of truth and centralized management

