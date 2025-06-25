# Flask Web Integration

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Flask integration must follow the established MVC patterns from `ROOT/web/` folder.

## 🎯 **Flask Integration Philosophy**

Flask serves as the web framework for providing browser-based interfaces across all Snake Game AI tasks and extensions. It leverages the existing MVC architecture established in `ROOT/web/` to ensure consistency and maintainability.

### **Core Design Principles**
- **MVC Architecture**: Following the patterns established in `ROOT/web/`
- **Extension Consistency**: Same web patterns across all algorithm types
- **API-First Design**: RESTful endpoints for programmatic access
- **Real-time Visualization**: WebSocket support for live game streaming

## 🏗️ **Web Architecture Integration**

### **Leveraging ROOT/web Infrastructure**
All extensions should reuse the foundational web components:

```python
# Extensions leverage existing web infrastructure
from web.controllers.base_controller import BaseController
from web.models.game_state_model import GameStateModel
from web.views.template_engines import TemplateEngine
from web.factories import create_web_app
```

### **MVC Pattern Consistency**
Following the established `ROOT/web/` structure:
- **Controllers**: Handle HTTP requests and route to appropriate handlers
- **Models**: Manage game state and data representation
- **Views**: Render templates and manage presentation logic
- **Factories**: Create configured web application instances

### **Extension-Specific Web Components**
Extensions extend base web functionality:

```python
# Example: Heuristic-specific web controller
class HeuristicWebController(BaseController):
    """Web controller for heuristic algorithm visualization"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_factory = HeuristicAgentFactory()
    
    def visualize_algorithm(self, algorithm_name: str):
        """Render algorithm-specific visualization"""
        agent = self.algorithm_factory.create_agent(algorithm_name)
        return self.render_template('heuristic_viz.html', agent=agent)
```

## 🌐 **Multi-Mode Web Support**

### **Replay Mode Integration**
Flask provides web-based replay capabilities:
- **Game History Visualization**: Step-through game progression
- **Algorithm Decision Display**: Show reasoning at each step
- **Performance Metrics**: Real-time statistics and comparisons
- **Export Capabilities**: Save visualizations and data

### **Live Game Streaming**
Real-time game visualization through WebSockets:
- **Algorithm Execution**: Watch algorithms make decisions
- **Training Progress**: Monitor ML model training
- **Comparative Views**: Side-by-side algorithm comparison
- **Interactive Controls**: Pause, resume, speed adjustment

### **API Endpoints**
RESTful API for programmatic access:
```python
# Standard API patterns for all extensions
@app.route('/api/v1/algorithms', methods=['GET'])
def list_algorithms():
    """Return available algorithms for this extension"""
    
@app.route('/api/v1/game/<game_id>/state', methods=['GET'])
def get_game_state(game_id):
    """Return current game state"""
    
@app.route('/api/v1/replay/<replay_id>', methods=['POST'])
def start_replay(replay_id):
    """Start game replay session"""
```

## 🚀 **Extension Implementation Guidelines**

### **Path Management Integration**
Following Final Decision 6:
```python
from extensions.common.path_utils import ensure_project_root

# Flask apps must use proper path management
def create_extension_web_app():
    ensure_project_root()
    app = create_web_app('extension_config')
    return app
```

### **Template Inheritance**
Leverage existing template hierarchy:
```html
<!-- Extend base templates from ROOT/web/templates/ -->
{% extends "base.html" %}
{% block content %}
    <!-- Extension-specific content -->
{% endblock %}
```

### **Static Asset Management**
Consistent asset organization:
```
web/static/
├── css/
│   ├── style.css          # Base styles (from ROOT/web)
│   └── extension.css      # Extension-specific styles
├── js/
│   ├── common.js          # Base JavaScript (from ROOT/web)
│   └── algorithm_viz.js   # Algorithm visualization
└── assets/
    └── extension_images/   # Extension-specific images
```

## 🎓 **Educational and Development Benefits**

### **Consistent User Experience**
- **Familiar Interface**: Same UI patterns across all extensions
- **Predictable Navigation**: Consistent menu and control structures
- **Shared Components**: Reusable visualization elements
- **Unified Theming**: Coherent visual design

### **Development Efficiency**
- **Code Reuse**: Leverage existing web infrastructure
- **Faster Development**: Focus on algorithm-specific features
- **Easier Maintenance**: Centralized web component updates
- **Standardized Patterns**: Well-established development practices

### **Cross-Extension Integration**
- **Unified Dashboard**: Single interface for all algorithm types
- **Comparative Tools**: Side-by-side algorithm comparison
- **Data Export**: Consistent export formats and APIs
- **Plugin Architecture**: Easy addition of new visualization modes

## 🔧 **Implementation Patterns**

### **Factory Pattern for Web Apps**
```python
class ExtensionWebFactory:
    """Factory for creating extension-specific web applications"""
    
    @staticmethod
    def create_app(extension_type: str, config: dict):
        """Create configured Flask app for extension type"""
        base_app = create_web_app(config)
        extension_blueprint = ExtensionBlueprintFactory.create(extension_type)
        base_app.register_blueprint(extension_blueprint)
        return base_app
```

### **Blueprint Organization**
```python
# Extension-specific blueprints
heuristic_bp = Blueprint('heuristics', __name__, url_prefix='/heuristics')
supervised_bp = Blueprint('supervised', __name__, url_prefix='/supervised')
rl_bp = Blueprint('reinforcement', __name__, url_prefix='/rl')
```

---

**Flask integration provides a powerful, consistent web interface across all Snake Game AI extensions. By leveraging the established MVC architecture from `ROOT/web/`, extensions maintain consistency while providing algorithm-specific visualization and interaction capabilities. This approach ensures a unified user experience while supporting the diverse needs of different AI approaches.**