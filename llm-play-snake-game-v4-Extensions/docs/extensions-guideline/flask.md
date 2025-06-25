# Flask Integration for Extensions

> **Important — Authoritative Reference:** This document supplements the `unified-streamlit-architecture-guide.md` and follows the web architecture established in `ROOT/web/`.

## 🎯 **Flask Integration Philosophy**

Extensions that require Flask integration should leverage the existing web infrastructure in `ROOT/web/` to maintain consistency with Task-0's web capabilities.

### **Flask Integration Overview**

Flask serves as the web framework for:
- **Real-time game visualization** through web browsers
- **RESTful API endpoints** for game control and state access
- **Multi-mode support** (live games, replay, human play)
- **Cross-platform compatibility** without native app requirements
- **Remote access** to running experiments and training sessions

### **Web Architecture Alignment**
Extensions should follow the same MVC patterns established in `ROOT/web/`:
- **Models**: `ROOT/web/models/` - Game state models and data structures
- **Views**: `ROOT/web/views/` - Template rendering and response formatting  
- **Controllers**: `ROOT/web/controllers/` - Request handling and business logic
- **Templates**: `ROOT/web/templates/` - HTML templates with consistent styling
- **Static Assets**: `ROOT/web/static/` - CSS, JavaScript, and other assets

### **Extension Flask Integration Pattern**
```python
# Extension Flask apps should extend the base web infrastructure
from web.controllers.base_controller import BaseController
from web.views.template_engines import render_template
from web.models.game_state_model import GameStateModel

class ExtensionController(BaseController):
    """Extension-specific Flask controller following ROOT/web patterns"""
    pass
```

## 🔧 **Implementation Guidelines**

### **Reuse ROOT/web Infrastructure**
- **Base Controllers**: Extend `web.controllers.base_controller.BaseController`
- **Template System**: Use `web.views.template_engines` for consistent rendering
- **Static Assets**: Leverage `web.static/` for consistent styling
- **Game State Models**: Use `web.models.game_state_model` for data consistency

### **Extension-Specific Additions**
Extensions may add their own:
- **Custom routes** for algorithm-specific functionality
- **Specialized templates** while maintaining consistent styling
- **Additional static assets** following the established patterns
- **Custom data models** that extend the base game state model

### **Key Flask Components for Extensions**
- **Web Controllers**: Handle HTTP requests and route to appropriate handlers
- **Template Engine**: Render dynamic HTML with game state
- **API Endpoints**: Provide JSON responses for AJAX requests
- **WebSocket Support**: Real-time updates for live game streaming
- **Static Assets**: CSS, JavaScript, and image resources

---

**This approach ensures Flask-based extensions maintain consistency with Task-0's web architecture while enabling algorithm-specific web functionality.**