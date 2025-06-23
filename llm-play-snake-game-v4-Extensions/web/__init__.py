"""
Web Interface Framework for Snake-GTP
--------------------

Modern MVC web architecture for Snake game modes with role-based controllers.
Provides clean separation of concerns between models, views, and controllers.

Design Patterns Used:
    - Model-View-Controller (MVC): Core architectural pattern
    - Observer Pattern: For model-view communication  
    - Template Method Pattern: In base controllers
    - Strategy Pattern: For different rendering modes
    - Factory Pattern: For controller and view creation

Architecture Overview:
    Controllers/    - Request handling and business logic coordination
    Models/         - Data management and business rules
    Views/          - Presentation layer and template rendering
    
Educational Goals:
    - Demonstrate clean MVC architecture principles
    - Show role-based inheritance patterns
    - Illustrate separation of concerns
    - Provide extensible web framework for future tasks
"""

# Import MVC components for easy access
from .controllers import (
    BaseWebController,
    HumanGameController,
    RequestType,
    RequestContext
)

from .models import (
    GameStateModel,
    StateProvider,
    GameEvent,
    Observer
)

from .views import (
    WebViewRenderer
)

from .factories import (
    ControllerFactory,
    ModelFactory,
    ViewRendererFactory
)

__all__ = [
    # Controllers
    'BaseWebController',
    'HumanGameController',
    'RequestType',
    'RequestContext',
    
    # Models
    'GameStateModel',
    'StateProvider',
    'GameEvent',
    'Observer',
    
    # Views
    'WebViewRenderer',
    
    # Factories
    'ControllerFactory',
    'ModelFactory', 
    'ViewRendererFactory'
] 