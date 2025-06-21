"""
MVC Component Factories
======================

Factory classes for creating MVC components with proper configuration
and dependency injection. Implements Factory pattern to centralize
object creation and configuration.

Design Patterns:
    - Factory Pattern: Centralized object creation
    - Builder Pattern: Complex object construction
    - Dependency Injection: Automatic dependency resolution
    - Strategy Pattern: Different creation strategies

Educational Goals:
    - Demonstrate Factory pattern in practice
    - Show dependency injection implementation
    - Illustrate how factories simplify object creation
    - Provide flexible component configuration
"""

from typing import Dict, Any, Type, Optional, Union
from flask import Flask
import logging

# Import MVC components
from .controllers import (
    BaseWebController, LLMGameController, HumanGameController, 
    ReplayController, GamePlayController, GameViewingController
)
from .models import (
    GameStateModel, StateProvider, LiveGameStateProvider, 
    ReplayStateProvider
)
from .views import WebViewRenderer, TemplateEngine, JinjaTemplateEngine

# Import core components
from core.game_controller import GameController
from replay.replay_engine import ReplayEngine

logger = logging.getLogger(__name__)


class ControllerFactory:
    """
    Factory for creating web controllers with proper configuration.
    
    Handles dependency injection and controller-specific setup.
    Uses Strategy pattern to create different controller types.
    
    Design Pattern: Factory + Strategy
    - Centralizes controller creation logic
    - Handles dependency injection automatically
    - Provides type-safe controller creation
    """
    
    def __init__(self):
        """Initialize controller factory with configuration."""
        self._controller_registry: Dict[str, Type[BaseWebController]] = {
            'llm_game': LLMGameController,
            'human_game': HumanGameController,
            'replay': ReplayController,
        }
        self._default_config = {
            'enable_rate_limiting': True,
            'max_requests_per_minute': 120,
            'enable_logging': True,
            'enable_performance_tracking': True
        }
    
    def create_controller(self, controller_type: str, 
                         model_manager: GameStateModel,
                         view_renderer: WebViewRenderer,
                         **kwargs) -> BaseWebController:
        """
        Create controller instance with dependency injection.
        
        Args:
            controller_type: Type of controller to create
            model_manager: Game state model instance
            view_renderer: View renderer instance
            **kwargs: Additional configuration options
            
        Returns:
            Configured controller instance
            
        Raises:
            ValueError: If controller type is not registered
        """
        if controller_type not in self._controller_registry:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        controller_class = self._controller_registry[controller_type]
        
        # Merge default config with provided kwargs
        config = {**self._default_config, **kwargs}
        
        try:
            # Create controller with dependency injection
            controller = controller_class(
                model_manager=model_manager,
                view_renderer=view_renderer,
                **config
            )
            
            logger.info(f"Created {controller_type} controller: {controller.__class__.__name__}")
            return controller
            
        except Exception as e:
            logger.error(f"Failed to create {controller_type} controller: {e}")
            raise
    
    def register_controller_type(self, name: str, controller_class: Type[BaseWebController]):
        """Register new controller type."""
        self._controller_registry[name] = controller_class
        logger.debug(f"Registered controller type: {name}")
    
    def get_available_types(self) -> list:
        """Get list of available controller types."""
        return list(self._controller_registry.keys())


class ModelFactory:
    """
    Factory for creating model components with state providers.
    
    Handles creation of GameStateModel with appropriate StateProvider
    based on the game mode and configuration.
    """
    
    def create_live_game_model(self, game_controller: GameController, 
                              game_mode: str = "human") -> GameStateModel:
        """
        Create model for live game sessions.
        
        Args:
            game_controller: Active game controller
            game_mode: Type of live game (human/llm)
            
        Returns:
            Configured GameStateModel with LiveGameStateProvider
        """
        from .models.game_state_model import GameMode, LiveGameStateProvider
        
        # Map string mode to enum
        mode_mapping = {
            "human": GameMode.LIVE_HUMAN,
            "llm": GameMode.LIVE_LLM,
            "demo": GameMode.DEMO
        }
        
        game_mode_enum = mode_mapping.get(game_mode, GameMode.LIVE_HUMAN)
        
        # Create state provider
        state_provider = LiveGameStateProvider(game_controller, game_mode_enum)
        
        # Create model with state provider
        model = GameStateModel(state_provider)
        
        logger.info(f"Created live game model for mode: {game_mode}")
        return model
    
    def create_replay_model(self, replay_engine: ReplayEngine) -> GameStateModel:
        """
        Create model for replay sessions.
        
        Args:
            replay_engine: Active replay engine
            
        Returns:
            Configured GameStateModel with ReplayStateProvider
        """
        # Create state provider
        state_provider = ReplayStateProvider(replay_engine)
        
        # Create model with state provider
        model = GameStateModel(state_provider)
        
        logger.info("Created replay game model")
        return model


class ViewRendererFactory:
    """
    Factory for creating view renderers with template engines.
    
    Handles creation and configuration of WebViewRenderer with
    appropriate template engines and decorators.
    """
    
    def create_renderer(self, template_folder: str, static_folder: str,
                       enable_caching: bool = True,
                       enable_compression: bool = True) -> WebViewRenderer:
        """
        Create view renderer with Jinja template engine.
        
        Args:
            template_folder: Path to template files
            static_folder: Path to static files
            enable_caching: Enable template caching
            enable_compression: Enable response compression
            
        Returns:
            Configured WebViewRenderer instance
        """
        # Create view renderer
        renderer = WebViewRenderer(template_folder, static_folder)
        
        # Add decorators based on configuration
        if enable_compression:
            from .views.decorators import CompressionDecorator
            renderer.add_decorator(CompressionDecorator())
        
        if enable_caching:
            from .views.decorators import CacheDecorator
            renderer.add_decorator(CacheDecorator(cache_timeout=300))
        
        logger.info("Created view renderer with Jinja template engine")
        return renderer


class WebApplicationFactory:
    """
    Factory for creating complete web applications.
    
    Orchestrates creation of all MVC components and wires them together
    into a functioning Flask application.
    
    Design Pattern: Builder + Factory
    - Handles complex application assembly
    - Provides different application configurations
    - Manages component lifecycle and dependencies
    """
    
    def __init__(self):
        """Initialize application factory."""
        self.controller_factory = ControllerFactory()
        self.model_factory = ModelFactory()
        self.view_factory = ViewRendererFactory()
    
    def create_live_game_app(self, game_controller: GameController,
                            game_mode: str = "human",
                            template_folder: str = "templates",
                            static_folder: str = "static",
                            **app_config) -> tuple[Flask, BaseWebController]:
        """
        Create Flask application for live game sessions.
        
        Args:
            game_controller: Active game controller
            game_mode: Type of live game (human/llm)
            template_folder: Template files location
            static_folder: Static files location
            **app_config: Additional Flask configuration
            
        Returns:
            Tuple of (Flask app, Controller instance)
        """
        # Create Flask app
        app = Flask(__name__, 
                   template_folder=template_folder,
                   static_folder=static_folder)
        app.config.update(app_config)
        
        # Create MVC components
        model = self.model_factory.create_live_game_model(game_controller, game_mode)
        view_renderer = self.view_factory.create_renderer(template_folder, static_folder)
        
        # Determine controller type
        controller_type = "llm_game" if game_mode == "llm" else "human_game"
        controller = self.controller_factory.create_controller(
            controller_type, model, view_renderer
        )
        
        # Register routes
        self._register_routes(app, controller)
        
        logger.info(f"Created live game application for {game_mode} mode")
        return app, controller
    
    def create_replay_app(self, replay_engine: ReplayEngine,
                         template_folder: str = "templates",
                         static_folder: str = "static",
                         **app_config) -> tuple[Flask, BaseWebController]:
        """
        Create Flask application for replay sessions.
        
        Args:
            replay_engine: Active replay engine
            template_folder: Template files location
            static_folder: Static files location
            **app_config: Additional Flask configuration
            
        Returns:
            Tuple of (Flask app, Controller instance)
        """
        # Create Flask app
        app = Flask(__name__,
                   template_folder=template_folder,
                   static_folder=static_folder)
        app.config.update(app_config)
        
        # Create MVC components
        model = self.model_factory.create_replay_model(replay_engine)
        view_renderer = self.view_factory.create_renderer(template_folder, static_folder)
        controller = self.controller_factory.create_controller(
            "replay", model, view_renderer
        )
        
        # Register routes
        self._register_routes(app, controller)
        
        logger.info("Created replay application")
        return app, controller
    
    def _register_routes(self, app: Flask, controller: BaseWebController):
        """
        Register Flask routes with controller methods.
        
        Args:
            app: Flask application instance
            controller: Controller to handle routes
        """
        from .controllers.base_controller import RequestType
        
        @app.route('/')
        def index():
            return controller.handle_request(
                request=flask.request, 
                request_type=RequestType.INDEX_GET
            )
        
        @app.route('/api/state')
        def api_state():
            return controller.handle_request(
                request=flask.request,
                request_type=RequestType.STATE_GET
            )
        
        @app.route('/api/control', methods=['POST'])
        def api_control():
            return controller.handle_request(
                request=flask.request,
                request_type=RequestType.CONTROL_POST
            )
        
        @app.route('/api/reset', methods=['POST'])
        def api_reset():
            return controller.handle_request(
                request=flask.request,
                request_type=RequestType.RESET_POST
            )
        
        @app.route('/api/health')
        def api_health():
            return controller.handle_request(
                request=flask.request,
                request_type=RequestType.HEALTH_GET
            )


# Convenience factory functions
def create_web_application(game_controller: GameController = None,
                          replay_engine: ReplayEngine = None,
                          game_mode: str = "human",
                          **config) -> tuple[Flask, BaseWebController]:
    """
    Convenience function to create web application.
    
    Args:
        game_controller: For live game modes
        replay_engine: For replay mode
        game_mode: Type of game mode
        **config: Application configuration
        
    Returns:
        Tuple of (Flask app, Controller instance)
    """
    factory = WebApplicationFactory()
    
    if replay_engine:
        return factory.create_replay_app(replay_engine, **config)
    elif game_controller:
        return factory.create_live_game_app(game_controller, game_mode, **config)
    else:
        raise ValueError("Either game_controller or replay_engine must be provided")


def create_controller_for_mode(mode: str, model: GameStateModel, 
                              view_renderer: WebViewRenderer) -> BaseWebController:
    """
    Convenience function to create controller for specific mode.
    
    Args:
        mode: Controller mode (human_game, llm_game, replay)
        model: Game state model
        view_renderer: View renderer
        
    Returns:
        Configured controller instance
    """
    factory = ControllerFactory()
    return factory.create_controller(mode, model, view_renderer) 