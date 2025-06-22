I'll analyze the current web implementation and provide 10 comprehensive OOP approaches to unify the web modes. Let me first examine the existing web files to understand the current structure.
Read file: scripts/main_web.py
Read file: scripts/replay_web.py
Read file: scripts/human_play_web.py
Now let me examine some additional web-related files to get a complete picture:
Read file: config/web_constants.py
Read file: utils/web_utils.py
Read file: project-structure-plan.md
Now let me examine a few more key files to understand the current architecture better:
Ran tool
Read file: web/templates/main.html
Now I have a comprehensive understanding of the current web implementation. Based on my analysis, here are **10 comprehensive OOP approaches** to unify the web modes into an elegant, coherent, and extensible architecture:

---

## **Approach 1: Abstract Web Application Factory Pattern**

**Core Philosophy:** Create a factory-based architecture where each web mode is a concrete implementation of an abstract web application that follows the Factory Method pattern.

**Implementation Ideas:**

**Base Architecture:**
- `BaseWebApplication` abstract class defining the core web application lifecycle
- `WebApplicationFactory` that creates appropriate web app instances based on mode
- Common interface for all web modes through the Template Method pattern

**Key Classes:**
```python
class BaseWebApplication(ABC):
    """Abstract base for all web applications following Template Method pattern"""
    
    def __init__(self, flask_app: Flask, config: WebConfig):
        self.app = flask_app
        self.config = config
        self._setup_common_routes()
        self._setup_middleware()
    
    @abstractmethod
    def create_controller(self) -> BaseController:
        """Factory method for creating mode-specific controllers"""
        pass
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Template file name for this mode"""
        pass
    
    def setup_routes(self):
        """Template method for route setup - calls hook methods"""
        self._register_common_routes()
        self._register_mode_specific_routes()
        self._register_api_routes()
    
    @abstractmethod
    def _register_mode_specific_routes(self):
        """Hook method for mode-specific routes"""
        pass
```

**Concrete Implementations:**
- `LiveGameWebApplication` (replaces main_web.py)
- `ReplayWebApplication` (replaces replay_web.py) 
- `HumanPlayWebApplication` (replaces human_play_web.py)

**Benefits:**
- Eliminates code duplication through template method pattern
- Easy to add new web modes by extending base class
- Consistent API structure across all modes
- Factory pattern provides loose coupling between client code and concrete implementations

**Design Patterns Used:**
- **Factory Method Pattern:** For creating mode-specific controllers
- **Template Method Pattern:** For defining common web app lifecycle
- **Strategy Pattern:** For different rendering/control strategies per mode

---

## **Approach 2: Component-Based Web Framework with Composition**

**Core Philosophy:** Build a component-based web framework where each web mode is composed of reusable, interchangeable components following the Composite pattern.

**Implementation Ideas:**

**Component Architecture:**
- `WebComponent` base class with lifecycle methods (render, update, handle_request)
- `WebComponentContainer` that manages collections of components
- `WebPageBuilder` that assembles components into complete pages

**Key Components:**
```python
class WebComponent(ABC):
    """Base class for all web components"""
    
    def __init__(self, component_id: str, config: dict = None):
        self.component_id = component_id
        self.config = config or {}
        self.children: List[WebComponent] = []
    
    @abstractmethod
    def render(self, data: dict) -> str:
        """Render component to HTML"""
        pass
    
    @abstractmethod
    def get_api_routes(self) -> List[tuple]:
        """Return list of (endpoint, handler) tuples"""
        pass
    
    def add_child(self, child: 'WebComponent'):
        """Composite pattern - add child component"""
        self.children.append(child)
```

**Specific Components:**
- `GameBoardComponent` - Renders game canvas and handles board display
- `GameControlsComponent` - Manages play/pause/reset controls
- `GameStatsComponent` - Displays score, steps, timing information
- `LLMResponseComponent` - Shows LLM reasoning and planned moves
- `ReplayControlsComponent` - Replay-specific navigation controls
- `HumanInputComponent` - Keyboard input handling for human play

**Page Assemblies:**
```python
class LiveGamePage(WebPageBuilder):
    def build(self) -> WebPage:
        return (WebPage("live_game")
            .add_component(GameBoardComponent())
            .add_component(GameControlsComponent(show_pause=True))
            .add_component(GameStatsComponent())
            .add_component(LLMResponseComponent()))
```

**Benefits:**
- Maximum reusability - components can be mixed and matched
- Easy testing - each component can be unit tested independently
- Consistent UI/UX across modes through shared components
- Easy to add new functionality by creating new components

**Design Patterns Used:**
- **Composite Pattern:** For hierarchical component structure
- **Builder Pattern:** For assembling complex page layouts
- **Command Pattern:** For component interactions and event handling

---

## **Approach 3: MVC Architecture with Role-Based Controllers**

**Core Philosophy:** Implement a clean Model-View-Controller architecture where different web modes are handled by specialized controllers that inherit from role-based base controllers.

**Implementation Ideas:**

**Controller Hierarchy:**
```python
class BaseWebController(ABC):
    """Abstract base controller with common web functionality"""
    
    def __init__(self, model_manager: ModelManager, view_renderer: ViewRenderer):
        self.model_manager = model_manager
        self.view_renderer = view_renderer
        self.request_filters: List[RequestFilter] = []
    
    @abstractmethod
    def handle_state_request(self) -> dict:
        """Handle /api/state requests"""
        pass
    
    @abstractmethod
    def handle_control_request(self, command: str) -> dict:
        """Handle /api/control requests"""
        pass

class GamePlayController(BaseWebController):
    """Base for controllers that manage active gameplay"""
    
    def __init__(self, game_engine: BaseGameEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_engine = game_engine
    
    def handle_move_request(self, move_data: dict) -> dict:
        """Common move handling logic"""
        pass

class GameViewingController(BaseWebController):
    """Base for controllers that display/replay games"""
    
    def __init__(self, data_source: DataSource, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_source = data_source
    
    def handle_navigation_request(self, nav_command: str) -> dict:
        """Common navigation logic"""
        pass
```

**Concrete Controllers:**
- `LLMGameController(GamePlayController)` - Manages LLM-driven gameplay
- `HumanGameController(GamePlayController)` - Handles human input
- `ReplayController(GameViewingController)` - Manages game replay

**Model Layer:**
```python
class GameStateModel:
    """Encapsulates game state and business logic"""
    
    def __init__(self, state_provider: StateProvider):
        self.state_provider = state_provider
        self.observers: List[Observer] = []
    
    def get_current_state(self) -> GameState:
        """Get current game state"""
        pass
    
    def notify_observers(self, event: GameEvent):
        """Observer pattern for state changes"""
        for observer in self.observers:
            observer.on_game_event(event)
```

**View Layer:**
```python
class WebViewRenderer:
    """Handles view rendering with template selection"""
    
    def __init__(self, template_engine: TemplateEngine):
        self.template_engine = template_engine
        self.view_decorators: List[ViewDecorator] = []
    
    def render_page(self, template_name: str, context: dict) -> str:
        """Render complete page with decorators"""
        pass
    
    def render_api_response(self, data: dict) -> str:
        """Render JSON API response"""
        pass
```

**Benefits:**
- Clear separation of concerns following MVC principles
- Role-based inheritance promotes code reuse while maintaining flexibility
- Easy to test each layer independently
- Observer pattern enables loose coupling between model and view

**Design Patterns Used:**
- **Model-View-Controller (MVC):** Core architectural pattern
- **Observer Pattern:** For model-view communication
- **Template Method Pattern:** In base controllers
- **Decorator Pattern:** For view enhancement

---

## **Approach 4: State Machine-Driven Web Applications**

**Core Philosophy:** Model each web mode as a state machine where the application behavior changes based on its current state, providing a clean way to handle different phases of gameplay, replay, and human interaction.

**Implementation Ideas:**

**State Machine Architecture:**
```python
class WebApplicationState(ABC):
    """Abstract base state for web application state machine"""
    
    def __init__(self, context: 'WebApplicationContext'):
        self.context = context
    
    @abstractmethod
    def handle_state_request(self) -> dict:
        """Handle state API requests in this state"""
        pass
    
    @abstractmethod
    def handle_control_request(self, command: str) -> dict:
        """Handle control API requests in this state"""
        pass
    
    @abstractmethod
    def get_valid_transitions(self) -> List[str]:
        """Return list of valid state transitions"""
        pass
    
    def transition_to(self, new_state_name: str):
        """Transition to new state if valid"""
        if new_state_name in self.get_valid_transitions():
            self.context.transition_to_state(new_state_name)
```

**State Implementations for Live Game Mode:**
```python
class GameInitializingState(WebApplicationState):
    """Game is being set up"""
    
    def handle_state_request(self) -> dict:
        return {"status": "initializing", "progress": self.context.init_progress}
    
    def get_valid_transitions(self) -> List[str]:
        return ["game_running", "game_error"]

class GameRunningState(WebApplicationState):
    """Game is actively running"""
    
    def handle_control_request(self, command: str) -> dict:
        if command == "pause":
            self.transition_to("game_paused")
            return {"status": "paused"}
        # ... other command handling

class GamePausedState(WebApplicationState):
    """Game is paused"""
    
    def handle_control_request(self, command: str) -> dict:
        if command == "resume":
            self.transition_to("game_running")
            return {"status": "resumed"}
```

**State Implementations for Replay Mode:**
```python
class ReplayLoadingState(WebApplicationState):
    """Loading replay data"""
    
class ReplayPlayingState(WebApplicationState):
    """Replay is playing"""
    
class ReplayPausedState(WebApplicationState):
    """Replay is paused"""
    
class ReplayNavigatingState(WebApplicationState):
    """User is navigating through replay"""
```

**Context Manager:**
```python
class WebApplicationContext:
    """Context object that manages state transitions"""
    
    def __init__(self, initial_state_name: str):
        self.states: Dict[str, WebApplicationState] = {}
        self.current_state_name = initial_state_name
        self.state_history: List[str] = []
        self.event_bus = EventBus()
    
    def register_state(self, name: str, state: WebApplicationState):
        """Register a state with the context"""
        self.states[name] = state
    
    def transition_to_state(self, new_state_name: str):
        """Perform state transition with validation"""
        if new_state_name in self.states:
            old_state = self.current_state_name
            self.current_state_name = new_state_name
            self.state_history.append(old_state)
            self.event_bus.publish(StateTransitionEvent(old_state, new_state_name))
```

**Benefits:**
- Clear modeling of application behavior in different phases
- Easy to add new states and transitions
- Built-in validation of state transitions
- Excellent for debugging - can see exact state history
- Natural way to handle async operations (loading, network requests)

**Design Patterns Used:**
- **State Pattern:** Core architecture for behavior changes
- **Context Pattern:** For managing state transitions
- **Observer Pattern:** For state change notifications
- **Command Pattern:** For handling user inputs in different states

---

## **Approach 5: Plugin Architecture with Extension Points**

**Core Philosophy:** Create a plugin-based architecture where each web mode is a plugin that extends core functionality through well-defined extension points, allowing for maximum flexibility and modularity.

**Implementation Ideas:**

**Core Plugin System:**
```python
class WebPlugin(ABC):
    """Abstract base class for web plugins"""
    
    def __init__(self, plugin_name: str, version: str):
        self.plugin_name = plugin_name
        self.version = version
        self.dependencies: List[str] = []
        self.extension_points: Dict[str, ExtensionPoint] = {}
    
    @abstractmethod
    def initialize(self, plugin_context: PluginContext) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def get_routes(self) -> List[Route]:
        """Return routes provided by this plugin"""
        pass
    
    @abstractmethod
    def get_template_extensions(self) -> Dict[str, str]:
        """Return template modifications/additions"""
        pass
    
    def register_extension_point(self, name: str, extension_point: ExtensionPoint):
        """Register an extension point"""
        self.extension_points[name] = extension_point
```

**Extension Points:**
```python
class ExtensionPoint:
    """Defines a point where plugins can extend functionality"""
    
    def __init__(self, name: str, description: str, contract: type):
        self.name = name
        self.description = description
        self.contract = contract  # Interface that extensions must implement
        self.extensions: List[Extension] = []
    
    def add_extension(self, extension: Extension):
        """Add an extension to this point"""
        if isinstance(extension, self.contract):
            self.extensions.append(extension)
        else:
            raise ValueError(f"Extension must implement {self.contract}")
    
    def execute_extensions(self, context: dict) -> List[Any]:
        """Execute all extensions at this point"""
        return [ext.execute(context) for ext in self.extensions]
```

**Core Extension Points:**
- `WebRouteExtension` - Add new API endpoints
- `TemplateExtension` - Modify or add new templates
- `JavaScriptExtension` - Add client-side functionality
- `StateProviderExtension` - Provide game state data
- `ControlHandlerExtension` - Handle control commands
- `AuthenticationExtension` - Add authentication layers

**Concrete Plugin Implementations:**
```python
class LiveGamePlugin(WebPlugin):
    """Plugin for LLM-driven live gameplay"""
    
    def initialize(self, plugin_context: PluginContext) -> bool:
        # Register state provider
        state_ext = LLMGameStateExtension()
        plugin_context.register_extension("state_provider", state_ext)
        
        # Register control handlers
        control_ext = LLMGameControlExtension()
        plugin_context.register_extension("control_handler", control_ext)
        
        return True
    
    def get_routes(self) -> List[Route]:
        return [
            Route("/", "GET", self.render_live_game_page),
            Route("/api/llm/state", "GET", self.get_llm_state),
            Route("/api/llm/control", "POST", self.handle_llm_control),
        ]

class ReplayPlugin(WebPlugin):
    """Plugin for game replay functionality"""
    
    def initialize(self, plugin_context: PluginContext) -> bool:
        # Register replay-specific extensions
        replay_state_ext = ReplayStateExtension()
        plugin_context.register_extension("state_provider", replay_state_ext)
        
        replay_nav_ext = ReplayNavigationExtension()
        plugin_context.register_extension("navigation_handler", replay_nav_ext)
        
        return True
```

**Plugin Manager:**
```python
class WebPluginManager:
    """Manages web plugins and their lifecycle"""
    
    def __init__(self):
        self.plugins: Dict[str, WebPlugin] = {}
        self.extension_points: Dict[str, ExtensionPoint] = {}
        self.plugin_context = PluginContext(self)
    
    def register_plugin(self, plugin: WebPlugin) -> bool:
        """Register and initialize a plugin"""
        if self._check_dependencies(plugin):
            if plugin.initialize(self.plugin_context):
                self.plugins[plugin.plugin_name] = plugin
                return True
        return False
    
    def create_flask_app(self) -> Flask:
        """Create Flask app with all plugin routes and extensions"""
        app = Flask(__name__)
        
        # Register routes from all plugins
        for plugin in self.plugins.values():
            for route in plugin.get_routes():
                app.route(route.path, methods=[route.method])(route.handler)
        
        return app
```

**Benefits:**
- Maximum modularity - each mode is completely independent
- Easy to disable/enable specific modes
- Third-party extensibility - others can create plugins
- Clear contracts through extension points
- Version management and dependency resolution

**Design Patterns Used:**
- **Plugin Pattern:** Core architecture
- **Extension Point Pattern:** For allowing modifications
- **Dependency Injection:** Through plugin context
- **Registry Pattern:** For managing plugins and extensions

---

## **Approach 6: Event-Driven Architecture with Message Bus**

**Core Philosophy:** Build an event-driven system where web modes communicate through a central message bus, promoting loose coupling and enabling real-time features across all modes.

**Implementation Ideas:**

**Event Bus Architecture:**
```python
class Event:
    """Base class for all events in the system"""
    
    def __init__(self, event_type: str, source: str, timestamp: float = None):
        self.event_type = event_type
        self.source = source
        self.timestamp = timestamp or time.time()
        self.data: Dict[str, Any] = {}
    
    def with_data(self, **kwargs) -> 'Event':
        """Fluent interface for adding event data"""
        self.data.update(kwargs)
        return self

class EventBus:
    """Central message bus for event communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.middleware: List[EventMiddleware] = []
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe to events of a specific type"""
        self.subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        """Publish an event to all subscribers"""
        # Apply middleware
        for middleware in self.middleware:
            event = middleware.process(event)
            if event is None:  # Middleware can filter events
                return
        
        # Store in history
        self.event_history.append(event)
        
        # Notify subscribers
        for handler in self.subscribers[event.event_type]:
            try:
                handler.handle(event)
            except Exception as e:
                error_event = Event("event_handler_error", "event_bus")
                error_event.with_data(original_event=event, error=str(e))
                self.publish(error_event)
```

**Event Types:**
```python
class GameEvents:
    GAME_STARTED = "game.started"
    GAME_ENDED = "game.ended"
    MOVE_MADE = "game.move_made"
    SCORE_CHANGED = "game.score_changed"
    STATE_CHANGED = "game.state_changed"

class ReplayEvents:
    REPLAY_LOADED = "replay.loaded"
    REPLAY_POSITION_CHANGED = "replay.position_changed"
    REPLAY_SPEED_CHANGED = "replay.speed_changed"

class WebEvents:
    CLIENT_CONNECTED = "web.client_connected"
    CLIENT_DISCONNECTED = "web.client_disconnected"
    API_REQUEST_RECEIVED = "web.api_request"
    WEBSOCKET_MESSAGE = "web.websocket_message"
```

**Web Mode Event Handlers:**
```python
class LiveGameEventHandler(EventHandler):
    """Handles events for live game mode"""
    
    def __init__(self, game_manager: GameManager, websocket_manager: WebSocketManager):
        self.game_manager = game_manager
        self.websocket_manager = websocket_manager
    
    def handle(self, event: Event):
        if event.event_type == GameEvents.STATE_CHANGED:
            # Broadcast state to all connected clients
            state_data = event.data.get("state")
            self.websocket_manager.broadcast("game_state_update", state_data)
        
        elif event.event_type == GameEvents.GAME_ENDED:
            # Handle game end logic
            self._handle_game_end(event.data)

class ReplayEventHandler(EventHandler):
    """Handles events for replay mode"""
    
    def __init__(self, replay_engine: ReplayEngine, websocket_manager: WebSocketManager):
        self.replay_engine = replay_engine
        self.websocket_manager = websocket_manager
    
    def handle(self, event: Event):
        if event.event_type == ReplayEvents.REPLAY_POSITION_CHANGED:
            # Update replay position for all viewers
            position_data = event.data.get("position")
            self.websocket_manager.broadcast("replay_position_update", position_data)
```

**WebSocket Integration:**
```python
class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.connections: Dict[str, WebSocketConnection] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
    
    def add_connection(self, connection_id: str, websocket: WebSocket):
        """Add a new WebSocket connection"""
        self.connections[connection_id] = WebSocketConnection(websocket)
        
        # Publish connection event
        event = Event(WebEvents.CLIENT_CONNECTED, "websocket_manager")
        event.with_data(connection_id=connection_id)
        self.event_bus.publish(event)
    
    def broadcast(self, message_type: str, data: dict, room: str = None):
        """Broadcast message to connections"""
        message = {"type": message_type, "data": data}
        
        connections_to_notify = (
            [self.connections[conn_id] for conn_id in self.rooms[room]]
            if room
            else self.connections.values()
        )
        
        for connection in connections_to_notify:
            connection.send(json.dumps(message))
```

**Web Application Integration:**
```python
class EventDrivenWebApp:
    """Web application built around event bus"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.websocket_manager = WebSocketManager(self.event_bus)
        self.flask_app = Flask(__name__)
        self._setup_routes()
    
    def register_mode_handler(self, handler: EventHandler, event_types: List[str]):
        """Register an event handler for specific event types"""
        for event_type in event_types:
            self.event_bus.subscribe(event_type, handler)
    
    @app.route('/api/state')
    def get_state(self):
        # Publish state request event
        event = Event(WebEvents.API_REQUEST_RECEIVED, "flask_app")
        event.with_data(endpoint="/api/state", method="GET")
        self.event_bus.publish(event)
        
        # Return current state (handler will have updated it)
        return jsonify(self.current_state)
```

**Benefits:**
- True real-time updates across all clients
- Loose coupling between web modes
- Easy to add new event types and handlers
- Built-in audit trail through event history
- Natural scaling for multiple concurrent users

**Design Patterns Used:**  
- **Observer Pattern:** Through event subscription
- **Mediator Pattern:** Event bus as central mediator
- **Command Pattern:** Events as commands
- **Publish-Subscribe Pattern:** Core communication mechanism

---

## **Approach 7: Microservices Architecture with API Gateway**

**Core Philosophy:** Split each web mode into independent microservices that communicate through well-defined APIs, with a central API gateway that routes requests and provides a unified interface.

**Implementation Ideas:**

**Service Architecture:**
```python
class BaseWebService(ABC):
    """Abstract base class for web microservices"""
    
    def __init__(self, service_name: str, port: int, config: ServiceConfig):
        self.service_name = service_name
        self.port = port
        self.config = config
        self.flask_app = Flask(service_name)
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self._setup_common_routes()
    
    @abstractmethod
    def setup_service_routes(self):
        """Setup service-specific routes"""
        pass
    
    @abstractmethod
    def get_service_info(self) -> ServiceInfo:
        """Return information about this service"""
        pass
    
    def _setup_common_routes(self):
        """Setup common routes for all services"""
        @self.flask_app.route('/health')
        def health():
            return jsonify(self.health_checker.get_status())
        
        @self.flask_app.route('/metrics')
        def metrics():
            return jsonify(self.metrics_collector.get_metrics())
    
    def start(self):
        """Start the service"""
        self.setup_service_routes()
        self.flask_app.run(host='0.0.0.0', port=self.port)
```

**Individual Services:**
```python
class LiveGameService(BaseWebService):
    """Microservice for live LLM gameplay"""
    
    def __init__(self, port: int = 8001):
        super().__init__("live-game-service", port, LiveGameConfig())
        self.game_manager = GameManager()
        self.llm_agent = LLMAgent()
    
    def setup_service_routes(self):
        @self.flask_app.route('/api/game/state', methods=['GET'])
        def get_game_state():
            state = self.game_manager.get_current_state()
            return jsonify(self._serialize_state(state))
        
        @self.flask_app.route('/api/game/control', methods=['POST'])
        def control_game():
            command = request.json.get('command')
            result = self.game_manager.handle_command(command)
            return jsonify(result)
        
        @self.flask_app.route('/api/llm/response', methods=['GET'])
        def get_llm_response():
            return jsonify(self.llm_agent.get_last_response())

class ReplayService(BaseWebService):
    """Microservice for game replay"""
    
    def __init__(self, port: int = 8002):
        super().__init__("replay-service", port, ReplayConfig())
        self.replay_engine = ReplayEngine()
    
    def setup_service_routes(self):
        @self.flask_app.route('/api/replay/load', methods=['POST'])
        def load_replay():
            replay_path = request.json.get('path')
            success = self.replay_engine.load_replay(replay_path)
            return jsonify({"success": success})
        
        @self.flask_app.route('/api/replay/state', methods=['GET'])
        def get_replay_state():
            state = self.replay_engine.get_current_state()
            return jsonify(self._serialize_state(state))
        
        @self.flask_app.route('/api/replay/navigate', methods=['POST'])
        def navigate_replay():
            direction = request.json.get('direction')  # 'next', 'prev', 'jump'
            result = self.replay_engine.navigate(direction, request.json.get('position'))
            return jsonify(result)

class HumanPlayService(BaseWebService):
    """Microservice for human gameplay"""
    
    def __init__(self, port: int = 8003):
        super().__init__("human-play-service", port, HumanPlayConfig())
        self.game_controller = HumanGameController()
    
    def setup_service_routes(self):
        @self.flask_app.route('/api/human/game/new', methods=['POST'])
        def new_game():
            config = request.json
            game_id = self.game_controller.start_new_game(config)
            return jsonify({"game_id": game_id})
        
        @self.flask_app.route('/api/human/game/<game_id>/move', methods=['POST'])
        def make_move(game_id):
            direction = request.json.get('direction')
            result = self.game_controller.make_move(game_id, direction)
            return jsonify(result)
```

**API Gateway:**
```python
class WebApiGateway:
    """Central API gateway that routes requests to appropriate services"""
    
    def __init__(self):
        self.flask_app = Flask("api-gateway")
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.rate_limiter = RateLimiter()
        self.auth_middleware = AuthMiddleware()
        self._setup_routes()
    
    def _setup_routes(self):
        @self.flask_app.route('/', methods=['GET'])
        def serve_index():
            # Determine which service should handle based on mode
            mode = request.args.get('mode', 'live')
            return self._proxy_to_service(mode, '/')
        
        @self.flask_app.route('/api/<service>/<path:endpoint>', methods=['GET', 'POST'])
        def proxy_api_request(service, endpoint):
            return self._proxy_to_service(service, f'/api/{service}/{endpoint}')
    
    def _proxy_to_service(self, service_name: str, path: str):
        """Proxy request to appropriate service"""
        # Apply middleware
        if not self.rate_limiter.allow_request(request.remote_addr):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        if not self.auth_middleware.authenticate(request):
            return jsonify({"error": "Authentication required"}), 401
        
        # Find service instance
        service_url = self.service_registry.get_service_url(service_name)
        if not service_url:
            return jsonify({"error": f"Service {service_name} not available"}), 503
        
        # Proxy the request
        response = self.load_balancer.forward_request(service_url + path, request)
        return response

class ServiceRegistry:
    """Registry for discovering and managing microservices"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.health_checker = ServiceHealthChecker()
    
    def register_service(self, service_name: str, instance: ServiceInstance):
        """Register a service instance"""
        self.services[service_name].append(instance)
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get URL for a healthy service instance"""
        healthy_instances = [
            instance for instance in self.services[service_name]
            if self.health_checker.is_healthy(instance)
        ]
        
        if healthy_instances:
            return self.load_balancer.select_instance(healthy_instances).url
        return None
```

**Service Discovery & Configuration:**
```python
class ServiceOrchestrator:
    """Manages the lifecycle of all microservices"""
    
    def __init__(self):
        self.services: List[BaseWebService] = []
        self.gateway = WebApiGateway()
        self.service_registry = ServiceRegistry()
    
    def add_service(self, service: BaseWebService):
        """Add a service to the orchestrator"""
        self.services.append(service)
        
        # Register with service registry
        instance = ServiceInstance(
            service.service_name,
            f"http://localhost:{service.port}",
            service.get_service_info()
        )
        self.service_registry.register_service(service.service_name, instance)
    
    def start_all_services(self):
        """Start all services and the gateway"""
        # Start services in parallel
        service_threads = []
        for service in self.services:
            thread = threading.Thread(target=service.start)
            thread.daemon = True
            thread.start()
            service_threads.append(thread)
        
        # Wait for services to be ready
        time.sleep(2)
        
        # Start the gateway
        self.gateway.flask_app.run(host='0.0.0.0', port=8000)
```

**Benefits:**
- True independent scaling of each web mode
- Fault isolation - one mode failing doesn't affect others
- Technology diversity - each service can use different tech stacks
- Easy deployment and updates
- Natural load balancing and high availability

**Design Patterns Used:**
- **Microservices Pattern:** Core architecture
- **API Gateway Pattern:** Central request routing
- **Service Registry Pattern:** Service discovery
- **Load Balancer Pattern:** Request distribution
- **Circuit Breaker Pattern:** Fault tolerance

---

## **Approach 8: Layered Architecture with Cross-Cutting Concerns**

**Core Philosophy:** Organize the web modes into distinct layers (Presentation, Business Logic, Data Access) with cross-cutting concerns (Security, Logging, Caching) handled by aspect-oriented programming techniques.

**Implementation Ideas:**

**Layer Architecture:**
```python
class LayeredWebApplication:
    """Application organized in distinct layers"""
    
    def __init__(self):
        # Cross-cutting concerns
        self.security_layer = SecurityLayer()
        self.logging_layer = LoggingLayer()
        self.caching_layer = CachingLayer()
        self.validation_layer = ValidationLayer()
        
        # Core layers
        self.presentation_layer = PresentationLayer()
        self.business_layer = BusinessLayer()
        self.data_layer = DataLayer()
        
        # Aspect weaver for cross-cutting concerns
        self.aspect_weaver = AspectWeaver()
        self._weave_aspects()
    
    def _weave_aspects(self):
        """Apply cross-cutting concerns to all layers"""
        layers = [self.presentation_layer, self.business_layer, self.data_layer]
        
        for layer in layers:
            self.aspect_weaver.apply_aspect(self.security_layer, layer)
            self.aspect_weaver.apply_aspect(self.logging_layer, layer)
            self.aspect_weaver.apply_aspect(self.caching_layer, layer)
```

**Presentation Layer:**
```python
class PresentationLayer:
    """Handles all web presentation concerns"""
    
    def __init__(self):
        self.view_models: Dict[str, ViewModel] = {}
        self.template_engine = TemplateEngine()
        self.response_formatters: Dict[str, ResponseFormatter] = {}
    
    def register_view_model(self, name: str, view_model: ViewModel):
        """Register a view model for a specific web mode"""
        self.view_models[name] = view_model
    
    def handle_request(self, request: WebRequest) -> WebResponse:
        """Handle incoming web request"""
        mode = self._determine_mode(request)
        view_model = self.view_models[mode]
        
        # Process request through view model
        presentation_data = view_model.process_request(request)
        
        # Format response
        formatter = self.response_formatters[request.content_type]
        return formatter.format_response(presentation_data)

class LiveGameViewModel(ViewModel):
    """View model for live game mode"""
    
    def __init__(self, game_service: GameService):
        self.game_service = game_service
        self.state_transformer = GameStateTransformer()
    
    def process_request(self, request: WebRequest) -> PresentationData:
        if request.path == "/api/state":
            game_state = self.game_service.get_current_state()
            return self.state_transformer.transform_for_web(game_state)
        
        elif request.path == "/api/control":
            command = request.data.get("command")
            result = self.game_service.handle_control(command)
            return PresentationData("control_result", result)

class ReplayViewModel(ViewModel):
    """View model for replay mode"""
    
    def __init__(self, replay_service: ReplayService):
        self.replay_service = replay_service
        self.replay_transformer = ReplayStateTransformer()
    
    def process_request(self, request: WebRequest) -> PresentationData:
        if request.path == "/api/replay/state":
            replay_state = self.replay_service.get_current_state()
            return self.replay_transformer.transform_for_web(replay_state)
```

**Business Layer:**
```python
class BusinessLayer:
    """Contains all business logic for different web modes"""
    
    def __init__(self):
        self.services: Dict[str, BusinessService] = {}
        self.workflow_engine = WorkflowEngine()
        self.rule_engine = RuleEngine()
    
    def register_service(self, name: str, service: BusinessService):
        """Register a business service"""
        self.services[name] = service

class GameService(BusinessService):
    """Business service for game-related operations"""
    
    def __init__(self, data_access: GameDataAccess):
        self.data_access = data_access
        self.game_rules = GameRules()
        self.state_manager = GameStateManager()
    
    def get_current_state(self) -> GameState:
        """Get current game state with business rules applied"""
        raw_state = self.data_access.get_current_state()
        return self.game_rules.apply_business_rules(raw_state)
    
    def handle_control(self, command: str) -> ControlResult:
        """Handle control command with validation"""
        if not self.game_rules.is_valid_command(command):
            return ControlResult(success=False, error="Invalid command")
        
        result = self.state_manager.execute_command(command)
        self.data_access.persist_state_change(result)
        return result

class ReplayService(BusinessService):
    """Business service for replay operations"""
    
    def __init__(self, data_access: ReplayDataAccess):
        self.data_access = data_access
        self.replay_rules = ReplayRules()
        self.navigation_manager = ReplayNavigationManager()
```

**Data Access Layer:**
```python
class DataLayer:
    """Handles all data access operations"""
    
    def __init__(self):
        self.repositories: Dict[str, Repository] = {}
        self.connection_manager = ConnectionManager()
        self.transaction_manager = TransactionManager()
    
    def register_repository(self, name: str, repository: Repository):
        """Register a data repository"""
        self.repositories[name] = repository

class GameDataAccess(Repository):
    """Data access for game-related operations"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.state_serializer = GameStateSerializer()
    
    def get_current_state(self) -> RawGameState:
        """Get raw game state from data source"""
        connection = self.connection_manager.get_connection()
        raw_data = connection.query("SELECT * FROM game_state WHERE active = true")
        return self.state_serializer.deserialize(raw_data)
    
    def persist_state_change(self, state_change: StateChange):
        """Persist game state changes"""
        with self.transaction_manager.transaction():
            connection = self.connection_manager.get_connection()
            serialized_change = self.state_serializer.serialize(state_change)
            connection.execute("INSERT INTO game_state_history VALUES (?)", serialized_change)
```

**Cross-Cutting Concerns:**
```python
class SecurityAspect(Aspect):
    """Security cross-cutting concern"""
    
    def before_method(self, method_name: str, args: tuple, kwargs: dict):
        """Apply security checks before method execution"""
        if self.requires_authentication(method_name):
            if not self.authenticate_request():
                raise SecurityException("Authentication required")
        
        if self.requires_authorization(method_name):
            if not self.authorize_request(method_name):
                raise SecurityException("Insufficient permissions")
    
    def after_method(self, method_name: str, result: Any):
        """Apply security filters after method execution"""
        if self.requires_data_filtering(method_name):
            return self.filter_sensitive_data(result)
        return result

class LoggingAspect(Aspect):
    """Logging cross-cutting concern"""
    
    def before_method(self, method_name: str, args: tuple, kwargs: dict):
        """Log method entry"""
        self.logger.info(f"Entering {method_name} with args: {args}")
        self.performance_monitor.start_timer(method_name)
    
    def after_method(self, method_name: str, result: Any):
        """Log method exit and performance"""
        duration = self.performance_monitor.end_timer(method_name)
        self.logger.info(f"Exiting {method_name}, duration: {duration}ms")
        
        if duration > self.slow_query_threshold:
            self.alert_manager.send_performance_alert(method_name, duration)

class CachingAspect(Aspect):
    """Caching cross-cutting concern"""
    
    def around_method(self, method_name: str, args: tuple, kwargs: dict, proceed_func):
        """Cache method results"""
        cache_key = self.generate_cache_key(method_name, args, kwargs)
        
        # Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute method and cache result
        result = proceed_func()
        self.cache_manager.set(cache_key, result, self.get_ttl(method_name))
        return result
```

**Aspect Weaver:**
```python
class AspectWeaver:
    """Weaves cross-cutting concerns into application layers"""
    
    def __init__(self):
        self.pointcut_matcher = PointcutMatcher()
        self.advice_executor = AdviceExecutor()
    
    def apply_aspect(self, aspect: Aspect, target: object):
        """Apply aspect to target object"""
        for method_name in dir(target):
            if self.pointcut_matcher.matches(aspect.pointcuts, method_name):
                original_method = getattr(target, method_name)
                enhanced_method = self._create_enhanced_method(aspect, original_method)
                setattr(target, method_name, enhanced_method)
    
    def _create_enhanced_method(self, aspect: Aspect, original_method):
        """Create method with aspect advice applied"""
        def enhanced_method(*args, **kwargs):
            try:
                # Before advice
                aspect.before_method(original_method.__name__, args, kwargs)
                
                # Around advice (if present)
                if hasattr(aspect, 'around_method'):
                    return aspect.around_method(
                        original_method.__name__, 
                        args, 
                        kwargs, 
                        lambda: original_method(*args, **kwargs)
                    )
                else:
                    result = original_method(*args, **kwargs)
                    # After advice
                    result = aspect.after_method(original_method.__name__, result)
                    return result
            
            except Exception as e:
                # Exception advice
                if hasattr(aspect, 'on_exception'):
                    aspect.on_exception(original_method.__name__, e)
                raise
        
        return enhanced_method
```

**Benefits:**
- Clean separation of concerns across layers
- Cross-cutting concerns handled uniformly
- Easy to test each layer independently
- Consistent security, logging, and caching across all web modes
- Clear dependency flow from top to bottom

**Design Patterns Used:**
- **Layered Architecture Pattern:** Core structure
- **Aspect-Oriented Programming:** Cross-cutting concerns
- **Repository Pattern:** Data access abstraction
- **View Model Pattern:** Presentation logic separation
- **Service Layer Pattern:** Business logic encapsulation

---

## **Approach 9: Domain-Driven Design with Bounded Contexts**

**Core Philosophy:** Apply Domain-Driven Design principles to identify distinct bounded contexts for each web mode, creating a rich domain model with ubiquitous language and proper domain boundaries.

**Implementation Ideas:**

**Domain Model Architecture:**
```python
# Core Domain - Shared concepts across all contexts
class SnakeGameDomain:
    """Core domain containing shared game concepts"""
    
    class Position(ValueObject):
        """Immutable position value object"""
        def __init__(self, x: int, y: int):
            self._x = x
            self._y = y
            self._validate()
        
        def _validate(self):
            if self._x < 0 or self._y < 0:
                raise ValueError("Position coordinates must be non-negative")
        
        @property
        def x(self) -> int:
            return self._x
        
        @property
        def y(self) -> int:
            return self._y
        
        def move(self, direction: 'Direction') -> 'Position':
            """Return new position after moving in direction"""
            return Position(
                self._x + direction.delta_x,
                self._y + direction.delta_y
            )
    
    class Direction(Enum):
        """Direction enumeration with movement deltas"""
        UP = (0, -1)
        DOWN = (0, 1)
        LEFT = (-1, 0)
        RIGHT = (1, 0)
        
        @property
        def delta_x(self) -> int:
            return self.value[0]
        
        @property
        def delta_y(self) -> int:
            return self.value[1]
        
        def opposite(self) -> 'Direction':
            """Get opposite direction"""
            opposites = {
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT
            }
            return opposites[self]
    
    class Snake(Entity):
        """Snake aggregate root"""
        def __init__(self, snake_id: SnakeId, initial_position: Position):
            self._id = snake_id
            self._positions = [initial_position]
            self._direction = Direction.RIGHT
            self._domain_events = []
        
        def move(self, direction: Direction, grow: bool = False):
            """Move snake in given direction"""
            if direction == self._direction.opposite():
                raise InvalidMoveError("Cannot reverse direction")
            
            self._direction = direction
            new_head = self._positions[0].move(direction)
            
            # Check for self-collision
            if new_head in self._positions:
                self._add_domain_event(SnakeCollidedWithSelfEvent(self._id))
```