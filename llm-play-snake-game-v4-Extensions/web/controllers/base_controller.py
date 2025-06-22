"""
Base Web Controller - MVC Architecture Foundation
===============================================

Abstract base controller implementing common web functionality and request processing pipeline.
Uses Template Method pattern to define request handling flow while allowing subclasses to customize behavior.

Design Patterns Used:
    - Template Method: handle_request() defines the processing pipeline
    - Strategy: ModelManager and ViewRenderer are pluggable strategies
    - Chain of Responsibility: Request filters can block or modify requests
    - Observer: State change notifications to registered observers

Educational Goals:
    - Show how Template Method pattern provides consistent request processing
    - Demonstrate separation between HTTP concerns and business logic
    - Illustrate how abstract base classes enforce architectural principles
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from flask import Request, Response, jsonify
import time
import logging
from dataclasses import dataclass
from enum import Enum

# Import MVC components
from ..models import GameStateModel, Observer, GameEvent
from ..views import WebViewRenderer

# Configure logging
logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of requests that controllers handle."""
    STATE_GET = "state_get"
    CONTROL_POST = "control_post"
    RESET_POST = "reset_post"
    HEALTH_GET = "health_get"
    INDEX_GET = "index_get"


@dataclass
class RequestContext:
    """
    Request context containing all information needed for processing.
    
    Encapsulates request data, metadata, and processing state to pass
    through the request pipeline cleanly.
    """
    request_type: RequestType
    raw_request: Request
    data: Dict[str, Any]
    timestamp: float
    client_ip: str
    user_agent: str
    processing_start: float
    metadata: Dict[str, Any]

    @classmethod
    def from_flask_request(cls, request: Request, request_type: RequestType) -> 'RequestContext':
        """Create RequestContext from Flask request object."""
        return cls(
            request_type=request_type,
            raw_request=request,
            data=request.get_json(silent=True) or {},
            timestamp=time.time(),
            client_ip=request.remote_addr or 'unknown',
            user_agent=request.headers.get('User-Agent', 'unknown'),
            processing_start=time.time(),
            metadata={}
        )


class RequestFilter(ABC):
    """
    Abstract request filter for preprocessing requests.
    
    Implements Chain of Responsibility pattern for request validation,
    authentication, rate limiting, etc.
    """
    
    @abstractmethod
    def should_process(self, context: RequestContext) -> bool:
        """
        Determine if request should be processed.
        
        Args:
            context: Request context with all request information
            
        Returns:
            True if request should continue processing, False to block
        """
        pass
    
    def preprocess(self, context: RequestContext) -> RequestContext:
        """
        Preprocess request context before handling.
        
        Args:
            context: Original request context
            
        Returns:
            Modified request context
        """
        return context
    
    def postprocess(self, context: RequestContext, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess response before sending to client.
        
        Args:
            context: Request context
            response: Response data
            
        Returns:
            Modified response data
        """
        return response


class RateLimitFilter(RequestFilter):
    """Request filter implementing rate limiting."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_history: Dict[str, List[float]] = {}
    
    def should_process(self, context: RequestContext) -> bool:
        """Check if client hasn't exceeded rate limit."""
        client_ip = context.client_ip
        # ------------------
        # Interactive-gameplay exemptions
        # ------------------
        #  1. CONTROL_POST – arrow-key spam can easily hit 500+ req/min.
        #  2. STATE_GET    – the front-end polls every 100 ms (≈600 req/min)
        #                     to animate the board; throttling these turns the
        #                     game into a slideshow and also starves the
        #                     control requests because browsers queue Ajax
        #                     ops per-origin.
        #
        # Any future *streaming* implementation (WebSockets/EventSource) would
        # drop STATE_GET altogether, but until then we simply exempt both
        # interactive endpoints from server-side rate limiting.

        if context.request_type in (RequestType.CONTROL_POST, RequestType.STATE_GET):
            return True  # no rate limiting for fast interactive endpoints

        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.request_history:
            self.request_history[client_ip] = [
                timestamp for timestamp in self.request_history[client_ip]
                if current_time - timestamp < 60  # Keep only last minute
            ]
        else:
            self.request_history[client_ip] = []
        
        # Check rate limit (non-interactive requests)
        if len(self.request_history[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
        
        # Record this request
        self.request_history[client_ip].append(current_time)
        return True


class BaseWebController(Observer, ABC):
    """
    Abstract base controller with common web functionality.
    
    Implements Template Method pattern for request processing pipeline.
    Subclasses customize behavior by overriding abstract methods.
    
    Template Method Flow:
        1. validate_request() - Check if request is valid
        2. preprocess_request() - Apply filters and transformations
        3. handle_state_request() OR handle_control_request() - Core logic
        4. postprocess_response() - Apply response transformations
        5. format_response() - Convert to HTTP response
    
    Design Patterns:
        - Template Method: handle_request() defines processing flow
        - Strategy: model_manager and view_renderer are pluggable
        - Observer: Implements Observer interface for model events
        - Chain of Responsibility: Request filters
    """
    
    def __init__(self, model_manager: GameStateModel, view_renderer: WebViewRenderer, **kwargs):
        """
        Initialize base controller with required dependencies.
        
        Args:
            model_manager: Handles game state and business logic
            view_renderer: Handles response rendering and templating
            **kwargs: Additional configuration options
        """
        super().__init__()
        self.model_manager = model_manager
        self.view_renderer = view_renderer
        self.request_filters: List[RequestFilter] = []
        self.observers: List[Observer] = []
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Register as observer for model events
        self.model_manager.add_observer(self)
        
        # Add default filters
        self._setup_default_filters()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _setup_default_filters(self):
        """Set up default request filters."""
        # Add rate limiting
        self.add_request_filter(RateLimitFilter(max_requests_per_minute=120))
    
    def add_request_filter(self, filter_instance: RequestFilter):
        """Add a request filter to the processing chain."""
        self.request_filters.append(filter_instance)
        logger.debug(f"Added request filter: {filter_instance.__class__.__name__}")
    
    def add_observer(self, observer: Observer):
        """Add observer for controller events."""
        self.observers.append(observer)
    
    def notify_observers(self, event: GameEvent):
        """Notify all observers of events."""
        for observer in self.observers:
            try:
                observer.on_game_event(event)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    # Template Method - defines the request processing flow
    def handle_request(self, request: Request, request_type: RequestType) -> Response:
        """
        Template method defining request processing pipeline.
        
        This method implements the Template Method pattern, defining the
        overall structure of request processing while allowing subclasses
        to customize specific steps.
        
        Args:
            request: Flask request object
            request_type: Type of request being processed
            
        Returns:
            Flask Response object
        """
        start_time = time.time()
        context = RequestContext.from_flask_request(request, request_type)
        
        try:
            self.request_count += 1
            
            # Step 1: Validate request
            if not self.validate_request(context):
                return self._create_error_response("Invalid request", 400)
            
            # Step 2: Apply request filters
            for filter_instance in self.request_filters:
                if not filter_instance.should_process(context):
                    return self._create_error_response("Request blocked by filter", 429)
                context = filter_instance.preprocess(context)
            
            # Step 3: Preprocess request
            context = self.preprocess_request(context)
            
            # Step 4: Route to appropriate handler
            if request_type == RequestType.STATE_GET:
                response_data = self.handle_state_request(context)
            elif request_type == RequestType.CONTROL_POST:
                response_data = self.handle_control_request(context)
            elif request_type == RequestType.RESET_POST:
                response_data = self.handle_reset_request(context)
            elif request_type == RequestType.HEALTH_GET:
                response_data = self.handle_health_request(context)
            elif request_type == RequestType.INDEX_GET:
                return self.handle_index_request(context)
            else:
                response_data = {"error": f"Unknown request type: {request_type}"}
            
            # Step 5: Postprocess response
            response_data = self.postprocess_response(context, response_data)
            
            # Step 6: Apply response filters
            for filter_instance in reversed(self.request_filters):
                response_data = filter_instance.postprocess(context, response_data)
            
            # Step 7: Format final response
            response = self.format_response(response_data)
            
            # Track performance
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request processing failed: {e}", exc_info=True)
            return self._create_error_response(f"Internal server error: {str(e)}", 500)
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle /api/state requests.
        
        Args:
            context: Request context with all request information
            
        Returns:
            Dictionary containing current game state
        """
        pass
    
    @abstractmethod
    def handle_control_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle /api/control requests.
        
        Args:
            context: Request context with command data
            
        Returns:
            Dictionary containing command execution result
        """
        pass
    
    # Hook methods that subclasses can override
    def validate_request(self, context: RequestContext) -> bool:
        """
        Validate incoming request.
        
        Args:
            context: Request context to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        return True
    
    def preprocess_request(self, context: RequestContext) -> RequestContext:
        """
        Preprocess request before handling.
        
        Args:
            context: Original request context
            
        Returns:
            Modified request context
        """
        return context
    
    def postprocess_response(self, context: RequestContext, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess response before formatting.
        
        Args:
            context: Request context
            response: Response data
            
        Returns:
            Modified response data
        """
        # Add common response metadata
        response.setdefault('timestamp', time.time())
        response.setdefault('controller', self.__class__.__name__)
        return response
    
    def handle_reset_request(self, context: RequestContext) -> Dict[str, Any]:
        """
        Handle /api/reset requests.
        
        Default implementation delegates to model manager.
        """
        try:
            self.model_manager.reset_game()
            return {
                "status": "success",
                "message": "Game reset successfully"
            }
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return {
                "status": "error", 
                "message": f"Reset failed: {str(e)}"
            }
    
    def handle_health_request(self, context: RequestContext) -> Dict[str, Any]:
        """Handle /api/health requests with performance metrics."""
        avg_processing_time = (
            self.total_processing_time / max(self.request_count, 1)
        )
        
        return {
            "status": "healthy",
            "controller": self.__class__.__name__,
            "performance": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "success_rate": 1.0 - (self.error_count / max(self.request_count, 1)),
                "avg_processing_time_ms": avg_processing_time * 1000
            },
            "model_status": self.model_manager.get_health_status()
        }
    
    def handle_index_request(self, context: RequestContext) -> Response:
        """
        Handle index page requests.
        
        Returns rendered HTML template response.
        """
        template_context = self.get_index_template_context()
        return self.view_renderer.render_template(
            self.get_index_template_name(),
            **template_context  # FIX: Unpack context as keyword arguments
        )
    
    def get_index_template_name(self) -> str:
        """Get template name for index page. Override in subclasses."""
        return "base.html"
    
    def get_index_template_context(self) -> Dict[str, Any]:
        """Get template context for index page. Override in subclasses."""
        return {
            "controller_name": self.__class__.__name__,
            "timestamp": time.time()
        }
    
    def format_response(self, data: Dict[str, Any]) -> Response:
        """Format response data as JSON."""
        return jsonify(data)
    
    def _create_error_response(self, message: str, status_code: int) -> Response:
        """Create standardized error response."""
        error_data = {
            "status": "error",
            "message": message,
            "timestamp": time.time(),
            "controller": self.__class__.__name__
        }
        return jsonify(error_data), status_code
    
    # Event handling for controllers that need it
    def handle_game_event(self, event: GameEvent):
        """
        Handle game events from model layer.
        
        Default implementation logs events. Subclasses can override
        for specific event handling.
        """
        logger.debug(f"Received game event: {event}")
    
    def on_game_event(self, event: GameEvent) -> None:
        """Receive game events; base controllers ignore by default."""
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": 1.0 - (self.error_count / max(self.request_count, 1)),
            "avg_processing_time_ms": (
                self.total_processing_time / max(self.request_count, 1)
            ) * 1000
        } 