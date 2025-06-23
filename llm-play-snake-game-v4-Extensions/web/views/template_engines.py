"""
Template Engines - MVC Architecture
--------------------

Template rendering engines for the web framework.
Provides abstraction layer for different template systems.

Design Patterns Used:
    - Strategy Pattern: Different template engines can be plugged in
    - Template Method Pattern: Common rendering workflow
    - Decorator Pattern: Template decorators for enhanced functionality

Educational Goals:
    - Show how Strategy pattern enables template engine flexibility
    - Demonstrate separation of template logic from view logic
    - Illustrate how abstraction enables easy switching between engines
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateEngine(ABC):
    """
    Abstract base class for template engines.
    
    Strategy Pattern: Allows different template engines to be used interchangeably.
    Template Method Pattern: Defines common template rendering workflow.
    """
    
    def __init__(self, template_folder: str, static_folder: str = "static", **kwargs):
        """
        Initialize template engine.
        
        Args:
            template_folder: Path to template files
            static_folder: Path to static files
            **kwargs: Engine-specific configuration
        """
        self.template_folder = Path(template_folder)
        self.static_folder = Path(static_folder)
        self.config = kwargs
        
        # Ensure directories exist
        self.template_folder.mkdir(parents=True, exist_ok=True)
        self.static_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with template folder: {template_folder}")
    
    @abstractmethod
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template file
            context: Variables to pass to the template
            
        Returns:
            Rendered template as string
        """
        pass
    
    @abstractmethod
    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists."""
        pass
    
    def get_template_path(self, template_name: str) -> Path:
        """Get full path to a template file."""
        return self.template_folder / template_name
    
    def get_static_path(self, static_file: str) -> Path:
        """Get full path to a static file."""
        return self.static_folder / static_file


class SimpleTemplateEngine(TemplateEngine):
    """
    Simple template engine with basic string substitution.
    
    Concrete Strategy: Implements basic template functionality for demonstration.
    Suitable for simple templates without complex logic.
    """
    
    def __init__(self, template_folder: str, static_folder: str = "static", **kwargs):
        """Initialize simple template engine."""
        super().__init__(template_folder, static_folder, **kwargs)
        self.template_cache = {}
        
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render template using simple string substitution.
        
        Uses Python's str.format() method for variable substitution.
        """
        try:
            # Load template content
            template_content = self._load_template(template_name)
            
            # Simple variable substitution
            rendered = template_content.format(**context)
            
            logger.debug(f"Rendered template: {template_name}")
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return f"<html><body><h1>Template Error</h1><p>{str(e)}</p></body></html>"
    
    def template_exists(self, template_name: str) -> bool:
        """Check if template file exists."""
        template_path = self.get_template_path(template_name)
        return template_path.exists() and template_path.is_file()
    
    def _load_template(self, template_name: str) -> str:
        """Load template content from file."""
        # Check cache first
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        template_path = self.get_template_path(template_name)
        
        if not template_path.exists():
            # Return default template if file doesn't exist
            default_content = self._get_default_template(template_name)
            self.template_cache[template_name] = default_content
            return default_content
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache the template
            self.template_cache[template_name] = content
            return content
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            return f"<html><body><h1>Error loading template: {template_name}</h1></body></html>"
    
    def _get_default_template(self, template_name: str) -> str:
        """Get default template content when file doesn't exist."""
        if template_name == 'index.html':
            return """<!DOCTYPE html>
<html>
<head>
    <title>Snake Game - {game_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .game-container {{ max-width: 800px; margin: 0 auto; }}
        .status {{ background: #f0f0f0; padding: 20px; margin: 20px 0; }}
        .controls {{ margin: 20px 0; }}
        button {{ padding: 10px 20px; margin: 5px; }}
    </style>
</head>
<body>
    <div class="game-container">
        <h1>Snake Game - {game_mode}</h1>
        <div class="status">
            <p>Score: {score}</p>
            <p>Status: {game_status}</p>
        </div>
        <div class="controls">
            <button onclick="makeMove('up')">Up</button>
            <button onclick="makeMove('down')">Down</button>
            <button onclick="makeMove('left')">Left</button>
            <button onclick="makeMove('right')">Right</button>
        </div>
    </div>
    <script>
        function makeMove(direction) {{
            fetch('/api/control', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{'action': 'move', 'direction': direction}})
            }});
        }}
    </script>
</body>
</html>"""
        else:
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Snake Game</title>
</head>
<body>
    <h1>Template: {template_name}</h1>
    <p>Default template content</p>
</body>
</html>"""


class JinjaTemplateEngine(TemplateEngine):
    """
    Jinja2-based template engine.
    
    Concrete Strategy: Implements full-featured template engine using Jinja2.
    Provides advanced templating features like inheritance, filters, etc.
    """
    
    def __init__(self, template_folder: str, static_folder: str = "static", **kwargs):
        """Initialize Jinja template engine."""
        super().__init__(template_folder, static_folder, **kwargs)
        
        # Try to import Jinja2, fall back to simple engine if not available
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_folder)),
                autoescape=select_autoescape(['html', 'xml']),
                cache_size=kwargs.get('cache_size', 50),
                auto_reload=kwargs.get('auto_reload', True)
            )
            
            # Add custom filters
            self.jinja_env.filters['json'] = self._json_filter
            
            self.jinja_available = True
            logger.info("Initialized Jinja2 template engine")
            
        except ImportError:
            logger.warning("Jinja2 not available, falling back to simple template engine")
            self.jinja_available = False
            self.fallback_engine = SimpleTemplateEngine(template_folder, static_folder, **kwargs)
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render template using Jinja2 or fallback engine."""
        if not self.jinja_available:
            return self.fallback_engine.render_template(template_name, context)
        
        try:
            template = self.jinja_env.get_template(template_name)
            rendered = template.render(**context)
            
            logger.debug(f"Rendered Jinja template: {template_name}")
            return rendered
            
        except Exception as e:
            logger.debug(f"Jinja template {template_name} not found, using fallback: {e}")
            
            # Try fallback
            if hasattr(self, 'fallback_engine'):
                return self.fallback_engine.render_template(template_name, context)
            
            # Create fallback content
            return self._get_fallback_template(template_name, context)
    
    def template_exists(self, template_name: str) -> bool:
        """Check if template exists."""
        if not self.jinja_available:
            return self.fallback_engine.template_exists(template_name)
        
        try:
            self.jinja_env.get_template(template_name)
            return True
        except:
            return False
    
    def _json_filter(self, value: Any) -> str:
        """Custom Jinja filter to convert values to JSON."""
        import json
        return json.dumps(value, default=str)
    
    def _get_fallback_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Get fallback template when Jinja template is not found."""
        if template_name == 'index.html':
            return """<!DOCTYPE html>
<html>
<head>
    <title>Snake Game - {game_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .game-container {{ max-width: 800px; margin: 0 auto; }}
        .status {{ background: #f0f0f0; padding: 20px; margin: 20px 0; }}
        .controls {{ margin: 20px 0; }}
        button {{ padding: 10px 20px; margin: 5px; }}
    </style>
</head>
<body>
    <div class="game-container">
        <h1>Snake Game - {game_mode}</h1>
        <div class="status">
            <p>Score: {score}</p>
            <p>Status: {game_status}</p>
        </div>
        <div class="controls">
            <button onclick="makeMove('up')">Up</button>
            <button onclick="makeMove('down')">Down</button>
            <button onclick="makeMove('left')">Left</button>
            <button onclick="makeMove('right')">Right</button>
        </div>
    </div>
    <script>
        function makeMove(direction) {{
            fetch('/api/control', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{'action': 'move', 'direction': direction}})
            }});
        }}
    </script>
</body>
</html>""".format(**context)
        else:
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Snake Game</title>
</head>
<body>
    <h1>Template: {template_name}</h1>
    <p>Fallback template content</p>
</body>
</html>"""


# Factory function for easy template engine creation
def create_template_engine(engine_type: str = "jinja", 
                          template_folder: str = "templates",
                          static_folder: str = "static",
                          **kwargs) -> TemplateEngine:
    """
    Create a template engine instance.
    
    Factory Pattern: Centralized creation of template engines.
    
    Args:
        engine_type: Type of engine ('jinja' or 'simple')
        template_folder: Path to template files
        static_folder: Path to static files
        **kwargs: Engine-specific configuration
        
    Returns:
        Configured template engine instance
    """
    if engine_type.lower() == 'jinja':
        return JinjaTemplateEngine(template_folder, static_folder, **kwargs)
    elif engine_type.lower() == 'simple':
        return SimpleTemplateEngine(template_folder, static_folder, **kwargs)
    else:
        raise ValueError(f"Unknown template engine type: {engine_type}") 