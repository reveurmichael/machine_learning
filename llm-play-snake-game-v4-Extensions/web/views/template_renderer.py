"""
Simple Template Renderer
========================

Basic template rendering implementation for the MVC framework.
Provides minimal functionality to support the web interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json
from pathlib import Path


class TemplateRenderer(ABC):
    """
    Abstract base class for template rendering.
    
    Defines the interface for rendering templates with data.
    """
    
    @abstractmethod
    def render_template(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of template to render
            context: Data to pass to template
            
        Returns:
            Rendered template as string
        """
        pass
    
    @abstractmethod
    def render_json(self, data: Dict[str, Any]) -> str:
        """
        Render data as JSON response.
        
        Args:
            data: Data to serialize as JSON
            
        Returns:
            JSON string
        """
        pass


class SimpleTemplateRenderer(TemplateRenderer):
    """
    Simple template renderer implementation.
    
    Provides basic template rendering without external dependencies.
    """
    
    def __init__(self, template_folder: str = None, static_folder: str = None):
        """
        Initialize simple template renderer.
        
        Args:
            template_folder: Path to template directory
            static_folder: Path to static files directory
        """
        self.template_folder = Path(template_folder) if template_folder else None
        self.static_folder = Path(static_folder) if static_folder else None
    
    def render_template(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """
        Render template with simple string substitution.
        
        Args:
            template_name: Name of template file
            context: Variables to substitute in template
            
        Returns:
            Rendered template string
        """
        context = context or {}
        
        # Try to load template file
        if self.template_folder:
            template_path = self.template_folder / template_name
            if template_path.exists():
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                    
                    # Simple variable substitution
                    for key, value in context.items():
                        placeholder = f"{{{{{key}}}}}"
                        template_content = template_content.replace(placeholder, str(value))
                    
                    return template_content
                except Exception as e:
                    return f"Error loading template {template_name}: {e}"
        
        # Fallback to simple HTML
        return self._generate_simple_html(template_name, context)
    
    def render_json(self, data: Dict[str, Any]) -> str:
        """
        Render data as JSON string.
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": f"JSON serialization failed: {e}"})
    
    def _generate_simple_html(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Generate simple HTML when template file is not found.
        
        Args:
            template_name: Name of requested template
            context: Template context
            
        Returns:
            Simple HTML page
        """
        title = context.get('title', 'Snake Game')
        controller_name = context.get('controller_name', 'Game Controller')
        game_mode = context.get('game_mode', 'unknown')
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2c3e50; }}
                .info {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .status {{ padding: 10px; background: #3498db; color: white; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🐍 {title}</h1>
                <div class="info">
                    <h2>Controller: {controller_name}</h2>
                    <p>Game Mode: {game_mode}</p>
                    <p>Template: {template_name}</p>
                </div>
                <div class="status">
                    MVC Framework Active - Template rendering working
                </div>
                <p>This is a simple fallback template. For full functionality, implement proper templates.</p>
            </div>
        </body>
        </html>
        """


class WebViewRenderer:
    """
    Main view renderer for web responses.
    
    Coordinates template rendering and response formatting.
    """
    
    def __init__(self, template_folder: str = None, static_folder: str = None):
        """
        Initialize web view renderer.
        
        Args:
            template_folder: Path to templates
            static_folder: Path to static files
        """
        self.template_renderer = SimpleTemplateRenderer(template_folder, static_folder)
        self.decorators = []
    
    def render_template(self, template_name: str, **context) -> str:
        """
        Render template with context.
        
        Args:
            template_name: Name of template
            **context: Template variables
            
        Returns:
            Rendered template
        """
        return self.template_renderer.render_template(template_name, context)
    
    def render_json_response(self, data: Dict[str, Any]) -> str:
        """
        Render JSON response.
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON string
        """
        return self.template_renderer.render_json(data)
    
    def render_error_page(self, error_message: str, status_code: int = 500) -> str:
        """
        Render error page.
        
        Args:
            error_message: Error description
            status_code: HTTP status code
            
        Returns:
            Error page HTML
        """
        context = {
            'title': f'Error {status_code}',
            'controller_name': 'Error Handler',
            'game_mode': 'error',
            'error_message': error_message,
            'status_code': status_code
        }
        
        return self.template_renderer._generate_simple_html('error.html', context)
    
    def add_decorator(self, decorator):
        """
        Add a decorator to the renderer.
        
        Args:
            decorator: Decorator instance to add
        """
        self.decorators.append(decorator) 