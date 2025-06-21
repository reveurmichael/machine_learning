"""
Views Module - MVC Architecture
==============================

View layer components for rendering responses and managing templates.
Provides clean separation between data presentation and business logic.

Design Patterns:
    - Strategy Pattern: Pluggable template engines
    - Decorator Pattern: View decorators for enhanced functionality
    - Template Method Pattern: Common rendering workflow
    - Factory Pattern: View component creation

Architecture Overview:
    Template Engines/   - Different template rendering strategies
    Decorators/        - View enhancement components
    Renderers/         - Response rendering coordination
    
Educational Goals:
    - Demonstrate clean view layer architecture
    - Show how Strategy pattern enables template flexibility
    - Illustrate proper separation of presentation logic
"""

from .template_renderer import WebViewRenderer, TemplateRenderer, SimpleTemplateRenderer
from .template_engines import TemplateEngine, SimpleTemplateEngine, JinjaTemplateEngine, create_template_engine

__all__ = [
    'WebViewRenderer',
    'TemplateRenderer', 
    'SimpleTemplateRenderer',
    'TemplateEngine',
    'SimpleTemplateEngine',
    'JinjaTemplateEngine',
    'create_template_engine'
] 