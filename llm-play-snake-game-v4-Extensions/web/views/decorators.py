"""
View Decorators - MVC Architecture
=================================

Decorator pattern implementation for view enhancement.
Provides composable view functionality like caching, compression, etc.

Design Patterns Used:
    - Decorator Pattern: Transparent view enhancement
    - Chain of Responsibility: Decorator chaining
    - Strategy Pattern: Different decoration strategies

Educational Goals:
    - Show how Decorator pattern enables flexible view enhancement
    - Demonstrate composable functionality
    - Illustrate separation of cross-cutting concerns
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import gzip
import json

logger = logging.getLogger(__name__)


class ViewDecorator(ABC):
    """
    Abstract base class for view decorators.
    
    Decorator Pattern: Provides transparent enhancement of view functionality.
    """
    
    @abstractmethod
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Decorate the response.
        
        Args:
            response: Original response content
            context: Request context and metadata
            
        Returns:
            Decorated response content
        """
        pass
    
    @abstractmethod
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get information about this decorator."""
        pass


class CompressionDecorator(ViewDecorator):
    """
    Decorator for response compression.
    
    Concrete Decorator: Adds gzip compression to responses.
    """
    
    def __init__(self, min_size: int = 1000, compression_level: int = 6):
        """
        Initialize compression decorator.
        
        Args:
            min_size: Minimum response size to compress
            compression_level: Gzip compression level (1-9)
        """
        self.min_size = min_size
        self.compression_level = compression_level
        
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply compression to response if appropriate.
        
        For demonstration purposes, this returns the original response
        with compression metadata added to context.
        """
        # Check if response is large enough to compress
        if len(response) < self.min_size:
            context['compression'] = {'applied': False, 'reason': 'too_small'}
            return response
        
        # Check if client accepts compression
        accept_encoding = context.get('headers', {}).get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            context['compression'] = {'applied': False, 'reason': 'not_accepted'}
            return response
        
        # For demo, just mark as compressed
        context['compression'] = {
            'applied': True,
            'original_size': len(response),
            'compression_level': self.compression_level
        }
        
        logger.debug(f"Applied compression to response ({len(response)} bytes)")
        return response
    
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get compression decorator information."""
        return {
            'type': 'compression',
            'min_size': self.min_size,
            'compression_level': self.compression_level
        }


class CacheDecorator(ViewDecorator):
    """
    Decorator for response caching.
    
    Concrete Decorator: Adds caching headers and metadata to responses.
    """
    
    def __init__(self, cache_timeout: int = 300, cache_type: str = "public"):
        """
        Initialize cache decorator.
        
        Args:
            cache_timeout: Cache timeout in seconds
            cache_type: Cache type (public, private, no-cache)
        """
        self.cache_timeout = cache_timeout
        self.cache_type = cache_type
        
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply caching metadata to response.
        
        Adds cache control headers and metadata to the context.
        """
        # Add cache control information
        cache_control = f"{self.cache_type}, max-age={self.cache_timeout}"
        
        # Update context with cache information
        if 'headers' not in context:
            context['headers'] = {}
        
        context['headers']['Cache-Control'] = cache_control
        context['headers']['Expires'] = time.time() + self.cache_timeout
        
        context['caching'] = {
            'applied': True,
            'timeout': self.cache_timeout,
            'type': self.cache_type,
            'expires_at': time.time() + self.cache_timeout
        }
        
        logger.debug(f"Applied caching to response (timeout: {self.cache_timeout}s)")
        return response
    
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get cache decorator information."""
        return {
            'type': 'cache',
            'timeout': self.cache_timeout,
            'cache_type': self.cache_type
        }


class SecurityDecorator(ViewDecorator):
    """
    Decorator for security headers.
    
    Concrete Decorator: Adds security headers to responses.
    """
    
    def __init__(self, enable_csp: bool = True, enable_xss_protection: bool = True):
        """
        Initialize security decorator.
        
        Args:
            enable_csp: Enable Content Security Policy
            enable_xss_protection: Enable XSS protection headers
        """
        self.enable_csp = enable_csp
        self.enable_xss_protection = enable_xss_protection
        
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply security headers to response.
        
        Adds various security headers to protect against common attacks.
        """
        if 'headers' not in context:
            context['headers'] = {}
        
        headers = context['headers']
        
        # Add security headers
        if self.enable_xss_protection:
            headers['X-XSS-Protection'] = '1; mode=block'
            headers['X-Content-Type-Options'] = 'nosniff'
            headers['X-Frame-Options'] = 'DENY'
        
        if self.enable_csp:
            headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        
        context['security'] = {
            'applied': True,
            'csp_enabled': self.enable_csp,
            'xss_protection_enabled': self.enable_xss_protection
        }
        
        logger.debug("Applied security headers to response")
        return response
    
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get security decorator information."""
        return {
            'type': 'security',
            'csp_enabled': self.enable_csp,
            'xss_protection_enabled': self.enable_xss_protection
        }


class PerformanceDecorator(ViewDecorator):
    """
    Decorator for performance monitoring.
    
    Concrete Decorator: Adds performance timing information to responses.
    """
    
    def __init__(self, include_timing: bool = True, include_size: bool = True):
        """
        Initialize performance decorator.
        
        Args:
            include_timing: Include response timing information
            include_size: Include response size information
        """
        self.include_timing = include_timing
        self.include_size = include_size
        self.start_time = time.time()
        
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply performance monitoring to response.
        
        Adds performance metrics to the response context.
        """
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        performance_info = {}
        
        if self.include_timing:
            performance_info['processing_time'] = processing_time
            performance_info['start_time'] = self.start_time
            performance_info['end_time'] = end_time
        
        if self.include_size:
            performance_info['response_size'] = len(response)
            performance_info['response_size_kb'] = len(response) / 1024
        
        context['performance'] = performance_info
        
        # Add performance header
        if 'headers' not in context:
            context['headers'] = {}
        
        context['headers']['X-Response-Time'] = f"{processing_time:.3f}s"
        
        logger.debug(f"Applied performance monitoring (time: {processing_time:.3f}s)")
        return response
    
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get performance decorator information."""
        return {
            'type': 'performance',
            'include_timing': self.include_timing,
            'include_size': self.include_size
        }


class CompositeDecorator(ViewDecorator):
    """
    Composite decorator for chaining multiple decorators.
    
    Composite Pattern: Allows treating multiple decorators as a single decorator.
    """
    
    def __init__(self, decorators: list[ViewDecorator]):
        """
        Initialize composite decorator.
        
        Args:
            decorators: List of decorators to apply in order
        """
        self.decorators = decorators
        
    def decorate_response(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply all decorators in sequence.
        
        Chain of Responsibility Pattern: Each decorator processes the response.
        """
        current_response = response
        
        for decorator in self.decorators:
            current_response = decorator.decorate_response(current_response, context)
        
        return current_response
    
    def get_decorator_info(self) -> Dict[str, Any]:
        """Get composite decorator information."""
        return {
            'type': 'composite',
            'decorators': [dec.get_decorator_info() for dec in self.decorators]
        }
    
    def add_decorator(self, decorator: ViewDecorator):
        """Add a decorator to the chain."""
        self.decorators.append(decorator)
    
    def remove_decorator(self, decorator_type: str):
        """Remove decorators of a specific type."""
        self.decorators = [
            dec for dec in self.decorators 
            if dec.get_decorator_info().get('type') != decorator_type
        ]


# Factory functions for common decorator combinations
def create_basic_decorators() -> CompositeDecorator:
    """Create basic decorator chain for most responses."""
    return CompositeDecorator([
        SecurityDecorator(),
        PerformanceDecorator()
    ])


def create_cached_decorators(cache_timeout: int = 300) -> CompositeDecorator:
    """Create decorator chain with caching enabled."""
    return CompositeDecorator([
        SecurityDecorator(),
        CacheDecorator(cache_timeout=cache_timeout),
        CompressionDecorator(),
        PerformanceDecorator()
    ])


def create_api_decorators() -> CompositeDecorator:
    """Create decorator chain optimized for API responses."""
    return CompositeDecorator([
        SecurityDecorator(enable_csp=False),  # Less restrictive for APIs
        CompressionDecorator(min_size=500),   # Compress smaller responses
        PerformanceDecorator()
    ]) 