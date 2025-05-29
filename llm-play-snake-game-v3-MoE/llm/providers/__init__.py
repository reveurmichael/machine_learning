"""
LLM provider implementations.
This module contains provider-specific implementations for different LLM services.

Each provider implements the BaseProvider interface for consistent interaction
across different LLM services.
"""

from .base_provider import BaseProvider
from .ollama_provider import OllamaProvider
from .mistral_provider import MistralProvider
from .hunyuan_provider import HunyuanProvider
from .deepseek_provider import DeepseekProvider

# Factory function to create the appropriate provider
def create_provider(provider_name: str):
    """Create a provider instance by name.
    
    Args:
        provider_name: Name of the provider to create
        
    Returns:
        An instance of the requested provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    providers = {
        "ollama": OllamaProvider,
        "mistral": MistralProvider,
        "hunyuan": HunyuanProvider,
        "deepseek": DeepseekProvider
    }
    
    provider_name = provider_name.lower()
    if provider_name not in providers:
        supported = ", ".join(providers.keys())
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {supported}")
    
    return providers[provider_name]()

__all__ = [
    "BaseProvider",
    "OllamaProvider", 
    "MistralProvider", 
    "HunyuanProvider", 
    "DeepseekProvider",
    "create_provider"
] 