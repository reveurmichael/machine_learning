"""
LLM provider implementations.
This module contains provider-specific implementations for different LLM services.

Each provider implements the BaseProvider interface for consistent interaction
across different LLM services.
"""

from .base_provider import BaseProvider

# Concrete provider classes
from .ollama_provider import OllamaProvider
from .mistral_provider import MistralProvider
from .hunyuan_provider import HunyuanProvider
from .deepseek_provider import DeepseekProvider

# ------------------------------------------------------------
# Provider registry – SINGLE SOURCE OF TRUTH
# ------------------------------------------------------------

_PROVIDER_REGISTRY = {
    "ollama": OllamaProvider,
    "mistral": MistralProvider,
    "hunyuan": HunyuanProvider,
    "deepseek": DeepseekProvider,
}


# ------------------------------------------------------------
# Helper API
# ------------------------------------------------------------


def get_provider_cls(name: str) -> type[BaseProvider]:
    """Return provider *class* (not instance). Raises ValueError if unknown."""

    try:
        return _PROVIDER_REGISTRY[name.lower()]
    except KeyError:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"Unsupported provider '{name}'. Supported: {supported}") from None


def create_provider(name: str) -> BaseProvider:
    """Instantiate provider by *name*. Uses registry."""

    return get_provider_cls(name)()


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""

    return sorted(_PROVIDER_REGISTRY)


def get_available_models(name: str) -> list[str]:
    """Return the ``available_models`` list for *provider* (may be empty)."""

    cls = get_provider_cls(name)
    return cls.get_available_models()


__all__ = [
    "BaseProvider",
    "OllamaProvider",
    "MistralProvider",
    "HunyuanProvider",
    "DeepseekProvider",
    # public API helpers
    "create_provider",
    "get_provider_cls",
    "list_providers",
    "get_available_models",
] 