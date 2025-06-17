"""Top-level LLM package initialisation.

This file deliberately keeps imports minimal to avoid circular-import
issues (e.g. `config` → `llm` → `config`). Down-stream modules should
import the helpers they need directly (`from llm.prompt_utils import …`).
Only the most-commonly used APIs are re-exported here.
"""

from importlib import import_module

# Core runtime client
from llm.client import LLMClient  # noqa: F401  (public API)

# Lightweight pass-throughs from provider registry
from llm.providers import (  # noqa: F401
    create_provider,
    list_providers,
    get_available_models,
)


def __getattr__(name):  # pragma: no cover – lazy imports
    """Lazy import heavy sub-modules on first access to break cycles."""

    mapping = {
        # prompt utilities
        "prepare_snake_prompt": "llm.prompt_utils",
        "create_parser_prompt": "llm.prompt_utils",
        # parsing utilities
        "parse_and_format": "llm.parsing_utils",
        "parse_llm_response": "llm.parsing_utils",
        # communication helpers
        "extract_state_for_parser": "llm.communication_utils",
        "get_llm_response": "llm.communication_utils",
        "check_llm_health": "llm.communication_utils",
        # setup helper
        "check_env_setup": "llm.setup_utils",
    }

    if name in mapping:
        module = import_module(mapping[name])
        return getattr(module, name)
    raise AttributeError(f"module 'llm' has no attribute '{name}'")


# Public API (only statically available names)
__all__ = [
    "LLMClient",
    "create_provider",
    "list_providers",
    "get_available_models",
] 