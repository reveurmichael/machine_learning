"""Top-level LLM package initialisation.

This file deliberately keeps imports minimal to avoid circular-import
issues (e.g. `config` → `llm` → `config`). Down-stream modules should
import the helpers they need directly (`from llm.prompt_utils import …`).
Only the most-commonly used APIs are re-exported here.
"""

from importlib import import_module

# Core runtime client
from llm.client import LLMClient  # noqa: F401  (public API)
# Task-0 agent wrapper – imported at top-level so static analysers recognise
# it as part of the public API even if not yet referenced by the live game
# loop.  Future refactors will instantiate this via the generic *agent*
# pathway in :pymod:`core.game_loop`.
from llm.agent_llm import LLMSnakeAgent  # noqa: F401

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
    "LLMSnakeAgent",
    "create_provider",
    "list_providers",
    "get_available_models",
] 