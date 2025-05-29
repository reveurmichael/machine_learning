"""
LLM integration module for the Snake game.
Provides a unified interface to language model functionality including communication,
prompt generation, response parsing, and provider-specific implementations.
"""

# Export client and core functionality
from llm.client import LLMClient

# Import and re-export prompt-related functions
from llm.prompt_utils import (
    prepare_snake_prompt,
    create_parser_prompt,
    format_raw_llm_response
)

# Import and re-export parsing-related functions
from llm.parsing_utils import (
    parse_and_format,
    parse_llm_response,
    handle_llm_response
)

# Import and re-export communication-related functions
from llm.communication_utils import (
    extract_state_for_parser,
    get_llm_response,
    check_llm_health
)

from llm.setup_utils import check_env_setup

# Define the public API
__all__ = [
    # Client
    'LLMClient',
    
    # Prompt utilities
    'prepare_snake_prompt',
    'create_parser_prompt',
    'format_raw_llm_response',
    
    # Parsing utilities
    'parse_and_format',
    'parse_llm_response',
    'handle_llm_response',
    
    # Communication utilities
    'check_llm_health',
    'extract_state_for_parser',
    'get_llm_response',
    
    # Setup utilities
    'check_env_setup'
] 