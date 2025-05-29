"""
LLM integration system for the Snake game.
Core module that provides a unified interface to all language model functionality,
including prompt generation, response handling, and communication management.
"""

# Import and re-export prompt-related functions
from utils.llm_prompt_utils import (
    prepare_snake_prompt,
    create_parser_prompt,
    format_raw_llm_response
)

# Import and re-export parsing-related functions
from utils.llm_parsing_utils import (
    parse_and_format,
    parse_llm_response,
    handle_llm_response
)

# Import and re-export communication-related functions
from utils.llm_communication_utils import (
    check_llm_health,
    extract_state_for_parser,
    get_llm_response
)

# Make everything available at the module level
__all__ = [
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
    'get_llm_response'
] 