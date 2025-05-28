"""
Utils package initialization.
This file exposes all functions, classes, and variables from utility modules.
"""

import sys
from colorama import Fore

# Import and expose all from json_utils
from .json_utils import (
    get_json_error_stats,
    reset_json_error_stats,
    extract_valid_json,
    preprocess_json_string,
    validate_json_format,
    extract_json_from_code_block,
    extract_json_from_text,
    extract_moves_pattern,
    extract_moves_from_arrays,
    save_experiment_info_json,
    update_experiment_info_json
)

# Import and expose all from file_utils first (because log_utils depends on it)
from .file_utils import (
    find_log_folders,
    extract_game_stats,
    extract_game_summary,
    get_next_game_number,
    clean_prompt_files,
    save_to_file
)

# Delay importing the remaining modules to avoid cyclic imports
# They'll be imported on-demand when needed

# Make it easy to import from specific modules
__all__ = [
    # json_utils
    'get_json_error_stats',
    'reset_json_error_stats',
    'extract_valid_json',
    'preprocess_json_string',
    'validate_json_format',
    'extract_json_from_code_block',
    'extract_json_from_text',
    'extract_moves_pattern',
    'extract_moves_from_arrays',
    'save_experiment_info_json',
    'update_experiment_info_json',
    
    # file_utils
    'find_log_folders',
    'extract_game_stats',
    'extract_game_summary',
    'get_next_game_number',
    'clean_prompt_files',
    'save_to_file',
    
    # setup_llm_clients function
    'setup_llm_clients'
]

def setup_llm_clients(game_manager, check_llm_health):
    """Set up the LLM clients with health checks.
    
    Args:
        game_manager: The GameManager instance
        check_llm_health: Function to check LLM health
        
    Returns:
        Boolean indicating if setup was successful
    """
    # Initialize primary LLM client
    game_manager.llm_client = game_manager.create_llm_client(
        game_manager.args.provider, 
        game_manager.args.model
    )
    
    print(Fore.GREEN + f"Using primary LLM provider: {game_manager.args.provider}")
    if game_manager.args.model:
        print(Fore.GREEN + f"Using primary LLM model: {game_manager.args.model}")
    
    # Perform health check for primary LLM
    primary_healthy = check_llm_health(game_manager.llm_client)[0]
    if not primary_healthy:
        print(Fore.RED + "❌ Primary LLM health check failed. The program cannot continue.")
        sys.exit(1)
    else:
        print(Fore.GREEN + "✅ Primary LLM health check passed!")
    
    # Configure secondary LLM (parser) if specified
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
        print(Fore.GREEN + f"Using parser LLM provider: {game_manager.args.parser_provider}")
        parser_model = game_manager.args.parser_model
        print(Fore.GREEN + f"Using parser LLM model: {parser_model}")
        
        # Set up the secondary LLM in the client
        game_manager.llm_client.set_secondary_llm(game_manager.args.parser_provider, parser_model)
        
        # Perform health check for parser LLM
        parser_healthy = check_llm_health(
            game_manager.create_llm_client(game_manager.args.parser_provider, parser_model)
        )[0]
        if not parser_healthy:
            print(Fore.RED + "❌ Parser LLM health check failed. Continuing without parser.")
            game_manager.args.parser_provider = "none"
            game_manager.args.parser_model = None
    else:
        print(Fore.YELLOW + "⚠️ No parser LLM specified. Using primary LLM output directly.")
        game_manager.args.parser_provider = "none"
        game_manager.args.parser_model = None
    
    return True
