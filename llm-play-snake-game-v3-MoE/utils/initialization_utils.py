"""
Initialization utilities for the Snake game.
Handles initial setup, game configuration, and data loading.
"""

import os
import json
from colorama import Fore
from llm.communication_utils import check_llm_health
import sys

def setup_log_directories(game_manager):
    """Set up log directories for storing game data.
    
    Args:
        game_manager: The GameManager instance
    """
    # Create session directory if specified
    if game_manager.args.log_dir:
        game_manager.log_dir = game_manager.args.log_dir
    else:
        # Create timestamped log directory with model name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get model name for directory
        model_name = "unknown"
        if game_manager.args.model:
            # Clean up model name for directory use
            model_name = game_manager.args.model.replace("/", "-").replace(":", "-")
        elif game_manager.args.provider:
            model_name = game_manager.args.provider
            
        # Create directory path with model name and timestamp
        game_manager.log_dir = os.path.join("logs", f"{model_name}_{timestamp}")
    
    # Create subdirectories for detailed logs
    game_manager.prompts_dir = os.path.join(game_manager.log_dir, "prompts")
    game_manager.responses_dir = os.path.join(game_manager.log_dir, "responses")
    
    # Create all directories
    os.makedirs(game_manager.log_dir, exist_ok=True)
    os.makedirs(game_manager.prompts_dir, exist_ok=True)
    os.makedirs(game_manager.responses_dir, exist_ok=True)
    
    print(Fore.GREEN + f"📁 Session directory created: {game_manager.log_dir}")

def setup_llm_clients(game_manager):
    """Set up the LLM clients for the game.
    
    Args:
        game_manager: The GameManager instance
    """
    # Set up primary LLM client
    provider = game_manager.args.provider
    model = game_manager.args.model
    
    # Create the client with the explicitly set model name
    game_manager.llm_client = game_manager.create_llm_client(provider, model)
    print(Fore.GREEN + f"Primary LLM: {provider} ({model})")

    # Check if primary LLM is operational
    is_healthy, _ = check_llm_health(game_manager.llm_client)
    if not is_healthy:
        print(Fore.RED + "❌ Primary LLM health check failed. Exiting.")
        sys.exit(1)

    # Set up parser LLM client if needed
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
        game_manager.parser_provider = game_manager.args.parser_provider
        game_manager.parser_model = game_manager.args.parser_model

        # Configure the secondary LLM in the main client
        success = game_manager.llm_client.set_secondary_llm(game_manager.parser_provider, game_manager.parser_model)

        if success:
            # Create a separate client for health check
            parser_client = game_manager.create_llm_client(game_manager.parser_provider, game_manager.parser_model)

            # Check if parser LLM is operational
            is_healthy, _ = check_llm_health(parser_client)

            if not is_healthy:
                print(Fore.RED + "❌ Parser LLM health check failed. Exiting.")
                sys.exit(1)
        else:
            print(Fore.RED + "❌ Failed to configure secondary LLM. Continuing without parser.")
            game_manager.args.parser_provider = "none"
            game_manager.args.parser_model = None
    else:
        # Clear parser fields to ensure they're not included in logs
        game_manager.args.parser_model = None
        print(Fore.GREEN + "🤖 Using single LLM mode (no secondary parser LLM)")

def initialize_game_state(game_manager):
    """Initialize the game state.
    
    Args:
        game_manager: The GameManager instance
    """
    import pygame
    
    # Initialize pygame if using GUI
    if game_manager.use_gui:
        pygame.init()
        pygame.font.init()
    
    # Set up the game
    game_manager.setup_game()
    