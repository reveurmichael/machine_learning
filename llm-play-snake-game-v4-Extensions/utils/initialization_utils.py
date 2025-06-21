"""
Initialization utilities for the Snake game.
Centralised helpers for directory setup, LLM client configuration and initial
pygame/game-state preparation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from colorama import Fore

from llm.communication_utils import check_llm_health
from llm.log_utils import ensure_llm_directories
from utils.path_utils import create_session_dir_path

# ---------------------
# Typing-only imports – avoid heavy dependencies at runtime
# ---------------------

if TYPE_CHECKING:
    from core.game_manager import GameManager

def setup_log_directories(game_manager: "GameManager") -> None:
    """Set up log directories for storing game data.
    
    Args:
        game_manager: The GameManager instance
    """
    # Create session directory if specified
    if game_manager.args.log_dir:
        game_manager.log_dir = game_manager.args.log_dir
    else:
        # Create timestamped log directory with model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get model name for directory
        model_name = "unknown"
        if game_manager.args.model:
            # Clean up model name for directory use
            model_name = game_manager.args.model.replace("/", "-").replace(":", "-")
        elif game_manager.args.provider:
            model_name = game_manager.args.provider
            
        # Create directory path with model name and timestamp
        game_manager.log_dir = str(create_session_dir_path(model_name, timestamp))
    
    # Set up LLM-specific directories using centralized utilities
    prompts_dir, responses_dir = ensure_llm_directories(game_manager.log_dir)
    game_manager.prompts_dir = str(prompts_dir)
    game_manager.responses_dir = str(responses_dir)
    
    # Create all directories
    os.makedirs(game_manager.log_dir, exist_ok=True)
    os.makedirs(game_manager.prompts_dir, exist_ok=True)
    os.makedirs(game_manager.responses_dir, exist_ok=True)
    
    print(Fore.GREEN + f"📁 Session directory created: {game_manager.log_dir}")

def setup_llm_clients(game_manager: "GameManager") -> None:
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

def initialize_game_state(game_manager: "GameManager") -> None:
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

def enforce_launch_sleep(args) -> None:  # type: ignore[valid-type]
    """Apply the --sleep-before-launching delay (in minutes) if set.

    This helper is reused by fresh and continuation sessions so the behaviour
    resides in a single place. The delay is inserted only after the LLM
    health-check passes to avoid wasting time when credentials or network
    connectivity are wrong.
    
    Args:
        args: Command line arguments containing sleep_before_launching setting
    """
    minutes: float = getattr(args, "sleep_before_launching", 0)
    if minutes and minutes > 0:
        from time import sleep

        plural = "s" if minutes != 1 else ""
        print(Fore.YELLOW + f"💤 Sleeping for {minutes} minute{plural} before launching…")
        sleep(minutes * 60)
        print(Fore.GREEN + "⏰ Waking up and starting the program…")
