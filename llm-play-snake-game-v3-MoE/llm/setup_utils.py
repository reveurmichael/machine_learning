"""
Setup utilities for the Snake game.
Handles LLM client setup and health checks to avoid cyclic imports.
"""

import sys
import os
from colorama import Fore

def check_env_setup(provider):
    """Check if the environment is properly set up for the selected provider.
    
    Args:
        provider: The selected LLM provider
        
    Returns:
        Boolean indicating if the setup is likely to work
    """
    # Skip for ollama as it can work without env variables
    if provider.lower() == 'ollama':
        # Check if OLLAMA_HOST is set
        if os.environ.get('OLLAMA_HOST'):
            print(Fore.GREEN + f"✅ Using custom Ollama host: {os.environ.get('OLLAMA_HOST')}")
        else:
            print(Fore.YELLOW + "⚠️ Using default Ollama host (localhost:11434). Set OLLAMA_HOST in .env if needed.")
        return True
        
    # Check for .env file
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if not os.path.exists(env_file):
        print(Fore.RED + f"❌ No .env file found at {env_file}")
        print(Fore.YELLOW + "Please create a .env file with your API keys. Example:")
        print(Fore.CYAN + """
HUNYUAN_API_KEY=your_hunyuan_api_key_here
OLLAMA_HOST=your_ollama_host_ip_address
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
        """)
        return False
        
    # Check for specific provider keys
    key_found = False
    if provider.lower() == 'hunyuan' and os.environ.get('HUNYUAN_API_KEY'):
        key_found = True
        print(Fore.GREEN + "✅ HUNYUAN_API_KEY found in environment")
    elif provider.lower() == 'deepseek' and os.environ.get('DEEPSEEK_API_KEY'):
        key_found = True
        print(Fore.GREEN + "✅ DEEPSEEK_API_KEY found in environment")
    elif provider.lower() == 'mistral' and os.environ.get('MISTRAL_API_KEY'):
        key_found = True
        print(Fore.GREEN + "✅ MISTRAL_API_KEY found in environment")
        
    if not key_found:
        print(Fore.YELLOW + f"⚠️ No API key found for {provider}. Make sure to set it in your .env file")
        return False
        
    return True

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
        parser_model = game_manager.args.parser_model or game_manager.args.model  # Default to primary model if parser model not specified
        print(Fore.GREEN + f"Using parser LLM model: {parser_model}")
        
        # Set up the secondary LLM in the client
        game_manager.llm_client.set_secondary_llm(game_manager.args.parser_provider, parser_model)
        
        # Verify that the secondary LLM was properly configured
        if not game_manager.llm_client.secondary_provider or not game_manager.llm_client.secondary_model:
            print(Fore.RED + "❌ Failed to configure secondary LLM. Continuing without parser.")
            game_manager.args.parser_provider = "none"
            game_manager.args.parser_model = None
        else:
            # Perform health check for parser LLM
            # Create a temporary client for testing the secondary LLM
            test_client = game_manager.create_llm_client(game_manager.args.parser_provider, parser_model)
            parser_healthy = check_llm_health(test_client)[0]
            
            if not parser_healthy:
                print(Fore.RED + "❌ Parser LLM health check failed. Continuing without parser.")
                game_manager.args.parser_provider = "none"
                game_manager.args.parser_model = None
                # Clear secondary LLM configuration to avoid confusion
                game_manager.llm_client.secondary_provider = None
                game_manager.llm_client.secondary_model = None
            else:
                print(Fore.GREEN + "✅ Parser LLM health check passed!")
    else:
        print(Fore.YELLOW + "⚠️ No parser LLM specified. Using primary LLM output directly.")
        game_manager.args.parser_provider = "none"
        game_manager.args.parser_model = None
    
    return True 