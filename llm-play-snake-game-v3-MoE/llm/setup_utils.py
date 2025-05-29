"""
Setup utilities for LLM configuration in the Snake game.
Handles environment validation for LLM providers.
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