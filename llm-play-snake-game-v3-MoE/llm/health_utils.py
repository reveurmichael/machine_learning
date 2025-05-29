"""
LLM health checking utilities.
Functions for verifying that LLM providers are accessible and responding correctly.
"""

import time
import traceback
from colorama import Fore

def check_llm_health(llm_client, max_retries=2, retry_delay=2):
    """Check if the LLM is accessible and responding by sending a simple test query.
    
    Args:
        llm_client: The LLM client to check
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        A tuple containing (is_healthy, response) where:
          - is_healthy: Boolean indicating if the LLM is healthy
          - response: The response from the LLM or an error message
    """
    test_prompt = "Hello, are you there? Please respond with 'Yes, I am here.'"
    
    for attempt in range(max_retries):
        try:
            print(f"Health check attempt {attempt+1}/{max_retries} for {llm_client.provider} LLM...")
            response = llm_client.generate_response(test_prompt)
            
            # Check if we got a response that contains the expected text or something reasonable
            if response and isinstance(response, str) and len(response) > 5 and "ERROR" not in response:
                print(Fore.GREEN + f"✅ {llm_client.provider}/{llm_client.model} LLM health check passed!")
                return True, response
            
            print(Fore.YELLOW + f"⚠️ {llm_client.provider}/{llm_client.model} LLM returned an unusual response: {response}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            error_msg = f"Error connecting to {llm_client.provider}/{llm_client.model} LLM: {str(e)}"
            print(Fore.RED + f"❌ {error_msg}")
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    return False, f"Failed to get a valid response from {llm_client.provider}/{llm_client.model} LLM after {max_retries} attempts" 