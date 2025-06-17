"""
Mistral provider implementation.
Handles communication with the Mistral API.
"""

import os
import traceback
from typing import Dict, Tuple, Optional
from mistralai import Mistral

from config.llm_constants import TEMPERATURE, MAX_TOKENS

from .base_provider import BaseProvider


class MistralProvider(BaseProvider):
    """Provider implementation for Mistral LLM service."""
    
    available_models: list[str] = sorted([
        "mistral-medium-latest",
        "mistral-large-latest",
    ])
    
    def get_default_model(self) -> str:
        """Get the default model for Mistral.
        
        Returns:
            The name of the default model
        """
        return "mistral-medium-latest"
    
    def generate_response(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate a response from Mistral LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The specific model to use
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple of (response_text, token_count)
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key or api_key == "your_mistral_api_key_here":
                print("Warning: Mistral API key not properly configured in .env file")
                return "ERROR LLMCLIENT: Mistral API key not properly configured", None

            # Extract parameters
            model = model or self.get_default_model()
            # Validate model selection
            model = self.validate_model(model)

            temperature = kwargs.get("temperature", TEMPERATURE)
            max_tokens = kwargs.get("max_tokens", MAX_TOKENS)

            print(f"Using Mistral model: {model}")

            # Create Mistral client
            client = Mistral(api_key=api_key)

            # Create message structure
            messages = [{"role": "user", "content": prompt}]

            print(f"Making API call to Mistral with model: {model}, temperature: {temperature}")

            # Make the API call
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Extract token counts if available
                token_count = None
                if hasattr(chat_response, 'usage'):
                    token_count = {
                        'prompt_tokens': chat_response.usage.prompt_tokens,
                        'completion_tokens': chat_response.usage.completion_tokens,
                        'total_tokens': chat_response.usage.total_tokens
                    }

                # Return the response
                return chat_response.choices[0].message.content, token_count
            except Exception as api_error:
                print(f"Error during Mistral API call: {api_error}")
                return f"ERROR LLMCLIENT: {api_error}", None

        except Exception as e:
            print(f"Error generating response from Mistral: {e}")
            traceback.print_exc()
            return f"ERROR LLMCLIENT: {e}", None 
