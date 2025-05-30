"""
Hunyuan provider implementation.
Handles communication with the Tencent Hunyuan API.
"""

import os
import traceback
from typing import Dict, Tuple, Optional
from openai import OpenAI

from .base_provider import BaseProvider


class HunyuanProvider(BaseProvider):
    """Provider implementation for Tencent Hunyuan LLM service."""
    
    def get_default_model(self) -> str:
        """Get the default model for Hunyuan.
        
        Returns:
            The name of the default model
        """
        return "hunyuan-turbos-latest"
    
    def validate_model(self, model: str) -> str:
        """Validate the model name for Hunyuan.
        
        Args:
            model: The model name to validate
            
        Returns:
            The validated model name
        """
        # Currently, Hunyuan has limited models, but we don't validate here
        return model
    
    def generate_response(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate a response from Hunyuan.

        Args:
            prompt: The prompt to send to the LLM
            model: The specific model to use
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple of (response_text, token_count)
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("HUNYUAN_API_KEY")
            if not api_key or api_key == "your_hunyuan_api_key_here":
                print("Warning: Hunyuan API key not properly configured in .env file")
                return "ERROR LLMCLIENT: Hunyuan API key not properly configured", None

            # Construct OpenAI client for Hunyuan
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.hunyuan.cloud.tencent.com/v1",
            )

            # Create the message
            messages = [{"role": "user", "content": prompt}]

            # Extract parameters - use the provided model (should be set already)
            temperature = kwargs.get("temperature", 0.2)  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)
            enable_enhancement = kwargs.get("enable_enhancement", True)

            print(f"Making API call to Hunyuan with model: {model}, temperature: {temperature}")

            # Make the API call
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "enable_enhancement": enable_enhancement,
                    },
                )
                
                # Extract token counts if available
                token_count = None
                if hasattr(completion, 'usage'):
                    token_count = {
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'completion_tokens': completion.usage.completion_tokens,
                        'total_tokens': completion.usage.total_tokens
                    }

                # Return the response
                return completion.choices[0].message.content, token_count
            except Exception as api_error:
                print(f"Error during Hunyuan API call: {api_error}")
                return f"ERROR LLMCLIENT: {api_error}", None

        except Exception as e:
            print(f"Error generating response from Hunyuan: {e}")
            traceback.print_exc()
            return f"ERROR LLMCLIENT: {e}", None 