"""
Ollama provider implementation.
Handles communication with the Ollama API.
"""

import os
import json
import requests
from typing import Dict, Tuple, Optional

from config.llm_constants import TEMPERATURE

from .base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama LLM service."""

    # Populate with concrete model identifiers as desired (dashboard uses it).
    available_models: list[str] = sorted(
        [
            "deepseek-r1:1.5b",
            "deepseek-r1:14b",
            "deepseek-r1:32b",
            "deepseek-r1:7b",
            "deepseek-r1:8b",
            "devstral:24b",
            "gemma2:27b",
            "gemma2:9b",
            "gemma3:12b-it-qat",
            "gemma3:12b",
            "gemma3:27b-it-qat",
            "gemma3:27b",
            "gemma3:4b-it-qat",
            "llama3.1:latest",
            "magistral:24b",
            "mistral:7b",
            "qwen3:8b"
        ]
    )

    def __init__(self):
        """Initialize the Ollama provider."""
        # Default host is set from environment or default to localhost
        self.server = os.environ.get("OLLAMA_HOST", "localhost")

    def get_default_model(self) -> str:
        """Get the default model for Ollama.
        
        Returns:
            The name of the default model
        """
        return "deepseek-r1:7b"

    # Inherit default validate_model from BaseProvider (allows any model if list empty)

    def generate_response(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate a response from Ollama.

        Args:
            prompt: The prompt to send to the LLM
            model: The specific model to use
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple of (response_text, token_count)
        """
        try:
            # Get server from kwargs or use the default
            server = kwargs.get("server", self.server)
            # Extract parameters
            temperature = kwargs.get("temperature", TEMPERATURE)

            # Set default model if none provided
            model = model or self.get_default_model()

            print(f"Making API call to Ollama at http://{server}:11434 with model: {model}, temperature: {temperature}")

            # Make the API call directly with requests
            response = requests.post(
                f"http://{server}:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                },
            )

            # Check if response is valid JSON
            try:
                response_json = response.json()

                # Extract token counts if available
                token_count = None
                if 'prompt_eval_count' in response_json and 'eval_count' in response_json:
                    prompt_tokens = response_json.get('prompt_eval_count', 0)
                    completion_tokens = response_json.get('eval_count', 0)
                    token_count = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    }

                # Get the response text
                response_text = response_json.get("response", "ERROR: No response field in JSON")

                return response_text, token_count

            except json.JSONDecodeError:
                error_message = f"ERROR: Invalid JSON response - {response.text[:100]}"
                print(error_message)
                return error_message, None

        except requests.exceptions.ConnectionError:
            error_message = f"Connection error to Ollama server at {server}"
            print(error_message)
            return f"ERROR LLMCLIENT: {error_message}", None

        except Exception as e:
            return self.handle_error(e) 
