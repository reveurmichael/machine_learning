"""
LLM client module for handling communication with different LLM providers.
"""

import traceback
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from llm.providers import create_provider, get_provider_cls
# type: ignore[import-untyped]  # Install types-colorama for stubs
from colorama import Fore, init as init_colorama  # type: ignore[import-untyped]

# Load environment variables from .env file
load_dotenv()

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)


class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, provider: str = "hunyuan", model: Optional[str] = None):
        """Initialize the LLM client.

        Args:
            provider: The LLM provider to use ("hunyuan", "ollama", "deepseek", or "mistral")
            model: The specific model to use with the provider
        """
        self.provider = provider.lower()
        self.model: Optional[str] = model
        self.last_token_count: Optional[Dict[str, Any]] = None
        self.secondary_provider: Optional[str] = None
        self.secondary_model: Optional[str] = None

        # Initialize primary provider
        self._provider_instance = create_provider(self.provider)

    def _extract_usage(self, raw_usage: dict) -> dict:
        """
        Return {'prompt_tokens': int|None, 'completion_tokens': int|None,
                'total_tokens': int|None} with NO default values.

        Args:
            raw_usage: The raw usage dictionary from the LLM provider

        Returns:
            Normalized usage dictionary with no default values
        """
        if not raw_usage:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return {
            "prompt_tokens": raw_usage.get("prompt_tokens"),
            "completion_tokens": raw_usage.get("completion_tokens"),
            "total_tokens": raw_usage.get("total_tokens")
        }

    def set_secondary_llm(self, provider: str, model: str):
        """Set the secondary LLM (parser) details.

        Args:
            provider: The LLM provider to use for the secondary LLM
            model: The specific model to use with the provider

        Returns:
            True if successfully configured, False otherwise
        """
        if not provider or provider.lower() == 'none' or not model:
            print(Fore.YELLOW + "Warning: Cannot configure secondary LLM with invalid provider or model")
            self.secondary_provider = None
            self.secondary_model = None
            return False

        # Initialize secondary provider
        try:
            provider = provider.lower()
            temp_instance = create_provider(provider)
            validated_model = temp_instance.validate_model(model)

            self.secondary_provider = provider
            self.secondary_model = validated_model
            print(Fore.GREEN + f"Secondary LLM configured: {provider}/{validated_model}")
            return True
        except ValueError as exception:
            print(Fore.YELLOW + f"Warning: {exception}")
            self.secondary_provider = None
            self.secondary_model = None
            return False

    def generate_text_with_secondary_llm(self, prompt: str, **kwargs) -> str:
        """Generate a response from the secondary LLM.

        Args:
            prompt: The prompt to send to the secondary LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The secondary LLM's response as a string
        """
        if not self.secondary_provider or not self.secondary_model:
            print(Fore.YELLOW + "Warning: Secondary LLM not configured properly")
            return "ERROR: Secondary LLM not configured"

        # Save current provider and model
        original_provider = self.provider
        original_model = self.model

        try:
            # Temporarily switch to secondary LLM
            from typing import cast
            self.provider = cast(str, self.secondary_provider)
            self.model = cast(str, self.secondary_model)

            # Generate response using the secondary LLM
            response = self.generate_response(prompt, **kwargs)

            return response
        finally:
            # Restore original provider and model
            self.provider = original_provider
            self.model = original_model

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The LLM's response as a string
        """
        # Reset token count for new request
        self.last_token_count = None

        print(Fore.BLUE + f"Generating response using provider: {self.provider}")

        try:
            # Ensure we have the correct provider instance (lazy re-create on provider change)
            if (
                not hasattr(self, "_provider_instance")
                or self._provider_instance is None
                or self._provider_instance.__class__ is not get_provider_cls(self.provider)
            ):
                self._provider_instance = create_provider(self.provider)

            # Extract model parameter if provided, otherwise use the instance model
            requested_model = kwargs.pop('model', None) or self.model

            # Validate model (raises ValueError on unsupported names)
            if requested_model:
                validated_model = get_provider_cls(self.provider).validate_model(requested_model)
                kwargs['model'] = validated_model
                # Keep self.model in sync
                self.model = validated_model
                print(f"Using model: {validated_model}")

            # Call the provider's generate_response method
            response, token_count = self._provider_instance.generate_response(prompt, **kwargs)

            # Store token usage information using the helper to avoid default values
            if token_count:
                self.last_token_count = self._extract_usage(token_count)

            # Print brief response preview for debugging
            preview = response[:100] + "..." if len(response) > 100 else response
            print(Fore.CYAN + f"Response preview: {preview}")

            # Print token usage if available
            if self.last_token_count:
                print(Fore.MAGENTA + f"Token usage: {self.last_token_count}")

            return response

        except Exception as exception:
            print(Fore.RED + f"Error generating response: {exception}")
            traceback.print_exc()
            return f"ERROR LLMCLIENT: {exception}"
