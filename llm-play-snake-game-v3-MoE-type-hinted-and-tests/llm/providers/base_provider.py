"""
Base provider interface for LLM services.
Defines the common interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseProvider(ABC):
    """Base class for all LLM provider implementations."""

    # Concrete subclasses should populate this list with the *visible* model
    # identifiers they support.  The dashboard UI pulls from it to populate
    # the model select-box.  Leave empty if you don't want to expose any
    # predefined options – the default model will still be shown.
    available_models: list[str] = []

    @abstractmethod
    def generate_response(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The specific model to use
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple of (response_text, token_count_dict)
            where token_count_dict may contain:
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                - total_tokens: Total number of tokens used
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_model(cls) -> str:
        """Get the default model for this provider.

        Returns:
            The name of the default model
        """
        pass

    @classmethod
    def validate_model(cls, model: str) -> str:  # noqa: D401 – simple validation
        """Return *model* if supported; raise ValueError otherwise.

        Concrete providers may override for more complex rules.  If
        ``available_models`` is empty we accept any input (open set).
        """

        if not cls.available_models:
            return model

        if model in cls.available_models:
            return model

        raise ValueError(
            f"Unsupported model '{model}' for provider {cls.__name__.replace('Provider','').lower()}. "
            f"Supported models: {', '.join(cls.available_models)}"
        )

    def handle_error(self, error: Exception) -> Tuple[str, None]:
        """Standard error handling for all providers.

        Args:
            error: The exception that occurred

        Returns:
            Tuple of (error_message, None)
        """
        import traceback
        error_message = f"ERROR LLMCLIENT: {str(error)}"
        print(f"Error generating response: {error}")
        traceback.print_exc()
        return error_message, None

    @classmethod
    def get_available_models(cls) -> list[str]:  # noqa: D401 – simple accessor
        """Return a list of supported model identifiers for this provider."""
        return cls.available_models or [cls.get_default_model()]
