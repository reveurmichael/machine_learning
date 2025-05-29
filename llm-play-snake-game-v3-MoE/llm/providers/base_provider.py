"""
Base provider interface for LLM services.
Defines the common interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseProvider(ABC):
    """Base class for all LLM provider implementations."""
    
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
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            The name of the default model
        """
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> str:
        """Validate and potentially correct the model name.
        
        Args:
            model: The model name to validate
            
        Returns:
            The validated (and potentially corrected) model name
        """
        pass

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