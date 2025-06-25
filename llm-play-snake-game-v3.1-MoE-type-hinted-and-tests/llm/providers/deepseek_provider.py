"""
Deepseek provider implementation.
Handles communication with the Deepseek API.
"""

import os
import traceback
from typing import Dict, Tuple, Optional, Any, cast
from openai import OpenAI

from config.llm_constants import TEMPERATURE, MAX_TOKENS

from .base_provider import BaseProvider


class DeepseekProvider(BaseProvider):
    """Provider implementation for DeepSeek LLM service."""

    available_models: list[str] = sorted([
        "deepseek-chat",
        "deepseek-reasoner",
    ])

    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model for Deepseek.

        Returns:
            The name of the default model
        """
        return "deepseek-chat"

    def generate_response(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate a response from Deepseek LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The specific model to use
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple of (response_text, token_count)
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key or api_key == "your_deepseek_api_key_here":
                print("Warning: Deepseek API key not properly configured in .env file")
                return "ERROR LLMCLIENT: Deepseek API key not properly configured", None

            # Construct OpenAI client for Deepseek
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )

            # Extract parameters
            model = model or self.get_default_model()

            temperature = kwargs.get("temperature", TEMPERATURE)
            max_tokens = kwargs.get("max_tokens", MAX_TOKENS)

            print(f"Using Deepseek model: {model}")

            # Create the messages with system prompt to ensure proper response format
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

            print(f"Making API call to Deepseek with model: {model}, temperature: {temperature}")

            # Make the API call
            try:
                completion = client.chat.completions.create(  # type: ignore[arg-type]
                    model=model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Extract token counts if available
                token_count: Optional[Dict[str, int]] = None
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    token_count = {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }

                # Ensure non-Optional content per function contract
                content: str = cast(str, completion.choices[0].message.content or "")
                return content, token_count
            except Exception as api_error:
                print(f"Error during Deepseek API call: {api_error}")
                return f"ERROR LLMCLIENT: {api_error}", None

        except Exception as exception:
            print(f"Error generating response from Deepseek: {exception}")
            traceback.print_exc()
            return f"ERROR LLMCLIENT: {exception}", None
