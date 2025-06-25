"""
Hunyuan provider implementation.
Handles communication with the Tencent Hunyuan API.
"""

import os
import traceback
from typing import Dict, Tuple, Optional, Any, cast
from openai import OpenAI

from config.llm_constants import TEMPERATURE, MAX_TOKENS

from .base_provider import BaseProvider


class HunyuanProvider(BaseProvider):
    """Provider implementation for Tencent Hunyuan LLM service."""

    available_models: list[str] = sorted([
        "hunyuan-turbos-latest",
        "hunyuan-t1-latest",
    ])

    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model for Hunyuan.

        Returns:
            The name of the default model
        """
        return "hunyuan-turbos-latest"

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

            # Ensure model is a concrete string (Deep type check)
            model = model or self.get_default_model()

            # Create the message
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

            # Extract parameters - use the provided model (should be set already)
            temperature = kwargs.get("temperature", TEMPERATURE)
            max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
            enable_enhancement = kwargs.get("enable_enhancement", True)

            print(f"Making API call to Hunyuan with model: {model}, temperature: {temperature}")

            # Make the API call
            try:
                completion = client.chat.completions.create(
                    model=cast(str, model),
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "enable_enhancement": enable_enhancement,
                    },
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

                content: str = cast(str, completion.choices[0].message.content or "")
                return content, token_count
            except Exception as api_error:
                print(f"Error during Hunyuan API call: {api_error}")
                return f"ERROR LLMCLIENT: {api_error}", None

        except Exception as exception:
            print(f"Error generating response from Hunyuan: {exception}")
            traceback.print_exc()
            return f"ERROR LLMCLIENT: {exception}", None
