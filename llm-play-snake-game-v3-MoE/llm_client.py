"""
LLM client module for handling communication with different LLM providers.
"""

import os
import json
import time
import requests
import traceback
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from mistralai import Mistral

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, provider: str = "hunyuan", model: str = None):
        """Initialize the LLM client.

        Args:
            provider: The LLM provider to use ("hunyuan", "ollama", "deepseek", or "mistral")
            model: The specific model to use with the provider
        """
        self.provider = provider
        self.model = model
        self.last_token_count = None

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
        
        print(f"Generating response using provider: {self.provider}")
        if self.provider == "hunyuan":
            response = self._generate_hunyuan_response(prompt, **kwargs)
        elif self.provider == "ollama":
            response = self._generate_ollama_response(prompt, **kwargs)
        elif self.provider == "deepseek":
            response = self._generate_deepseek_response(prompt, **kwargs)
        elif self.provider == "mistral":
            response = self._generate_mistral_response(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # Print brief response preview for debugging
        preview = response[:50] + "..." if len(response) > 50 else response
        print(f"Response preview: {preview}")
        
        # Print token usage if available
        if self.last_token_count:
            print(f"Token usage: {self.last_token_count}")

        return response

    def _generate_hunyuan_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Tencent Hunyuan LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The LLM's response as a string
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("HUNYUAN_API_KEY")
            if not api_key or api_key == "your_hunyuan_api_key_here":
                print("Warning: Hunyuan API key not properly configured in .env file")
                return "ERROR LLMCLIENT"

            # Construct OpenAI client for Hunyuan
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.hunyuan.cloud.tencent.com/v1",
            )

            # Create the message
            messages = [{"role": "user", "content": prompt}]

            model = kwargs.get("model", self.model) or "hunyuan-turbos-latest"
            temperature = kwargs.get(
                "temperature", 0.2
            )  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)
            enable_enhancement = kwargs.get("enable_enhancement", True)

            print(
                f"Making API call to Hunyuan with model: {model}, temperature: {temperature}"
            )

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
                
                # Store token usage information
                if hasattr(completion, 'usage'):
                    self.last_token_count = {
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'completion_tokens': completion.usage.completion_tokens,
                        'total_tokens': completion.usage.total_tokens
                    }

                # Return the response
                return completion.choices[0].message.content
            except Exception as api_error:
                print(f"Error during Hunyuan API call: {api_error}")
                return "ERROR LLMCLIENT"

        except Exception as e:
            print(f"Error generating response from Hunyuan: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"

    def _generate_deepseek_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Deepseek LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The LLM's response as a string
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key or api_key == "your_deepseek_api_key_here":
                print("Warning: Deepseek API key not properly configured in .env file")
                return "ERROR LLMCLIENT"

            # Construct OpenAI client for Deepseek
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )

            # Extract parameters
            model = kwargs.get("model", "deepseek-chat")  # Default to deepseek-chat
            # Validate model selection
            if model not in ["deepseek-chat", "deepseek-reasoner"]:
                print(
                    f"Warning: Unknown Deepseek model '{model}', using deepseek-chat instead"
                )
                model = "deepseek-chat"

            temperature = kwargs.get(
                "temperature", 0.2
            )  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)

            print(f"Using Deepseek model: {model}")

            # Create the messages with system prompt to ensure proper response format
            messages = [{"role": "user", "content": prompt}]

            print(
                f"Making API call to Deepseek with model: {model}, temperature: {temperature}"
            )

            # Make the API call
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Store token usage information
                if hasattr(completion, 'usage'):
                    self.last_token_count = {
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'completion_tokens': completion.usage.completion_tokens,
                        'total_tokens': completion.usage.total_tokens
                    }

                # Return the response
                return completion.choices[0].message.content
            except Exception as api_error:
                print(f"Error during Deepseek API call: {api_error}")
                return "ERROR LLMCLIENT"

        except Exception as e:
            print(f"Error generating response from Deepseek: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"

    def _generate_mistral_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Mistral LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The LLM's response as a string
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key or api_key == "your_mistral_api_key_here":
                print("Warning: Mistral API key not properly configured in .env file")
                return "ERROR LLMCLIENT"

            # Extract parameters
            model = kwargs.get(
                "model", "mistral-medium-latest"
            )  # Default to medium model
            # Validate model selection
            valid_models = [
                "mistral-tiny",
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-latest",
            ]
            if model not in valid_models:
                print(
                    f"Warning: Unknown Mistral model '{model}', using mistral-medium-latest instead"
                )
                model = "mistral-medium-latest"

            temperature = kwargs.get(
                "temperature", 0.2
            )  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)

            print(f"Using Mistral model: {model}")

            # Create Mistral client
            client = Mistral(api_key=api_key)

            # Create message structure
            messages = [{"role": "user", "content": prompt}]

            print(
                f"Making API call to Mistral with model: {model}, temperature: {temperature}"
            )

            # Make the API call
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Store token usage information
                if hasattr(chat_response, 'usage'):
                    self.last_token_count = {
                        'prompt_tokens': chat_response.usage.prompt_tokens,
                        'completion_tokens': chat_response.usage.completion_tokens,
                        'total_tokens': chat_response.usage.total_tokens
                    }

                # Return the response
                return chat_response.choices[0].message.content
            except Exception as api_error:
                print(f"Error during Mistral API call: {api_error}")
                return "ERROR LLMCLIENT"

        except Exception as e:
            print(f"Error generating response from Mistral: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"

    def _generate_ollama_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Ollama LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The LLM's response as a string
        """
        try:
            server = kwargs.get("server", os.environ.get("OLLAMA_HOST", "localhost"))
            temperature = kwargs.get("temperature", 0.2)

            model = kwargs.get("model", self.model)
            if not model:
                model = "deepseek-r1:7b"

            print(
                f"Making API call to Ollama with model: {model}, temperature: {temperature}"
            )

            # Make the API call
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
                
                # Store token usage information if available
                if 'prompt_eval_count' in response_json and 'eval_count' in response_json:
                    prompt_tokens = response_json.get('prompt_eval_count', 0)
                    completion_tokens = response_json.get('eval_count', 0)
                    self.last_token_count = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    }
                
                return response_json.get("response", "ERROR: No response field in JSON")
            except json.JSONDecodeError:
                return f"ERROR: Invalid JSON response - {response.text[:100]}"

        except requests.exceptions.Timeout:
            print(f"Timeout error connecting to Ollama server at {server}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"
        except requests.exceptions.ConnectionError:
            print(f"Connection error to Ollama server at {server}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT"
