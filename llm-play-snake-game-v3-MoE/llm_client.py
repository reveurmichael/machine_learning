"""
LLM client module for the Snake game.
Provides interfaces for interacting with language models.
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
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, provider="openai", model=None):
        """Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name to use
        """
        self.provider = provider.lower()
        self.model = model
        
        # Set up API keys
        if self.provider == "openai":
            openai.api_key = OPENAI_API_KEY
        elif self.provider == "anthropic":
            anthropic.api_key = ANTHROPIC_API_KEY
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(self, prompt, **kwargs):
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            The LLM's response text
        """
        if self.provider == "openai":
            return self._generate_openai_response(prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic_response(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai_response(self, prompt, **kwargs):
        """Generate a response using OpenAI.
        
        Args:
            prompt: The prompt to send to OpenAI
            **kwargs: Additional arguments for the OpenAI call
            
        Returns:
            The OpenAI response text
        """
        # Set up the model
        model = kwargs.get('model', self.model) or "gpt-3.5-turbo"
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and return the response text
        return response.choices[0].message.content

    def _generate_anthropic_response(self, prompt, **kwargs):
        """Generate a response using Anthropic.
        
        Args:
            prompt: The prompt to send to Anthropic
            **kwargs: Additional arguments for the Anthropic call
            
        Returns:
            The Anthropic response text
        """
        # Set up the model
        model = kwargs.get('model', self.model) or "claude-2"
        
        # Make the API call
        response = anthropic.Client().completion(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            model=model,
            max_tokens_to_sample=1000,
            temperature=0.7
        )
        
        # Extract and return the response text
        return response.completion

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


class LLMOutputParser(LLMClient):
    """LLM client specialized for parsing and formatting responses."""

    def __init__(self, provider="openai", model=None):
        """Initialize the LLM output parser.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name to use
        """
        super().__init__(provider, model)

    def parse_and_format(self, response_text, head_pos, apple_pos, body_cells):
        """Parse and format an LLM response.
        
        Args:
            response_text: The raw LLM response to parse
            head_pos: Current head position
            apple_pos: Current apple position
            body_cells: Current body cell positions
            
        Returns:
            Tuple of (parsed_response, parser_prompt)
        """
        # Create the parser prompt
        parser_prompt = self._create_parser_prompt(
            response_text,
            head_pos,
            apple_pos,
            body_cells
        )
        
        # Get the parsed response
        parsed_response = self.generate_response(parser_prompt)
        
        return parsed_response, parser_prompt

    def _create_parser_prompt(self, response_text, head_pos, apple_pos, body_cells):
        """Create a prompt for the parser LLM.
        
        Args:
            response_text: The raw LLM response to parse
            head_pos: Current head position
            apple_pos: Current apple position
            body_cells: Current body cell positions
            
        Returns:
            The parser prompt
        """
        return f"""Please parse and format the following LLM response for a Snake game. The response should be a valid JSON object with a "moves" array containing the sequence of moves to make.

Current game state:
- Head position: {head_pos}
- Apple position: {apple_pos}
- Body cells: {body_cells}

Raw LLM response:
{response_text}

Please extract the moves and format them as a JSON object like this:
{{
    "moves": ["UP", "RIGHT", "DOWN", "LEFT"]
}}

Only include valid moves (UP, DOWN, LEFT, RIGHT). If no valid moves are found, return an empty moves array.
"""
