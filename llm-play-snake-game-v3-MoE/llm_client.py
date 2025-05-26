"""
LLM client module for the Snake game.
Provides interfaces for interacting with language models.
"""

import os
import json
import time
import requests
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from mistralai import Mistral
from config import PARSER_PROMPT_TEMPLATE

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, provider="ollama", model=None):
        """Initialize the LLM client.
        
        Args:
            provider: LLM provider ("hunyuan", "ollama", "deepseek", or "mistral")
            model: Model name to use
        """
        self.provider = provider.lower()
        self.model = model
        
        # Statistics tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.response_times = []
        
        # Validate provider
        valid_providers = ["hunyuan", "ollama", "deepseek", "mistral", "none"]
        if self.provider not in valid_providers:
            print(f"Warning: Unsupported provider '{provider}'. Defaulting to 'ollama'.")
            self.provider = "ollama"
            
        # For 'none' provider, we'll use a special bypass method
        if self.provider == "none":
            print("Using 'none' provider - will bypass parser LLM and use primary LLM output directly.")

    def get_token_stats(self):
        """Get token usage statistics.
        
        Returns:
            Dictionary with token usage statistics
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_prompt_tokens": self.total_prompt_tokens / max(1, len(self.response_times)),
            "avg_completion_tokens": self.total_completion_tokens / max(1, len(self.response_times)),
            "avg_total_tokens": self.total_tokens / max(1, len(self.response_times))
        }
    
    def get_response_time_stats(self):
        """Get response time statistics.
        
        Returns:
            Dictionary with response time statistics
        """
        if not self.response_times:
            return {
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0
            }
            
        return {
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times)
        }

    def generate_response(self, prompt, **kwargs):
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            The LLM's response text
        """
        start_time = time.time()
        response = ""
        
        if self.provider == "hunyuan":
            response, token_info = self._generate_hunyuan_response(prompt, **kwargs)
        elif self.provider == "ollama":
            response, token_info = self._generate_ollama_response(prompt, **kwargs)
        elif self.provider == "deepseek":
            response, token_info = self._generate_deepseek_response(prompt, **kwargs)
        elif self.provider == "mistral":
            response, token_info = self._generate_mistral_response(prompt, **kwargs)
        elif self.provider == "none":
            # Special case for bypassing the parser
            response = prompt  # Just return the input directly
            token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            print(f"Unsupported provider: {self.provider}. Falling back to Ollama.")
            response, token_info = self._generate_ollama_response(prompt, **kwargs)
        
        # Record response time
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Update token statistics
        self.last_prompt_tokens = token_info.get("prompt_tokens", 0)
        self.last_completion_tokens = token_info.get("completion_tokens", 0)
        self.last_total_tokens = token_info.get("total_tokens", 0)
        
        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens
        self.total_tokens += self.last_total_tokens
            
        return response

    def _generate_hunyuan_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response from Tencent Hunyuan LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple containing the LLM's response as a string and token counts dictionary
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("HUNYUAN_API_KEY")
            if not api_key or api_key == "your_hunyuan_api_key_here":
                print("Warning: Hunyuan API key not properly configured in .env file")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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

                # Extract token usage
                token_info = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }

                # Return the response and token info
                return completion.choices[0].message.content, token_info
            except Exception as api_error:
                print(f"Error during Hunyuan API call: {api_error}")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        except Exception as e:
            print(f"Error generating response from Hunyuan: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _generate_ollama_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response from Ollama LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple containing the LLM's response as a string and token counts dictionary
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
                result = response_json.get("response", "ERROR: No response field in JSON")
                
                # Extract token information if available (Ollama may not provide this)
                prompt_tokens = response_json.get("prompt_eval_count", 0)
                completion_tokens = response_json.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                token_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                return result, token_info
            except json.JSONDecodeError:
                return f"ERROR: Invalid JSON response - {response.text[:100]}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        except requests.exceptions.Timeout:
            print(f"Timeout error connecting to Ollama server at {server}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except requests.exceptions.ConnectionError:
            print(f"Connection error to Ollama server at {server}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _generate_deepseek_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response from Deepseek LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple containing the LLM's response as a string and token counts dictionary
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key or api_key == "your_deepseek_api_key_here":
                print("Warning: Deepseek API key not properly configured in .env file")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            # Construct OpenAI client for Deepseek
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
            )

            # Create the message
            messages = [{"role": "user", "content": prompt}]

            model = kwargs.get("model", self.model) or "deepseek-ai/deepseek-chat-8b"
            temperature = kwargs.get(
                "temperature", 0.2
            )  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)

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

                # Extract token usage
                token_info = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }

                # Return the response and token info
                return completion.choices[0].message.content, token_info
            except Exception as api_error:
                print(f"Error during Deepseek API call: {api_error}")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        except Exception as e:
            print(f"Error generating response from Deepseek: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _generate_mistral_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response from Mistral LLM.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Tuple containing the LLM's response as a string and token counts dictionary
        """
        try:
            # Check if API key is properly set
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key or api_key == "your_mistral_api_key_here":
                print("Warning: Mistral API key not properly configured in .env file")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            # Create Mistral client
            client = Mistral(api_key=api_key)

            # Create the message
            model = kwargs.get("model", self.model) or "mistral-medium"
            temperature = kwargs.get(
                "temperature", 0.2
            )  # Lower temperature for more deterministic responses

            print(
                f"Making API call to Mistral with model: {model}, temperature: {temperature}"
            )

            # Make the API call
            try:
                chat_response = client.chat(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )

                # Extract token usage
                token_info = {
                    "prompt_tokens": chat_response.usage.prompt_tokens,
                    "completion_tokens": chat_response.usage.completion_tokens,
                    "total_tokens": chat_response.usage.total_tokens
                }

                # Return the response and token info
                return chat_response.choices[0].message.content, token_info
            except Exception as api_error:
                print(f"Error during Mistral API call: {api_error}")
                return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        except Exception as e:
            print(f"Error generating response from Mistral: {e}")
            traceback.print_exc()
            return "ERROR LLMCLIENT", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class LLMOutputParser(LLMClient):
    """LLM client specialized for parsing and formatting responses."""

    def __init__(self, provider="ollama", model=None):
        """Initialize the LLM output parser.
        
        Args:
            provider: LLM provider ("hunyuan", "ollama", "deepseek", "mistral", or "none")
            model: Model name to use
        """
        super().__init__(provider, model)

    def parse_and_format(self, response_text, head_pos, apple_pos, body_cells):
        """Parse and format the LLM response.
        
        Args:
            response_text: The raw LLM response text
            head_pos: The current head position
            apple_pos: The current apple position
            body_cells: The current body cells
            
        Returns:
            Tuple containing the parsed response and a boolean indicating success
        """
        # If we're using the 'none' provider, just return the input directly
        if self.provider == "none":
            return response_text, True
            
        # Create the parser prompt
        parser_prompt = self._create_parser_prompt(response_text, head_pos, apple_pos, body_cells)
        
        # Generate a response from the parser
        parser_response = self.generate_response(parser_prompt)
        
        # Add token stats to track secondary LLM usage
        token_info = {
            "prompt_tokens": self.last_prompt_tokens,
            "completion_tokens": self.last_completion_tokens,
            "total_tokens": self.last_total_tokens
        }
        
        # Check if the response is valid
        if parser_response.startswith("ERROR"):
            return "ERROR", False
            
        # Return the parsed response
        return parser_response, True

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
        prompt = PARSER_PROMPT_TEMPLATE
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_FIRST_LLM_RESPONSE", response_text)
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", str(head_pos))
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", str(apple_pos))
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", str(body_cells))
        
        return prompt
