"""
LLM client module for handling communication with different LLM providers.
"""

import os
import json
import time
import requests
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
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The LLM's response as a string
        """
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
                print(f"HUNYUAN_API_KEY environment variable: {api_key[:5]}..." if api_key and len(api_key) > 5 else "Not set")
                return self._get_fallback_response()
                
            # Construct OpenAI client for Hunyuan
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.hunyuan.cloud.tencent.com/v1",
            )
            
            # Create the message
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Extract parameters, preferring provided keyword args over instance variables
            model = kwargs.get("model", self.model) or "hunyuan-turbos-latest"
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
                    }
                )
                
                # Return the response
                return completion.choices[0].message.content
            except Exception as api_error:
                print(f"Error during Hunyuan API call: {api_error}")
                return self._get_fallback_response()
            
        except Exception as e:
            print(f"Error generating response from Hunyuan: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response()
    
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
                print(f"DEEPSEEK_API_KEY environment variable: {api_key[:5]}..." if api_key and len(api_key) > 5 else "Not set")
                return self._get_fallback_response()
                
            # Construct OpenAI client for Deepseek
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
            
            # Extract parameters
            model = kwargs.get("model", "deepseek-chat")  # Default to deepseek-chat
            # Validate model selection
            if model not in ["deepseek-chat", "deepseek-reasoner"]:
                print(f"Warning: Unknown Deepseek model '{model}', using deepseek-chat instead")
                model = "deepseek-chat"
                
            temperature = kwargs.get("temperature", 0.2)  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)
            
            print(f"Using Deepseek model: {model}")
            
            # Create the messages with system prompt to ensure proper response format
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            print(f"Making API call to Deepseek with model: {model}, temperature: {temperature}")
            
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
                return self._get_fallback_response()
            
        except Exception as e:
            print(f"Error generating response from Deepseek: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response()
    
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
                print(f"MISTRAL_API_KEY environment variable: {api_key[:5]}..." if api_key and len(api_key) > 5 else "Not set")
                return self._get_fallback_response()
                
            # Extract parameters
            model = kwargs.get("model", "mistral-medium-latest")  # Default to medium model
            # Validate model selection
            valid_models = ["mistral-tiny", "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
            if model not in valid_models:
                print(f"Warning: Unknown Mistral model '{model}', using mistral-medium-latest instead")
                model = "mistral-medium-latest"
                
            temperature = kwargs.get("temperature", 0.2)  # Lower temperature for more deterministic responses
            max_tokens = kwargs.get("max_tokens", 8192)
            
            print(f"Using Mistral model: {model}")
            
            # Create Mistral client
            client = Mistral(api_key=api_key)
            
            # Create message structure
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            print(f"Making API call to Mistral with model: {model}, temperature: {temperature}")
            
            # Make the API call
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Return the response
                return chat_response.choices[0].message.content
            except Exception as api_error:
                print(f"Error during Mistral API call: {api_error}")
                return self._get_fallback_response()
            
        except Exception as e:
            print(f"Error generating response from Mistral: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Return a fallback response if the LLM is not available.
        
        This provides a simple deterministic movement pattern if we can't
        get a proper response from the LLM.
        """
        # Simple movement pattern: RIGHT, DOWN, LEFT, UP in sequence
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        # Use milliseconds to select a direction pseudorandomly but deterministically
        index = int((time.time() * 1000) % 4)
        return f"GENERATED_CODE:\n{directions[index]}"
    
    def _generate_ollama_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Ollama LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            # Extract parameters, preferring provided keyword args over instance variables
            model = kwargs.get("model", self.model)
            if not model:
                model = self._get_best_ollama_model()
                
            server = kwargs.get("server", os.environ.get("OLLAMA_HOST", "localhost"))
            temperature = kwargs.get("temperature", 0.7)
            
            print(f"Making API call to Ollama with model: {model}, temperature: {temperature}")
            
            # Make the API call
            response = requests.post(
                f"http://{server}:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                }
            )
            
            # Check if response is valid JSON
            try:
                response_json = response.json()
                return response_json.get("response", "ERROR: No response field in JSON")
            except json.JSONDecodeError:
                return f"ERROR: Invalid JSON response - {response.text[:100]}"
            
        except requests.exceptions.Timeout:
            print(f"Timeout error connecting to Ollama server at {server}")
            return self._get_fallback_response()
        except requests.exceptions.ConnectionError:
            print(f"Connection error to Ollama server at {server}")
            return self._get_fallback_response()
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return self._get_fallback_response()
    
    def _get_best_ollama_model(self, server: str = None) -> str:
        """Get the 'best' (largest) Ollama model available locally.
        
        Args:
            server: IP address or hostname of the Ollama server
            
        Returns:
            The name of the largest model available, or a fallback default
        """
        fallback_model = "llama3.2:1b"  # Fallback default
        
        # Use the provided server or get from environment variable or default to localhost
        if server is None:
            server = os.environ.get("OLLAMA_HOST", "localhost")
        
        try:
            # Try to get list of models from Ollama API
            response = requests.get(f"http://{server}:11434/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                # No models available
                if not models:
                    print(f"No Ollama models found. Using fallback model: {fallback_model}")
                    return fallback_model
                
                # Try to find models with parameter information
                models_with_size = []
                for model in models:
                    model_name = model.get("name")
                    size_mb = model.get("size") / (1024 * 1024) if model.get("size") else 0
                    
                    # Look for parameter count in name (like 7b, 13b, 70b, etc.)
                    param_size = 0
                    name_parts = (
                        model_name.lower().replace("-", " ").replace(":", " ").split()
                    )
                    for part in name_parts:
                        if part.endswith("b") and part[:-1].isdigit():
                            try:
                                param_size = int(part[:-1])
                                break
                            except ValueError:
                                pass
                    
                    models_with_size.append((model_name, param_size, size_mb))
                
                # Sort models by parameter size (primary) and file size (secondary)
                models_with_size.sort(key=lambda x: (x[1], x[2]), reverse=True)
                
                # Return the largest model
                if models_with_size:
                    best_model = models_with_size[0][0]
                    print(f"Selected largest available model: {best_model}")
                    return best_model
                
                # If we couldn't determine sizes, just return the first model
                print(
                    f"Couldn't determine model sizes. Using first available model: {models[0]['name']}"
                )
                return models[0]["name"]
                
        except Exception as e:
            print(f"Error getting Ollama models: {e}")
            print(f"Using fallback model: {fallback_model}")
        
        # Only try command line if server is localhost
        if server == "localhost":
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse the output to find models
                    lines = result.stdout.strip().split("\n")
                    models = []
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            parts = line.split()
                            if parts:
                                models.append(parts[0])  # First column is the model name
                    
                    if models:
                        # Sort models by potential size indicators in name
                        # This is a heuristic approach - larger models often have numbers like 70b, 13b, 7b
                        def extract_size(model_name):
                            name = model_name.lower()
                            for size in ["70b", "34b", "13b", "7b", "3b", "1b"]:
                                if size in name:
                                    return int(size[:-1])  # Convert '7b' to 7
                            return 0
                        
                        models.sort(key=extract_size, reverse=True)
                        best_model = models[0]
                        print(f"Selected model with largest parameter count: {best_model}")
                        return best_model
            except Exception as e:
                print(f"Error running ollama command: {e}")
        
        return fallback_model 