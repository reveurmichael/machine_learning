"""
LLM client module for handling communication with different LLM providers.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

# Load environment variables from .env file
load_dotenv()

class LLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, provider: str = "hunyuan"):
        """Initialize the LLM client.
        
        Args:
            provider: The LLM provider to use ("hunyuan" or "ollama")
        """
        self.provider = provider
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The LLM's response as a string
        """
        if self.provider == "hunyuan":
            return self._generate_hunyuan_response(prompt, **kwargs)
        elif self.provider == "ollama":
            return self._generate_ollama_response(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
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
            
            # Extract parameters
            model = kwargs.get("model", "hunyuan-turbos-latest")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)
            enable_enhancement = kwargs.get("enable_enhancement", True)
            
            # Make the API call
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
            
        except Exception as e:
            print(f"Error generating response from Hunyuan: {e}")
            return "No response from Hunyuan"
    
    
    def _generate_ollama_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Ollama LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            # Extract parameters
            model = kwargs.get("model", self._get_best_ollama_model())
            server = kwargs.get("server", "localhost")
            temperature = kwargs.get("temperature", 0.7)
            
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
            
            # Return the response
            return response.json()["response"]
            
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return f"ERROR: {str(e)}"
    
    def _get_best_ollama_model(self, server: str = "localhost") -> str:
        """Get the 'best' (largest) Ollama model available locally.
        
        Args:
            server: IP address or hostname of the Ollama server
            
        Returns:
            The name of the largest model available, or a fallback default
        """
        fallback_model = "llama3.2:1b"  # Fallback default
        
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