"""
LLM client module for handling communication with different LLM providers.
"""

import os
import json
import yaml
import requests
import subprocess
import tempfile
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
from mistralai import Mistral
from config import get_llm_log_path, LOGGING_CONFIG

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger("LLMClient")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOGGING_CONFIG["log_format"])

# File handler for LLM interactions
llm_log_file = get_llm_log_path()
file_handler = logging.FileHandler(llm_log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LLMClientBase(ABC):
    """Base class for all LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None, 
                 temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """
        Generate text from the LLM

        Args:
            prompt (str): The prompt to send to the LLM
            system_message (str, optional): The system message to use
            history (List[Dict[str, str]], optional): Conversation history
            temperature (float): Sampling temperature (0.0-1.0)
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: Generated response
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get a list of available models"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is running properly"""
        pass


class CloudLLMClient(LLMClientBase):
    """Client for cloud-based LLM providers"""
    
    def __init__(self, provider: str = "mistral", api_key: Optional[str] = None, 
                 model: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the cloud LLM client

        Args:
            provider (str): The provider to use ("mistral", "deepseek", "hunyuan", etc.)
            api_key (str, optional): API key for the provider
            model (str, optional): The model to use
            api_base (str, optional): Base URL for API requests
        """
        self.provider = provider.lower()
        
        # Set up provider-specific configurations
        if self.provider == "mistral":
            # Explicitly load Mistral API key from environment
            self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
            self.api_base = api_base or "https://api.mistral.ai/v1"
            self.model = model or "mistral-medium-latest"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif self.provider == "deepseek":
            # Explicitly load DeepSeek API key from environment
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
            self.api_base = api_base or "https://api.deepseek.com"
            self.model = model or "deepseek-chat"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif self.provider == "hunyuan":
            # Explicitly load Hunyuan API key from environment
            self.api_key = api_key or os.getenv("HUNYUAN_API_KEY", "")
            self.api_base = api_base or "https://api.hunyuan.cloud.tencent.com/v1"
            self.model = model or "hunyuan-turbos-latest"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(self, prompt: str, system_message: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None, 
                 temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """Generate text from the cloud LLM"""
        if not self.api_key:
            return f"API key for {self.provider} is not set"
        
        history = history or []
        
        if self.provider == "mistral":
            return self._generate_mistral(prompt, system_message, history, temperature, max_tokens)
        elif self.provider == "deepseek":
            return self._generate_deepseek(prompt, system_message, history, temperature, max_tokens)
        elif self.provider == "hunyuan":
            return self._generate_hunyuan(prompt, system_message, history, temperature, max_tokens)
        else:
            return f"Unsupported provider: {self.provider}"
    
    def _generate_mistral(self, prompt: str, system_message: Optional[str], 
                         history: List[Dict[str, str]], 
                         temperature: float, max_tokens: int) -> str:
        """Generate text using Mistral AI API"""
        try:
            url = f"{self.api_base}/chat/completions"
            
            # Prepare conversation messages
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add conversation history
            for msg in history:
                messages.append(msg)
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request data
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the request
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            # Extract and return the response content
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Mistral AI API: {str(e)}"
    
    def _generate_deepseek(self, prompt: str, system_message: Optional[str], 
                         history: List[Dict[str, str]], 
                         temperature: float, max_tokens: int) -> str:
        """Generate text using DeepSeek API"""
        try:
            url = f"{self.api_base}/chat/completions"
            
            # Prepare conversation messages
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Add conversation history
            for msg in history:
                messages.append(msg)
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request data
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the request
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            # Extract and return the response content
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"
    
    def _generate_hunyuan(self, prompt: str, system_message: Optional[str], 
                           history: List[Dict[str, str]], 
                           temperature: float, max_tokens: int) -> str:
        """Generate text using Hunyuan API"""
        try:
            url = f"{self.api_base}/chat/completions"
            
            # Prepare conversation messages
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add conversation history
            for msg in history:
                messages.append(msg)
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request data
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the request
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()

            # Extract and return the response content
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Hunyuan API: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get a list of available models"""
        try:
            if self.provider == "mistral":
                url = f"{self.api_base}/models"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return [model["id"] for model in response.json()["data"]]
            
            elif self.provider == "deepseek":
                url = f"{self.api_base}/models"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return [model["id"] for model in response.json()["data"]]
            
            elif self.provider == "hunyuan":
                url = f"{self.api_base}/models"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return [model["id"] for model in response.json()["data"]]
            
            else:
                return []
        except Exception as e:
            print(f"Error getting available models: {str(e)}")
            return []
    
    def health_check(self) -> bool:
        """Check if the LLM service is running properly"""
        try:
            # Simple health check by getting models
            models = self.get_available_models()
            return len(models) > 0
        except Exception:
            return False


class LocalLLMClient(LLMClientBase):
    """Client for locally running LLM models (e.g., Ollama)"""
    
    def __init__(self, provider: str = "ollama", host: str = "remote-ollama-server", 
                 port: int = 11434, model: str = "mistral"):
        """
        Initialize the local LLM client

        Args:
            provider (str): The provider to use (default: "ollama")
            host (str): Host where the local LLM is running
            port (int): Port for the local LLM
            model (str): The model to use
        """
        self.provider = provider.lower()
        self.host = host
        self.port = port
        self.model = model
        
        if self.provider == "ollama":
            self.api_base = f"http://{host}:{port}"
            self.headers = {"Content-Type": "application/json"}
        else:
            raise ValueError(f"Unsupported local provider: {provider}")

    def generate(self, prompt: str, system_message: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None, 
                 temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """Generate text from the local LLM"""
        history = history or []
        
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system_message, history, temperature, max_tokens)
        else:
            return f"Unsupported provider: {self.provider}"
    
    def _generate_ollama(self, prompt: str, system_message: Optional[str], 
                        history: List[Dict[str, str]], 
                        temperature: float, max_tokens: int) -> str:
        """Generate text using Ollama API"""
        try:
            # Use completions API instead of chat API if there are problems
            use_completions_api = False
            url = f"{self.api_base}/api/chat"
            
            if use_completions_api:
                url = f"{self.api_base}/api/generate"

            # Build messages from history
            messages = []
            for msg in history:
                role = msg["role"]
                # Ollama uses "assistant" instead of "system"
                if role == "system":
                    role = "assistant" 
                messages.append({"role": role, "content": msg["content"]})

            # Add system message if provided
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request data for chat API
            if not use_completions_api:
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False  # Disable streaming to avoid JSON parsing issues
                }
            else:
                # Prepare request data for completions API
                combined_prompt = ""
                
                # Add system message
                if system_message:
                    combined_prompt += f"System: {system_message}\n\n"
                
                # Add history
                for msg in history:
                    role = msg["role"]
                    if role == "user":
                        combined_prompt += f"User: {msg['content']}\n\n"
                    elif role == "assistant" or role == "system":
                        combined_prompt += f"Assistant: {msg['content']}\n\n"
                
                # Add current prompt
                combined_prompt += f"User: {prompt}\n\nAssistant: "
                
                data = {
                    "model": self.model,
                    "prompt": combined_prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False  # Disable streaming
                }
            
            # Make the request with increased timeout
            logger.info(f"Sending request to Ollama API: {url}")
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Ollama API returned status code {response.status_code}: {response.text}")
                return f"Error: Ollama API returned status code {response.status_code}"
            
            # Try to extract the response content from JSON
            try:
                # First try standard JSON parsing
                response_json = response.json()
                
                if use_completions_api:
                    return response_json.get("response", "No response from Ollama")
                else:
                    return response_json.get("message", {}).get("content", "No content in Ollama response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, Response text: {response.text[:200]}")
                
                # Try to handle streaming or malformed responses
                response_text = response.text.strip()
                
                # If the response looks like multiple JSON objects
                if response_text.startswith("{") and "}{" in response_text:
                    try:
                        # Split at the boundaries of JSON objects
                        json_parts = response_text.replace("}{", "}|||{").split("|||")
                        
                        # Get the last complete JSON object
                        last_json = json_parts[-1]
                        last_response = json.loads(last_json)
                        
                        if use_completions_api:
                            return last_response.get("response", "")
                        else:
                            return last_response.get("message", {}).get("content", "")
                    except Exception as e2:
                        logger.error(f"Error parsing split JSON: {e2}")
                
                # If the response is in JSONL format (one JSON per line)
                if "\n" in response_text and response_text.strip().startswith("{"):
                    try:
                        # Get the last line that is a complete JSON
                        json_lines = [line for line in response_text.split("\n") if line.strip().startswith("{")]
                        if json_lines:
                            last_line = json_lines[-1]
                            last_response = json.loads(last_line)
                            
                            if use_completions_api:
                                return last_response.get("response", "")
                            else:
                                return last_response.get("message", {}).get("content", "")
                    except Exception as e3:
                        logger.error(f"Error parsing JSONL: {e3}")
                
                # Last resort: extract content between quotes after "content": 
                import re
                content_pattern = r'"content":\s*"((?:\\.|[^"\\])*)"'
                match = re.search(content_pattern, response_text)
                if match:
                    return match.group(1).replace('\\"', '"').replace('\\\\', '\\')
                
                # If all extraction attempts fail, return error
                return f"Error parsing Ollama response: Response format not recognized"
                
        except requests.RequestException as e:
            logger.error(f"Request to Ollama API failed: {e}")
            return f"Error connecting to Ollama API: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Ollama client: {e}")
            return f"Error calling Ollama API: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get a list of available models"""
        try:
            if self.provider == "ollama":
                url = f"{self.api_base}/api/tags"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return [model["name"] for model in response.json()["models"]]
            else:
                return []
        except Exception as e:
            print(f"Error getting available models: {str(e)}")
            return []
    
    def health_check(self) -> bool:
        """Check if the LLM service is running properly"""
        try:
            if self.provider == "ollama":
                url = f"{self.api_base}/api/tags"
                response = requests.get(url, headers=self.headers)
                return response.status_code == 200
            return False
        except Exception:
            return False
    
    def pull_model(self, model_name: Optional[str] = None) -> str:
        """Pull a model from Ollama"""
        model = model_name or self.model
        
        try:
            if self.provider == "ollama":
                url = f"{self.api_base}/api/pull"
                data = {"name": model}
                response = requests.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                return f"Successfully pulled model: {model}"
            else:
                return f"Pulling models not supported for provider: {self.provider}"
        except Exception as e:
            return f"Error pulling model: {str(e)}"
    
    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a model is available locally"""
        model = model_name or self.model
        available_models = self.get_available_models()
        return model in available_models


class LLMClientManager:
    """Manager for handling different LLM clients"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM client manager

        Args:
            config_path (str, optional): Path to the configuration file
        """
        self.clients = {}
        self.default_client = None
        
        # Load configuration if provided
        if config_path:
            try:
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f)
                
                # Extract LLM config section
                config = full_config.get("llm", {})
                
                # Create clients from config
                for name, client_config in config.get("clients", {}).items():
                    client_type = client_config.get("type", "")
                    
                    if client_type == "local":
                        client = LocalLLMClient(
                            provider=client_config.get("provider", "ollama"),
                            host=client_config.get("host", "localhost"),
                            port=client_config.get("port", 11434),
                            model=client_config.get("model", "mistral")
                        )
                        self.add_client(name, client)
                    
                    elif client_type == "cloud":
                        provider = client_config.get("provider", "mistral")
                        # Create cloud client with explicit provider and no API key
                        # API keys will be loaded from environment variables
                        client = CloudLLMClient(
                            provider=provider,
                            api_key=None,  # Will be loaded from environment variable
                            model=client_config.get("model", ""),
                            api_base=client_config.get("api_base", "")
                        )
                        self.add_client(name, client)
                
                # Set default client
                default_client_name = config.get("default_client", "")
                if default_client_name and default_client_name in self.clients:
                    self.default_client = default_client_name
            except Exception as e:
                logger.error(f"Error loading LLM configuration: {e}")
        
        # If no clients were configured, create a default Ollama client
        if not self.clients:
            self.clients["ollama"] = LocalLLMClient()
            self.default_client = "ollama"
    
    def add_client(self, name: str, client: LLMClientBase) -> None:
        """
        Add a new LLM client

        Args:
            name (str): Name for the client
            client (LLMClientBase): The client instance
        """
        self.clients[name] = client
        if self.default_client is None:
            self.default_client = name
    
    def remove_client(self, name: str) -> bool:
        """
        Remove a client by name

        Args:
            name (str): Name of the client to remove

        Returns:
            bool: True if successful, False otherwise
        """
        if name in self.clients:
            if self.default_client is name:
                self.default_client = None
            del self.clients[name]
            return True
        return False
    
    def set_default_client(self, name: str) -> bool:
        """
        Set the default client by name
        
        Args:
            name (str): Name of the client to set as default
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name in self.clients:
            self.default_client = name
            return True
        return False
    
    def generate(self, prompt: str, system_message: Optional[str] = None,
                history: Optional[List[Dict[str, str]]] = None,
                temperature: float = 0.7, max_tokens: int = 8192,
                client_name: Optional[str] = None) -> str:
        """
        Generate text using the specified or default client
        
        Args:
            prompt (str): The prompt to send
            system_message (str, optional): System message to use
            history (List[Dict[str, str]], optional): Conversation history
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            client_name (str, optional): Name of the client to use
            
        Returns:
            str: Generated response
        """
        # First determine which client to use
        original_client_name = client_name or self.default_client
        
        if client_name and client_name in self.clients:
            client = self.clients[client_name]
        elif self.default_client:
            client = self.clients[self.default_client]
        else:
            return "No LLM client available"
        
        # Log the request
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client": original_client_name,
            "model": client.model,
            "prompt": prompt,
            "system_message": system_message,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        logger.info(f"LLM Request: {json.dumps(log_entry)}")
        
        # Try to generate a response
        try:
            # Generate the response
            response = client.generate(
                prompt=prompt,
                system_message=system_message,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Check if the response indicates an error
            if response.startswith("Error calling") or response.startswith("Error parsing"):
                # Try a fallback client if available
                if "fallback_ollama" in self.clients and original_client_name != "fallback_ollama":
                    logger.warning(f"Primary LLM client failed, trying fallback: {response}")
                    fallback_client = self.clients["fallback_ollama"]
                    
                    # Try the fallback client
                    try:
                        fallback_response = fallback_client.generate(
                            prompt=prompt,
                            system_message=system_message,
                            history=history,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # If the fallback worked, use its response
                        if not fallback_response.startswith("Error"):
                            response = fallback_response
                            logger.info("Successfully used fallback LLM client")
                    except Exception as e:
                        logger.error(f"Fallback LLM client also failed: {str(e)}")
            
            # Log the response
            logger.info(f"LLM Response: {response[:200]}...")
            
            # Save complete interaction to log file
            try:
                with open(get_llm_log_path(), "a") as f:
                    f.write("\n" + "-"*80 + "\n")
                    f.write(f"TIMESTAMP: {log_entry['timestamp']}\n")
                    f.write(f"CLIENT: {log_entry['client']} - MODEL: {log_entry['model']}\n")
                    f.write(f"SYSTEM MESSAGE:\n{system_message or 'None'}\n\n")
                    f.write(f"PROMPT:\n{prompt}\n\n")
                    f.write(f"RESPONSE:\n{response}\n\n")
            except Exception as e:
                logger.error(f"Error writing to interaction log: {e}")
            
            return response
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            logger.error(error_message)
            
            # Try fallback client if available
            if "fallback_ollama" in self.clients and original_client_name != "fallback_ollama":
                logger.warning("Primary LLM client exception, trying fallback")
                try:
                    fallback_client = self.clients["fallback_ollama"]
                    fallback_response = fallback_client.generate(
                        prompt=prompt,
                        system_message=system_message,
                        history=history,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Log the fallback response
                    logger.info(f"Fallback LLM Response: {fallback_response[:200]}...")
                    
                    return fallback_response
                except Exception as fallback_e:
                    logger.error(f"Fallback LLM client also failed: {str(fallback_e)}")
            
            return error_message
    
    def save_config(self, config_path: str) -> bool:
        """
        Save the current configuration to a file

        Args:
            config_path (str): Path to save the configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to load existing config if it exists
            full_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f) or {}
            
            # Prepare LLM config structure
            llm_config = {
                "default_client": self.default_client,
                "clients": {}
            }
            
            # Add client configurations
            for name, client_obj in self.clients.items():
                client_info = {
                    "model": client_obj.model
                }
                
                if isinstance(client_obj, LocalLLMClient):
                    client_info.update({
                        "type": "local",
                        "provider": client_obj.provider,
                        "host": client_obj.host,
                        "port": client_obj.port
                    })
                elif isinstance(client_obj, CloudLLMClient):
                    client_info.update({
                        "type": "cloud",
                        "provider": client_obj.provider,
                        "api_base": client_obj.api_base
                    })
                    
                    # Never include API keys in the config file
                    # All API keys should be in the .env file only
                
                llm_config["clients"][name] = client_info
            
            # Update LLM section in full config
            full_config["llm"] = llm_config
            
            # Write to file
            with open(config_path, "w") as f:
                yaml.dump(full_config, f, default_flow_style=False)
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def list_clients(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available clients
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of client configurations
        """
        result = {}
        
        for name, client in self.clients.items():
            if isinstance(client, CloudLLMClient):
                result[name] = {
                    "type": "cloud",
                    "provider": client.provider,
                    "model": client.model,
                    "is_default": client is self.clients[self.default_client]
                }
            elif isinstance(client, LocalLLMClient):
                result[name] = {
                    "type": "local",
                    "provider": client.provider,
                    "host": client.host,
                    "port": client.port,
                    "model": client.model,
                    "is_default": client is self.clients[self.default_client]
                }
        
        return result
    
    def is_client_healthy(self, name: Optional[str] = None) -> bool:
        """
        Check if a client is healthy
        
        Args:
            name (str, optional): Name of the client to check
            
        Returns:
            bool: True if healthy, False otherwise
        """
        if name:
            if name in self.clients:
                return self.clients[name].health_check()
            return False
        elif self.default_client:
            return self.clients[self.default_client].health_check()
        return False
    
    def get_available_models(self, client_name: Optional[str] = None) -> List[str]:
        """
        Get available models for a client
        
        Args:
            client_name (str, optional): Name of the client
            
        Returns:
            List[str]: List of available model names
        """
        if client_name and client_name in self.clients:
            return self.clients[client_name].get_available_models()
        elif self.default_client:
            return self.clients[self.default_client].get_available_models()
        return []
