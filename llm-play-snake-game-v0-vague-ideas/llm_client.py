"""
Simple LLM client functions - no classes, just basic API calls

NOTE: This represents our "vague idea" approach to LLM integration!
This shows how you can quickly get LLM functionality working with simple functions.
"""

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
# Loading config at import time - simple and direct for v0
load_dotenv()

def call_llm(prompt, provider="hunyuan"):
    """
    Call an LLM API with a prompt and get the response back.
    
    This is the main entry point for LLM communication. It routes to
    specific provider functions based on the provider parameter.
    
    Args:
        prompt (str): The text prompt to send to the LLM
        provider (str): Which LLM provider to use ("hunyuan", "ollama")
        
    Returns:
        str: The LLM's response text, or a fallback direction on error
        
    """
    print(f"🤖 Calling {provider} LLM...")
    
    # Route to specific provider functions
    # FUTURE IMPROVEMENT: In v1, we could use a provider factory pattern
    if provider == "hunyuan":
        return call_hunyuan(prompt)
    elif provider == "ollama":
        return call_ollama(prompt)
    else:
        print(f"Unknown provider: {provider}")
        return "UP"  # Safe fallback for game continuity

def call_hunyuan(prompt):
    """
    Call the Tencent Hunyuan API for LLM response.
    
    Uses the OpenAI-compatible API interface to communicate with Hunyuan.
    Includes basic error handling to keep the game running smoothly.
    
    Args:
        prompt (str): The text prompt to send to Hunyuan
        
    Returns:
        str: The response from Hunyuan, or "UP" on any error
        
    This function demonstrates basic API integration.
    We keep it simple with embedded configuration for quick prototyping.
    """
    try:
        # Get API key from environment variables
        # Simple and direct approach for v0
        api_key = os.environ.get("HUNYUAN_API_KEY")
        if not api_key:
            print("No Hunyuan API key found!")
            return "UP"  # Keep game running with fallback
        
        # Create OpenAI client configured for Hunyuan endpoint
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
        )
        
        # Make the API call with reasonable defaults
        # FUTURE IMPROVEMENT: In v1, these could be configurable parameters
        response = client.chat.completions.create(
            model="hunyuan-turbos-latest",  # Good general model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,               # Balanced creativity/consistency
            max_tokens=1000               # Sufficient for our short responses
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Simple error handling keeps the game running
        # In v1, we might add more sophisticated error recovery
        print(f"Error calling Hunyuan: {e}")
        return "UP"  # Graceful fallback

def call_ollama(prompt, model="llama3.2:latest"):
    """
    Call a local Ollama API for LLM response.
    
    Connects to a locally running Ollama server and sends the prompt
    to the specified model. Great for local development and testing.
    
    Args:
        prompt (str): The text prompt to send to Ollama
        model (str): The model name to use (default: "llama3.2:latest")
        
    Returns:
        str: The response from Ollama, or "UP" on any error
        
    This shows how to integrate with local LLM servers.
    Simple HTTP requests get the job done for our v0 prototype.
    """
    try:
        # Connect to local Ollama server
        # Standard localhost setup for development
        url = "http://localhost:11434/api/generate"
        
        # Prepare request payload
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }
        
        # Make HTTP POST request
        # Simple and direct - gets our prototype working
        response = requests.post(url, json=data)
        
        # Basic status code checking
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Ollama error: {response.status_code}")
            return "UP"  # Fallback to keep game running
            
    except Exception as e:
        # Keep the game running even if LLM fails
        print(f"Error calling Ollama: {e}")
        return "UP"  # Graceful fallback


