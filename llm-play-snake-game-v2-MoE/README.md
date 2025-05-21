# How to play the game

```
python main.py --provider ollama --model deepseek-r1:14b --parser-provider ollama --parser-model mistral:7b
```

## Installation

Set up API keys in a `.env` file:

```
HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
OLLAMA_HOST=<your_ollama_host_ip_address>
DEEPSEEK_API_KEY=<your_deepseek_api_key_here>
MISTRAL_API_KEY=<your_mistral_api_key_here>
```

## Features

- Two-LLM approach:
  - First LLM generates moves based on game state
  - Second LLM ensures responses are properly formatted as JSON

This bears some similarity to MoE (Mixture-of-Experts) architecture.
