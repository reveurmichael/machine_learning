![](./a.jpg)


# LLM-Powered Snake Game (MoE Variant)

This project implements a classic Snake game controlled by two Large Language Models in a Mixture-of-Experts inspired approach.

## How to Play the Game

Run the game with different LLM providers and models:

```
python main.py --provider ollama --model deepseek-r1:7b --parser-provider ollama --parser-model mistral:7b

python main.py --provider ollama --model deepseek-r1:7b --parser-provider ollama --parser-model llama3.1:8b

python main.py --provider ollama --model deepseek-r1:7b --parser-provider ollama --parser-model gemma2:9b

python main.py --provider ollama --model deepseek-r1:14b --parser-provider ollama --parser-model mistral:7b

python main.py --provider ollama --model deepseek-r1:14b --parser-provider ollama --parser-model llama3.1:8b

python main.py --provider ollama --model deepseek-r1:14b --parser-provider ollama --parser-model gemma2:9b

python main.py --provider ollama --model deepseek-r1:32b --parser-provider ollama --parser-model mistral:7b

python main.py --provider ollama --model deepseek-r1:32b --parser-provider ollama --parser-model llama3.1:8b

python main.py --provider ollama --model deepseek-r1:32b --parser-provider ollama --parser-model gemma2:9b
```

## Installation

Set up API keys in a `.env` file:

```
HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
OLLAMA_HOST=<your_ollama_host_ip_address>
DEEPSEEK_API_KEY=<your_deepseek_api_key_here>
MISTRAL_API_KEY=<your_mistral_api_key_here>
```

## Two-LLM Architecture (MoE Approach)

This project implements a Mixture-of-Experts inspired approach where two specialized LLMs work together:

1. **Primary LLM (Game Strategy Expert)**
   - Takes the game state as input
   - Analyzes the Snake's position, apple location, and available moves
   - Generates a logical plan to navigate toward the apple
   - May or may not generate properly structured output

2. **Secondary LLM (Formatting Expert)**
   - Takes the primary LLM's output as input
   - Specializes in parsing and formatting the response
   - Ensures the final output follows the required JSON format
   - Acts as a guarantor of response quality

## Command Line Arguments

- `--provider`: LLM provider for the primary LLM (hunyuan, ollama, deepseek, or mistral)
- `--model`: Model name for the primary LLM
- `--parser-provider`: LLM provider for the secondary LLM (defaults to primary provider if not specified)
- `--parser-model`: Model name for the secondary LLM
- `--max-games`: Maximum number of games to play
- `--move-pause`: Pause between sequential moves in seconds
