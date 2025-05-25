![](./img/a.jpg)


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

You can also run the game with just a primary LLM (no parser):

```
python main.py --provider ollama --model deepseek-r1:7b --parser-provider none
```

This will bypass the secondary LLM and use the primary LLM's output directly.

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
   - Can be disabled with `--parser-provider none` to use primary LLM output directly

## Command Line Arguments

- `--provider`: LLM provider for the primary LLM (hunyuan, ollama, deepseek, or mistral)
- `--model`: Model name for the primary LLM
- `--parser-provider`: LLM provider for the secondary LLM (defaults to primary provider if not specified). Use "none" to skip using a parser.
- `--parser-model`: Model name for the secondary LLM
- `--max-games`: Maximum number of games to play
- `--move-pause`: Pause between sequential moves in seconds
- `--max-steps`: Maximum steps a snake can take in a single game (default: 400)
- `--sleep-before-launching`: Time to sleep (in minutes) before launching the program

## Game Termination Conditions

The snake game will terminate under any of the following conditions:
1. Snake hits a wall (boundary of the game board)
2. Snake collides with its own body
3. Maximum steps limit is reached (default: 400 steps)
4. Three consecutive empty moves are returned without ERROR
   - An empty move occurs when the LLM returns `{"moves":[], "reasoning":"..."}`
   - If the reasoning contains "ERROR", the consecutive count is reset

### Log Files

The game automatically logs all prompts, responses, and game statistics to a folder named `primarymodel_timestamp` where:
- `primarymodel` is the name of the primary model (with ":" replaced by "-")
- `timestamp` is the current date and time

When running without a parser LLM (`--parser-provider none`), the system will not create parser-related log files.

## Game Summary

At the end of each game, a summary is generated with:
- Score and steps taken
- Game end reason (wall collision, self collision, max steps, or consecutive empty moves)
- Performance metrics
- LLM usage statistics