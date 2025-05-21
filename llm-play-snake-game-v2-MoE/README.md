# LLM Snake Game

A classic Snake game controlled by an LLM (Language Model).

## Features

- LLM-controlled snake that learns to play the game through text representation
- Multi-step planning: LLM provides a sequence of moves to reduce API calls
- Two-LLM approach:
  - First LLM generates moves based on game state
  - Second LLM ensures responses are properly formatted as JSON
- Comprehensive logging system:
  - Timestamps for all prompts and responses
  - Detailed game summaries with performance metrics
  - Complete tracking of both LLMs' interactions
  - Parser usage statistics

## How the Two-LLM System Works

This project implements a Mixture-of-Experts (MoE) approach using two different LLMs:

1. **Primary LLM (Move Generation)**: Receives the game state and generates a strategic plan for the snake to reach the apple. This LLM focuses on the game logic and strategy.

2. **Secondary LLM (Response Parsing)**: Takes the output from the primary LLM and ensures it conforms to the required JSON format. This improves reliability by handling cases where the primary LLM's output is correct logically but doesn't follow the exact format requirements.

**Advantages of this approach:**
- Improved reliability: Even if the primary LLM doesn't follow the exact JSON format, the game can still use its strategic insights
- Separation of concerns: Each LLM can focus on a specific task (strategy vs. formatting)
- Better error handling: The system can recover from most formatting errors
- Flexibility: You can use different models for each task (e.g., a more powerful model for strategy and a smaller, faster model for parsing)

The system first tries to directly extract valid JSON from the primary LLM's response, only using the secondary LLM when needed, optimizing for both performance and API costs.

## Installation

1. Clone the repository

2. Set up API keys in a `.env` file:

```
HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
OLLAMA_HOST=<your_ollama_host_ip_address>
DEEPSEEK_API_KEY=<your_deepseek_api_key_here>
MISTRAL_API_KEY=<your_mistral_api_key_here>
```

## Running the Game

To run the game with LLM control:

```bash
python main.py
```

Options:
- `--provider hunyuan`, `--provider ollama`, `--provider deepseek`, or `--provider mistral` - Choose the main LLM provider for generating moves
- `--model <model_name>` - Specify which model to use for the main LLM:
  - For Ollama: any available model 
  - For DeepSeek: `deepseek-chat` (default) or `deepseek-reasoner`
  - For Mistral: `mistral-medium-latest` (default) or `mistral-large-latest`
- `--parser-provider <provider>` - Choose the LLM provider for the parser (defaults to same as main provider)
- `--parser-model <model_name>` - Specify which model to use for the parser
- `--max-games 10` - Set maximum number of games to play
- `--move-pause 0.5` - Set pause time between moves in seconds (default: 1.0)

> **Important:** Always use `--model` for the primary LLM and `--parser-model` for the parser LLM. Using `--model` twice will cause an error. For example, use:
> ```bash
> python main.py --provider ollama --model deepseek-r1:32b --parser-provider ollama --parser-model mistral:7b
> ```

### Using DeepSeek Models

DeepSeek offers two models:
- `deepseek-chat` - The standard chat model (DeepSeek-V3)
- `deepseek-reasoner` - The reasoning model (DeepSeek-R1) which may perform better for logical tasks

Example commands:
```bash
python main.py --provider deepseek --model deepseek-chat
python main.py --provider deepseek --model deepseek-reasoner
```

### Using Mistral Models

Mistral offers several models, with the primary ones being:
- `mistral-medium-latest` - The default medium-sized model
- `mistral-large-latest` - The more powerful large model

Example commands:
```bash
# Using medium model (default for mistral provider)
python main.py --provider mistral

# Explicitly using medium model
python main.py --provider mistral --model mistral-medium-latest

# Using large model 
python main.py --provider mistral --model mistral-large-latest

# Using different providers for main LLM and parser
python main.py --provider deepseek --model deepseek-reasoner --parser-provider mistral --parser-model mistral-medium-latest
```

## Project Structure

- `main.py` - Main entry point and game loop
- `snake_game.py` - Core game logic
- `llm_client.py` - LLM API communication for interacting with different providers
- `llm_parser.py` - Secondary LLM for parsing and formatting responses
- `gui.py` - Visual display interface
- `config.py` - Game settings and prompt templates

## Logging System

When you run the game, a timestamped log directory will be created with:

- `game_logs_YYYYMMDD_HHMMSS/` - Root log directory
  - `info.txt` - Experiment configuration and final statistics
  - `gameN_summary.txt` - Detailed summary for each game
  - `prompts/` - All prompts sent to both LLMs
    - Original game state prompts
    - Parser prompts when needed
  - `responses/` - All responses from both LLMs
    - Raw responses from the first LLM
    - Parsed responses from the second LLM (when needed)

Each log file contains timestamps, model information, and detailed metrics to help analyze the system's performance.

## How to improve the game performance

- Maybe, instead of giving multiple moves, LLM only gives one move. But this will drastically increase the time for the whole game playing, as well as the cost of the API calls.
- Or, there are many other insights/details to specify about snake game in the prompt. Those additional insights/details might help LLM figure out better moves.

## License

MIT 