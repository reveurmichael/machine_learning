# LLM Snake Game

A classic Snake game controlled by an LLM (Language Model).

## Features

- LLM-controlled snake that learns to play the game through text representation
- Multi-step planning: LLM provides a sequence of moves to reduce API calls
- Visual display of game state and LLM response information
- Improved text display area for better readability of LLM responses
- Configurable game parameters
- Log saving for analysis of LLM decisions

## Requirements

- Python 3.7+
- PyGame
- NumPy
- OpenAI Python SDK (for Hunyuan API and DeepSeek API access)
- Colorama (for colored terminal output)
- Python-dotenv (for environment variable handling)

## Installation

1. Clone the repository
2. Set up API keys in a `.env` file:

```
HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
OLLAMA_HOST=<your_ollama_host_ip_address>
DEEPSEEK_API_KEY=<your_deepseek_api_key_here>
```

## Running the Game

To run the game with LLM control:

```bash
python main.py
```

Options:
- `--provider hunyuan`, `--provider ollama`, or `--provider deepseek` - Choose the LLM provider
- `--model <model_name>` - Specify which model to use:
  - For Ollama: any available model 
  - For DeepSeek: `deepseek-chat` (default) or `deepseek-reasoner`
- `--max-games 10` - Set maximum number of games to play
- `--move-pause 0.5` - Set pause time between moves in seconds (default: 1.0)

### Using DeepSeek Models

DeepSeek offers two models:
- `deepseek-chat` - The standard chat model (DeepSeek-V3)
- `deepseek-reasoner` - The reasoning model (DeepSeek-R1) which may perform better for logical tasks

Example commands:
```bash
python main.py --provider deepseek --model deepseek-chat
python main.py --provider deepseek --model deepseek-reasoner
```

You can also use the provided helper script for a simpler interface:
```bash
# Run with default deepseek-chat model
python run_deepseek.py

# Run with the reasoning model
python run_deepseek.py reasoner

# Add additional arguments
python run_deepseek.py chat --move-pause 0.5
```

### Controls During Game

- `R` - Reset game
- `ESC` - Quit game

## How Multi-Step Planning Works

The game reduces API costs by:

1. Requesting a sequence of 5-10 moves from the LLM at once
2. Executing these moves sequentially with a configurable pause between each move
3. Requesting a new plan when:
   - All planned moves have been executed
   - The snake eats an apple (requiring a new strategy)
   - The game is reset

This approach significantly reduces API calls while maintaining gameplay quality.

## Project Structure

- `main.py` - Main entry point and game loop
- `snake_game.py` - Core game logic
- `llm_client.py` - LLM API communication
- `gui.py` - Visual display interface
- `config.py` - Game settings and prompt templates


## License

MIT 