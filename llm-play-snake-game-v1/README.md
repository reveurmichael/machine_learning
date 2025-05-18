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
- OpenAI Python SDK (for Hunyuan API access)
- Colorama (for colored terminal output)
- Python-dotenv (for environment variable handling)

## Installation

1. Clone the repository
2. Set up API keys in a `.env` file:

```
HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
OLLAMA_HOST=<your_ollama_host_ip_address>
```

## Running the Game

To run the game with LLM control:

```bash
python main.py
```

Options:
- `--provider hunyuan` or `--provider ollama` - Choose the LLM provider
- `--max-games 10` - Set maximum number of games to play
- `--move-pause 0.5` - Set pause time between moves in seconds (default: 1.0)

### Controls During Game

- `R` - Reset game
- `SPACE` - Toggle game speed
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

## Game Logs

The game saves logs for analysis in the `game_logs_[timestamp]` directory:

- Prompts sent to the LLM
- Responses received from the LLM
- Game summaries with scores and steps

## License

MIT 