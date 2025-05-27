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


## Continue mode

For some reasons, you might want to pause an ongoing experiment, and then continue with the same game. 

In continue mode, only "--no-gui" and "--sleep-before-launching" are allowed when on "--continue-with-game-in-dir" mode. 


## Project Structure

The codebase is organized in a modular structure to ensure maintainability and separation of concerns:

- `/core`: Core game engine components
  - `game_controller.py`: Base game logic that can run with or without a GUI
  - `game_logic.py`: Snake game implementation with LLM integration
  - `game_data.py`: Game state tracking and statistics management
  
- `/gui`: Graphical user interface components
  - `base.py`: Base UI classes
  - `game_gui.py`: Game-specific UI implementation
  - `replay_gui.py`: UI for replaying saved games
  
- `/utils`: Utility modules
  - `file_utils.py`: File management utilities
  - `game_manager_utils.py`: Game session management helpers
  - `game_stats_utils.py`: Statistics gathering and reporting
  - `json_utils.py`: JSON processing and validation
  - `llm_utils.py`: LLM response handling
  - `log_utils.py`: Logging utilities
  - `replay_utils.py`: Game replay functionality
  - `text_utils.py`: Text processing for LLM responses
  
- `/replay`: Replay functionality
  - `replay_engine.py`: Engine for replaying saved games
  
- Main modules:
  - `main.py`: Entry point with command-line argument parsing
  - `game_manager.py`: Manages game sessions and statistics
  - `llm_client.py`: Client for communicating with LLMs
  - `config.py`: Configuration constants

## Game Termination Conditions

The snake game will terminate under any of the following conditions:
1. Snake hits a wall (boundary of the game board)
2. Snake collides with its own body
3. Maximum steps limit is reached (default: 400 steps)
4. Three consecutive empty moves occur without ERROR
   - Empty moves are checked immediately when detected
   - An empty move occurs when the LLM returns `{"moves":[], "reasoning":"..."}`
   - If the reasoning contains "ERROR", the consecutive count is reset
5. A game error occurs
   - The system will catch errors, log them, and continue to the next game
   - Error information is saved in the game summary

After game termination (for any reason), the system will automatically start the next game until the maximum number of games is reached.


