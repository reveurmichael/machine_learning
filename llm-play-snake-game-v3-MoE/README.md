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

or simply:

```
python main.py --provider ollama --model deepseek-r1:7b
```

This will bypass the secondary LLM and use the primary LLM's output directly.

## Installation

Set up API keys in a `.env` file:

```
# Ollama settings (optional, defaults to localhost)
OLLAMA_HOST=localhost:11434

# API keys for cloud LLM providers (required for respective providers)
HUNYUAN_API_KEY=your_hunyuan_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
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
- `--max-empty-moves`: Maximum consecutive empty moves before game over
- `--no-gui`: Run without the graphical interface (text-only mode)
- `--log-dir`: Directory to store log files

## Project Structure

The codebase is organized in a modular structure to ensure maintainability and separation of concerns:

- `/core`: Core game engine components
- `/gui`: Graphical user interface components
- `/llm`: Language model integration  
- `/utils`: Utility modules  
- `/replay`: Replay functionality
  
- Main modules:
  - `main.py`: Entry point with command-line argument parsing
  - `config.py`: Configuration constants
  - `app.py`: Streamlit dashboard for analyzing game statistics and replaying games
  - `replay.py`: Command-line interface for replaying saved games
  - `human_play.py`: Human-playable version of the Snake game

## Data Output

The system generates structured output for each game session:

- `game_N.json`: Contains complete data for game number N, including moves, statistics, and time metrics
- `summary.json`: Contains aggregated statistics for the entire session
- `prompts/`: Directory containing all prompts sent to the LLMs
- `responses/`: Directory containing all responses received from the LLMs

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


## How this project resembles a real research project

logging into json files: very comprehensive. Keep track of almost everything, maybe it can be useful, somehow.

--no-gui: it's like a headless mode. Visual + non visual mode.

replay mode: for double checking results.

app.py: for analyzing preliminary results, giving insights and adjust experiment parameters.

fail fast, going lean: ```python main.py --provider ollama --model mistral:7b --parser-provider ollama --parser-model mistral:7b --max-games 1 --no-gui --sleep-before-launching 1 --max-steps 3 --max-consecutive-errors-allowed 0```, instead of doing a complete run each time.


## What's missing, if this were a real research project

Launching the application can be in a more "script" way and often parallelized, instead of using command line arguments. 

Helper script for checking GPU usage, and then launching the application accordingly. 

Logging system with log files (DEBUG, INFO, WARNING, ERROR, CRITICAL).

Data processing and analysis.












