![](./img/a.jpg)

# LLM Snake Game

A classic Snake game controlled by an LLM (Language Model).

## Installation

Set up API keys in a `.env` file:

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
- `--provider hunyuan`, `--provider ollama`, `--provider deepseek`, or `--provider mistral` - Choose the LLM provider
- `--model <model_name>` - Specify which model to use:
  - For Ollama: any available model 
  - For DeepSeek: `deepseek-chat` (default) or `deepseek-reasoner`
  - For Mistral: `mistral-medium-latest` (default) or `mistral-large-latest`
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
```

## How to improve the game performance

- Maybe, instead of giving multiple moves, LLM only gives one move. But this will drastically increase the time for the whole game playing, as well as the cost of the API calls.
- Or, there are many other insights/details to specify about snake game in the prompt. Those additional insights/details might help LLM figure out better moves.

## License

MIT 