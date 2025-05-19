# LLM-Guided Web Navigator for Quotes.toscrape.com

This project demonstrates how to use a local or cloud LLM to guide web navigation on quotes.toscrape.com using the llm_selenium_agent package. The LLM analyzes the page structure and suggests navigation actions based on user input rules.

## Features

- LLM-guided web navigation
- Support for multiple LLM providers (Tencent Hunyuan and Ollama)
- Interactive user input for custom navigation rules every two actions
- Login/logout functionality with proper credentials
- Screenshot capturing for debugging
- Debug logging of each round's prompts and responses

## Setup

### Prerequisites

- Python 3.8+
- Chrome browser
- Either Ollama installed locally or a Tencent Hunyuan API key

### Installation

1. Clone the repository and navigate to the project directory

2. Edit the `.env` file with your API credentials:

   ```
   HUNYUAN_API_KEY=<your_hunyuan_api_key_here>
   OLLAMA_HOST=<your_ollama_host_ip_address>
   DEEPSEEK_API_KEY=<your_deepseek_api_key_here>
   MISTRAL_API_KEY=<your_mistral_api_key_here>
   ```

## Usage

### Basic Usage

Run the navigator with default settings (using Hunyuan LLM):

```bash
python navigator.py
```

Run with Ollama instead:

```bash
python navigator.py --provider ollama --model <your_ollama_model_name>
```

### Command-line Options

- `--provider`: LLM provider to use (`hunyuan`, `ollama`, `deepseek`, `mistral`, default is `hunyuan`)
- `--model`: LLM model to use
- `--max-actions`: Maximum number of actions to take (default: 50)
- `--headless`: Run the browser in headless mode
- You can also add any number of additional rules as positional arguments

Example:

```bash
python navigator.py --provider ollama --model deepseek-r1:7b  --max-actions 20 "Prefer visiting author pages" "Look for tags related to love or happiness"
```

### Interactive Rules

After every two actions, the program will prompt you to enter a new rule. This rule will be treated as the highest priority command for the LLM. For example:

- "login now please"
- "go to the next page"
- "logout"
- "look for quotes by Albert Einstein"
- "if the page contains the word LOVE, then filter by that tag"

### Debug Logging

The navigator automatically logs each round of interaction with the LLM:

- Each round's prompt and response are saved in a timestamped debug directory
- Files are organized in subdirectories named `round_1`, `round_2`, etc.
- For each round, two files are saved:
  - `prompt.txt`: The complete prompt sent to the LLM
  - `response.txt`: The LLM's full response
- The same directory also contains screenshots taken during navigation

This logging helps in understanding the LLM's decision-making process and debugging any issues.

## Project Structure

- `navigator.py`: Main module that ties everything together
- `selenium_driver.py`: Selenium driver module for web navigation
- `llm_client.py`: LLM client module for handling communication with LLMs
- `config.py`: Configuration settings for the project 