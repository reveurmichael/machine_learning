# How to play the game

```
python main.py --provider ollama --model deepseek-r1:14b --parser-provider ollama --parser-model mistral:7b
```

## Features

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
