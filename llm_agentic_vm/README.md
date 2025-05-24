# AgenticVM

AgenticVM is an agentic virtual machine assistant powered by multiple LLM backends including Ollama (local) and cloud providers like Mistral AI, DeepSeek, and Hunyuan. It allows users to manage a cloud virtual machine through natural language, serving as an educational tool for students learning about LLMs and agentic AI.

## Installation

1. Clone this repository or copy the code to your cloud VM
2. Install the required dependencies:

```bash
cd agentic_cloud_machine
pip install -r requirements.txt
```

3. Create a `.env` file in the project directory with your API key(s):

```
MISTRAL_API_KEY=your_mistral_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
HUNYUAN_API_KEY=your_hunyuan_api_key_here
```

4. (Optional) Copy the LLM configuration example to create your own configuration:

```bash
cp llm_config.example.yml llm_config.yml
```

## Usage

1. Start the AgenticVM application:

```bash
cd agentic_cloud_machine
sh run.sh
```

2. Open the provided URL in your browser (typically http://localhost:8501)

3. Start interacting with AgenticVM by typing natural language prompts

## Security Notice

AgenticVM executes shell commands directly on the host system. This is intended for educational use in isolated environments. Use with caution and never expose the application to the public internet without proper security measures. For production use, implement strict access controls and command filtering.
