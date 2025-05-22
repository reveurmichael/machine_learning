# AgenticVM

AgenticVM is an agentic virtual machine assistant powered by multiple LLM backends including Ollama (local) and cloud providers like Mistral AI, DeepSeek, and Hunyuan. It allows users to manage a cloud virtual machine through natural language, serving as an educational tool for students learning about LLMs and agentic AI.

## Enhanced Features

- **Multiple LLM Backends**: Configure and use both local (Ollama) and cloud-based LLM providers
- **Background Process Management**: Run, monitor, and control background processes
- **System Monitoring**: Live monitoring of system resources and performance
- **Advanced File Operations**: File search, content analysis, archiving, and extraction
- **Code Analysis and Documentation**: Analyze code, generate documentation, and create project scaffolding
- **Git Integration**: Complete git workflow support for version control
- **Environment Management**: Manage virtual environments and package installations
- **Network Tools**: Port scanning and service discovery
- **Self-Awareness**: System introspection capabilities, including environment analysis and code understanding
- **Self-Evolution**: Ability to modify and extend itself through code generation and application

## Prerequisites

- Python 3.7+
- For local LLM usage:
  - Ollama (installed and running)
- For cloud LLM usage:
  - Mistral AI, DeepSeek, or Hunyuan API key

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
streamlit run app.py
```

2. Open the provided URL in your browser (typically http://localhost:8501)
3. Configure your LLM provider in the sidebar:
   - For local usage, configure Ollama
   - For cloud usage, enter your API key and select a model
4. Start interacting with AgenticVM by typing natural language prompts

## Example Prompts

Here are some example prompts you can use with AgenticVM:

- "Check the current system status and available disk space"
- "Install the pandas and matplotlib packages"
- "Create a new folder called 'data' and navigate to it"
- "Write a Python script that calculates the factorial of a number"
- "Clone a git repository from GitHub"
- "Create a new Python virtual environment and install requests package"
- "Show me a list of running processes and how much memory they're using"
- "Generate a requirements.txt file for this project"
- "Create a basic Python package structure for a new project called 'data_analyzer'"
- "Execute a Python script that prints the current date and time"

## Architecture

AgenticVM consists of the following components:

- `app.py`: Main Streamlit application and user interface
- `shell.py`: Core shell execution functionality with enhanced system operations
- `code_utils.py`: Advanced code and project management utilities
- `llm_client.py`: Flexible LLM client supporting multiple providers and models
- `self_awareness.py`: System introspection and codebase analysis functionality
- `self_evolution.py`: Code modification and extension capabilities
- `self_evolving_vm.py`: Integration of self-awareness and self-evolution features

The system is designed with minimal predefined code to maximize flexibility and learning opportunities.

## Educational Value

AgenticVM provides students with:

1. Hands-on experience with agentic AI systems
2. Understanding of LLM capabilities and limitations
3. Learning about system administration through natural language
4. Practical experience with local and cloud-based LLMs
5. Code generation, analysis and execution in multiple languages
6. Project structure and best practices in software development
7. Learning about self-evolving AI systems that can modify their own code

## Security Notice

AgenticVM executes shell commands directly on the host system. This is intended for educational use in isolated environments. Use with caution and never expose the application to the public internet without proper security measures. For production use, implement strict access controls and command filtering.
