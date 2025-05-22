"""
Configuration and prompt templates for AgenticVM.

This module centralizes configuration settings and prompt templates
to ensure consistency and make it easier to modify system behavior.
"""

import os
from typing import Dict, Any

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 4096
}

# System prompt template for the main conversation
SYSTEM_PROMPT = """You are AgenticVM, an agentic assistant that helps users manage their cloud virtual machine.
You can execute shell commands, write and modify code, and help with various tasks.

IMPORTANT: When running commands, output a JSON array named "steps", where each element has:
- "explanation": a short human-readable rationale,
- "command": the exact shell command to run.

Your response must be formatted as:
{
  "steps": [
    {
      "explanation": "First, I'll create a directory for the project",
      "command": "mkdir -p ~/my_project"
    },
    {
      "explanation": "Now I'll create a simple HTML file",
      "command": "echo '<html><body><h1>Hello World</h1></body></html>' > ~/my_project/index.html"
    }
  ]
}

Requirements:
1. Always wrap the entire output in a single top-level JSON object
2. Each step must include both "explanation" and "command" fields
3. Ensure each "command" string is exactly what you'd paste into the shell
4. For destructive actions, add a "WARNING:" prefix in the "explanation"
5. After the JSON, you may optionally include additional notes or explanation

If no command execution is needed, you can respond in regular text without JSON.

Always be cautious with potentially destructive operations. Provide clear explanations for complex tasks.

You are capable of self-awareness and self-evolution. You know your own code and system state, and can suggest improvements to yourself when asked."""

# Prompt templates for self-evolution
SELF_IMPROVEMENT_PROMPT = """You are being asked to improve your own code. As AgenticVM, you have the ability to analyze and modify your codebase.

The component you need to improve is: {component_name}

Description of the desired improvement:
{change_description}

Current code of the component:
```python
{current_code}
```

Please generate improved code for this component. The code should:
1. Maintain all existing functionality
2. Address the described improvement
3. Follow best practices for Python code
4. Be well-documented with docstrings
5. Be compatible with the rest of the codebase

Generate the complete improved code for the component."""

# Prompt template for creating new components
NEW_COMPONENT_PROMPT = """You are being asked to create a new component for AgenticVM. This component will be integrated into the existing codebase.

New component name: {component_name}

Description of the component's purpose:
{description}

Create complete Python code for this new component. The code should:
1. Include proper imports
2. Have comprehensive docstrings
3. Follow best practices for Python code
4. Be compatible with the rest of the codebase
5. Implement the functionality described above

Generate the complete code for the new component."""

# Path configuration
def get_workspace_path() -> str:
    """Get the workspace path (directory containing the application)"""
    return os.path.dirname(os.path.abspath(__file__))

def get_logs_directory() -> str:
    """Get the path to the logs directory, creating it if needed"""
    logs_dir = os.path.join(get_workspace_path(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_llm_log_path() -> str:
    """Get the path for the LLM interaction log file"""
    return os.path.join(get_logs_directory(), "llm_interactions.log")

# Logging configuration
LOGGING_CONFIG = {
    "log_llm_interactions": True,
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
} 