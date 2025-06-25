# AgenticVM

AgenticVM is an agentic virtual machine assistant powered by multiple LLM backends including Ollama (local) and cloud providers like Mistral AI, DeepSeek, and Hunyuan. It allows users to manage a cloud virtual machine through natural language, serving as an educational tool for students learning about LLMs and agentic AI.

## Option 1:  Standard Installation on VM

1. Clone this repository or copy (e.g. via `scp`) the code to your cloud VM

2. Run the script to install the dependencies:

```bash
sh vm-run.sh
```

3. Modify the `.env` file in the project directory with your API key(s):

```
MISTRAL_API_KEY=your_mistral_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
HUNYUAN_API_KEY=your_hunyuan_api_key_here
```

4. Modify the `config.yml` file

5. Start the AgenticVM application, always with `sh vm-run.sh`

6. Open the provided URL in your browser (http://<your_vm_ip>:8501) to start interacting with AgenticVM by typing natural language prompts


## Option 2: Docker Installation (Recommended)

1. Use the provided script to set up and start the Docker containers:

```bash
python docker-build.py start
```

The Python script provides more options:
```
python docker-build.py build     # Build the containers
python docker-build.py start     # Start the containers
python docker-build.py stop      # Stop the containers
python docker-build.py restart   # Restart the containers
python docker-build.py logs      # View the logs
python docker-build.py status    # Check container status
```

3. Open the provided URL in your browser: http://localhost:8501


## Security Notice

AgenticVM executes shell commands directly on the host system. This is intended for educational use in isolated environments. Use with caution and never expose the application to the public internet without proper security measures. For production use, implement strict access controls and command filtering.

