#!/bin/bash

# Check if Python is installed (try both python and python3)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Neither python nor python3 is installed. Please install Python first."
    exit 1
fi

echo "Found Python command: ${PYTHON_CMD}"
which ${PYTHON_CMD}

# Check if pip is installed (try both pip and pip3)
PIP_CMD=""
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "Neither pip nor pip3 is installed. Please install pip first."
    exit 1
fi

echo "Found pip command: ${PIP_CMD}"
which ${PIP_CMD}

# Install dependencies
echo "Checking dependencies..."
if ! $PIP_CMD list | grep -q streamlit; then
    echo "Installing dependencies..."
    $PIP_CMD install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies. Please check your internet connection and try again."
        exit 1
    fi
    echo "Dependencies installed successfully."
else
    echo "Dependencies already installed."
fi

# Check for LLM configuration
if [ ! -f config.yml ] && [ -f config.example.yml ]; then
    echo "Creating configuration from example..."
    cp config.example.yml config.yml
    echo "Created configuration file. You may need to edit it for your specific setup."
    echo "NOTE: By default, Ollama is configured to connect to a remote server."
    echo "      Edit config.yml if you need to change the Ollama server address."
fi

sudo apt install python3.10-venv

# Check if .env file exists, if not, create it from example. If the .env file exists, start the application.
if [ ! -f .env ]; then
    if [ -f env.example ]; then
        echo "Creating .env file from example..."
        cp env.example .env
        echo "Please edit the .env file and add your API keys."
    else
        echo "Creating .env file..."
        echo "# API Keys for LLM providers" > .env
        echo "MISTRAL_API_KEY=" >> .env
        echo "DEEPSEEK_API_KEY=" >> .env
        echo "HUNYUAN_API_KEY=" >> .env
        echo "Please edit the .env file and add your API keys."
    fi
else
    echo "Starting AgenticVM..."
    streamlit run app.py --server.address=0.0.0.0
fi
