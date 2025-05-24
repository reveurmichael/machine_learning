#!/bin/bash

# Exit on error
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "Created .env file from example. Please edit it with your API keys before running the application."
    else
        cat > .env << EOF
# API Keys for LLM providers
MISTRAL_API_KEY=
DEEPSEEK_API_KEY=
HUNYUAN_API_KEY=
EOF
        echo "Created blank .env file. Please add your API keys before running the application."
    fi
fi

# Check if config.yml file exists
if [ ! -f config.yml ]; then
    echo "Creating config.yml file from example..."
    if [ -f config.example.yml ]; then
        cp config.example.yml config.yml
        echo "Created config.yml file from example."
    else
        echo "Warning: config.example.yml not found. The container will use default settings."
    fi
fi

# Build and start the containers
echo "Building and starting Docker containers..."
docker-compose up --build -d

# Check if containers are running
echo "Checking container status..."
docker-compose ps

# Print the URL
echo "AgenticVM is now running at http://localhost:8501" 