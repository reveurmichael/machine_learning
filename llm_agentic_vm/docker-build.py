"""
Docker build and management script for AgenticVM
This script provides an interactive way to build and manage AgenticVM Docker containers.
"""

import os
import subprocess
import shutil
import argparse
import time
import sys
from typing import List, Optional

def run_command(command: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result"""
    print(f"Running command: {' '.join(command)}")
    return subprocess.run(command, check=check, text=True, capture_output=True)

def check_docker_installed() -> bool:
    """Check if Docker and Docker Compose are installed"""
    try:
        run_command(["docker", "--version"])
        run_command(["docker-compose", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_config_files() -> None:
    """Setup configuration files if they don't exist"""
    # Check and create .env file
    if not os.path.exists(".env"):
        print("Creating .env file...")
        if os.path.exists("env.example"):
            shutil.copy("env.example", ".env")
            print("Created .env file from example. Please edit it with your API keys.")
        else:
            with open(".env", "w") as f:
                f.write("# API Keys for LLM providers\n")
                f.write("MISTRAL_API_KEY=\n")
                f.write("DEEPSEEK_API_KEY=\n")
                f.write("HUNYUAN_API_KEY=\n")
            print("Created blank .env file. Please add your API keys.")
    
    # Check and create config.yml file
    if not os.path.exists("config.yml"):
        print("Creating config.yml file...")
        if os.path.exists("config.example.yml"):
            shutil.copy("config.example.yml", "config.yml")
            print("Created config.yml file from example.")
        else:
            print("Warning: config.example.yml not found. The container will use default settings.")

def build_containers() -> None:
    """Build the Docker containers"""
    print("Building Docker containers...")
    run_command(["docker-compose", "build"])

def start_containers(detach: bool = True) -> None:
    """Start the Docker containers"""
    print("Starting Docker containers...")
    cmd = ["docker-compose", "up"]
    if detach:
        cmd.append("-d")
    run_command(cmd, check=False)  # Don't check return code as it blocks when not detached

def stop_containers() -> None:
    """Stop the Docker containers"""
    print("Stopping Docker containers...")
    run_command(["docker-compose", "down"])

def show_logs(service: str = "agentic-vm", follow: bool = True) -> None:
    """Show logs for a service"""
    print(f"Showing logs for {service}...")
    cmd = ["docker-compose", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(service)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped log viewing")

def check_container_status() -> None:
    """Check the status of the containers"""
    print("Checking container status...")
    run_command(["docker-compose", "ps"])

def main() -> None:
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Docker build and management script for AgenticVM")
    parser.add_argument("action", choices=["build", "start", "stop", "restart", "logs", "status", "setup"],
                      help="Action to perform")
    parser.add_argument("--detach", "-d", action="store_true", help="Run containers in detached mode")
    parser.add_argument("--service", "-s", default="agentic-vm", help="Service to perform action on")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs output")

    args = parser.parse_args()

    # Check if Docker is installed
    if not check_docker_installed():
        print("Error: Docker and/or Docker Compose are not installed.")
        print("Please install Docker and Docker Compose before continuing.")
        sys.exit(1)

    # Create the logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Perform the requested action
    if args.action == "setup":
        setup_config_files()
    elif args.action == "build":
        setup_config_files()
        build_containers()
    elif args.action == "start":
        setup_config_files()
        start_containers(args.detach)
        if args.detach:  # Only show status if detached
            time.sleep(2)  # Wait for containers to start
            check_container_status()
            print(f"AgenticVM is now running at http://localhost:8501")
    elif args.action == "stop":
        stop_containers()
    elif args.action == "restart":
        stop_containers()
        start_containers(args.detach)
        if args.detach:
            time.sleep(2)
            check_container_status()
    elif args.action == "logs":
        show_logs(args.service, args.follow)
    elif args.action == "status":
        check_container_status()

if __name__ == "__main__":
    main() 