#!/usr/bin/env python3
import os
import platform
import subprocess
import sys

def run_command(command):
    """Run a shell command and handle errors"""
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_docker():
    """Check if Docker is installed and running"""
    if run_command("docker --version"):
        print("Docker is installed.")
    else:
        print("Docker is not installed or not in PATH. Please install Docker to continue.")
        sys.exit(1)
    
    # Check if Docker is running
    if platform.system() == "Windows":
        # Windows check
        result = run_command("docker info > nul 2>&1")
    else:
        # Linux/Mac check
        result = run_command("docker info > /dev/null 2>&1")
    
    if not result:
        print("Docker daemon is not running. Please start Docker Desktop or Docker service.")
        sys.exit(1)
    
    print("Docker is running.")

def build_cv():
    """Build the Docker image and generate the PDF"""
    print("Building Docker image...")
    if not run_command("docker compose build"):
        print("Failed to build Docker image.")
        sys.exit(1)
    
    print("Running container to generate PDF...")
    if not run_command("docker compose run --rm cv-builder"):
        print("Failed to generate PDF.")
        sys.exit(1)
    
    # Check if PDF was generated
    if os.path.exists("main.pdf"):
        print("CV built successfully. PDF file is available as main.pdf")
    else:
        print("PDF generation seems to have failed. main.pdf not found.")

if __name__ == "__main__":
    check_docker()
    build_cv() 