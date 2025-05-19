import yaml
import sys
from .config import *
from .logger import *
from .network import *
from .env import *
import llm_selenium_agent.streamlit_app.app
import os
import inspect
import subprocess
from .decorator import log_function_call
import argparse
import socket


@log_function_call
def perform_initial_setup():
    """
    Perform initial setup for the LLM Selenium Agent.
    This includes checking for config.yml, installing Chrome and Firefox drivers,
    and setting up environment variables.
    """
    print("🔧 Starting LLM Selenium Agent initial setup...")
    config_path = get_config_file_path()

    chrome_driver_path = (
        load_configuration().get("selenium", {}).get("chrome_driver_path")
    )

    firefox_driver_path = (
        load_configuration().get("selenium", {}).get("firefox_driver_path")
    )

    if not os.path.isfile(config_path):
        # Create a basic config file
        config = {
            "selenium": {
                "headless": False,
                "chrome_options": {
                    "ignore-certificate-errors": "true"
                }
            },
            "streamlit": {
                "address": "127.0.0.1",
                "port": "8501",
                "allow_run_on_save": False
            }
        }
        update_configuration(config)
        print(f"✅ Created a basic {config_path} file with default settings.")
    else:
        print(f"✅ {config_path} is present.")

    # Check for ChromeDriver
    if chrome_driver_path and os.path.exists(chrome_driver_path):
        print(f"✅ ChromeDriver is already installed at {chrome_driver_path}")
    else:
        print("⏳ Installing ChromeDriver (this may take a moment)...")
        success, message, path = install_chrome_driver()
        if success == 1:
            print(f"✅ {message} Location: {path}")
        else:
            print(f"❌ {message}")
            print("This might be due to network issues. You may need to use a VPN if you're in a restricted network environment.")

    # Check for FirefoxDriver
    if firefox_driver_path and os.path.exists(firefox_driver_path):
        print(f"✅ FirefoxDriver is already installed at {firefox_driver_path}")
    else:
        print("⏳ Installing FirefoxDriver (this may take a moment)...")
        success, message, path = install_firefox_driver()
        if success == 1:
            print(f"✅ {message} Location: {path}")
        else:
            print(f"❌ {message}")
            print("This might be due to network issues. You may need to use a VPN if you're in a restricted network environment.")

    # Check for .env file
    if not is_env_file_present():
        print("⚠️ .env file not found. Creating a template .env file...")
        with open(".env", "w") as f:
            f.write("# LLM Selenium Agent environment variables\n")
            f.write("# Add your API keys below if needed\n\n")
        print("✅ Created a template .env file.")
    else:
        print("✅ .env file is present.")

    print("\n🎉 Setup complete! You can now run the Streamlit app with:")
    print("   llm_selenium_agent_streamlit_app")


@log_function_call
def launch_streamlit_app():
    """
    Launch the Streamlit app for LLM Selenium Agent.
    This function handles port allocation and server configuration.
    """
    def find_available_port(start_port: int) -> int:
        """Find an available port starting from start_port."""
        _port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    if s.connect_ex(("localhost", _port)) != 0:  # Port is available
                        return _port
                    else:
                        _port += 1
                except Exception as e:
                    logger.info(
                        f"An exception occurred while finding an available port: {e}"
                    )
                    _port += 1

    python_file_path = os.path.abspath(
        inspect.getfile(llm_selenium_agent.streamlit_app.app)
    )
    address = get_streamlit_server_address()
    port = str(get_streamlit_server_port())  # Ensure port is a string
    port = str(find_available_port(int(port)))  # Convert to string
    allow_run_on_save = str(get_streamlit_allow_run_on_save()).lower()

    print(f"🚀 Launching LLM Selenium Agent Streamlit app on {address}:{port}...")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        python_file_path,
        "--server.address",
        address,
        "--server.port",
        port,
        "--server.allowRunOnSave",
        allow_run_on_save,
        "--client.toolbarMode",
        "minimal",
        "--browser.gatherUsageStats",
        "false",
    ]

    # Ensure all command elements are strings
    command = [str(arg) for arg in command]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while launching Streamlit app: {e}")
        print(f"❌ An error occurred while launching the Streamlit app: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(f"Command list: {command}")
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)


@log_function_call
def llm_selenium_agent_launch_streamlit_app():
    """
    Entry point for launching the Streamlit app from the command line.
    """
    print("🤖 LLM Selenium Agent - Streamlit Interface")
    launch_streamlit_app()

