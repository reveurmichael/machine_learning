import yaml
import os
from .logger import logger, log_function_call
from typing import Dict, Any, List, Optional
import pandas as pd
import re
import platform


def get_chrome_binary_location() -> Optional[str]:
    def find_chrome_on_windows() -> Optional[str]:
        potential_paths = [
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        for path in potential_paths:
            if os.path.isfile(path):
                return path
        return None

    def find_chrome_on_macos() -> Optional[str]:
        potential_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/local/bin/google-chrome",
            "/usr/bin/google-chrome",
            "/opt/homebrew/bin/google-chrome",
        ]
        for path in potential_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    system = platform.system()
    chrome_path = None

    if system == "Windows":
        chrome_path = find_chrome_on_windows()
    elif system == "Darwin":
        chrome_path = find_chrome_on_macos()
    else:
        logger.warning(
            f"Chrome binary location: Unsupported operating system: {system}"
        )

    if chrome_path:
        current_config_path = (
            load_configuration().get("selenium", {}).get("chrome_binary_location")
        )
        if chrome_path != current_config_path:
            config = load_configuration()
            config.setdefault("selenium", {})["chrome_binary_location"] = chrome_path
            try:
                update_configuration(config)
                logger.info(
                    f"Updated Chrome binary location in config.yml: {chrome_path}"
                )
            except Exception as e:
                logger.error(f"Failed to update config.yml with new Chrome path: {e}")
        return chrome_path
    else:
        logger.error("Chrome binary not found in typical locations.")
        return None


def get_firefox_binary_location() -> Optional[str]:
    def find_firefox_on_windows() -> Optional[str]:
        potential_paths = [
            "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
            "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe",
            os.path.expandvars(r"%APPDATA%\Mozilla Firefox\firefox.exe"),
            os.path.expanduser(r"~\AppData\Local\Mozilla Firefox\firefox.exe"),
        ]
        for path in potential_paths:
            if os.path.isfile(path):
                return path
        return None

    def find_firefox_on_macos() -> Optional[str]:
        potential_paths = [
            "/Applications/Firefox.app/Contents/MacOS/firefox",
            "/usr/local/bin/firefox",
            "/opt/homebrew/bin/firefox", 
        ]
        for path in potential_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    system = platform.system()
    firefox_path = None

    if system == "Windows":
        firefox_path = find_firefox_on_windows()
    elif system == "Darwin":
        firefox_path = find_firefox_on_macos()
    else:
        logger.warning(f"Firefox binary location: Unsupported operating system: {system}")

    if firefox_path:
        current_config_path = (
            load_configuration().get("selenium", {}).get("firefox_binary_location")
        )
        if firefox_path != current_config_path:
            config = load_configuration()
            config.setdefault("selenium", {})["firefox_binary_location"] = firefox_path
            try:
                update_configuration(config)
                logger.info(
                    f"Updated Firefox binary location in config.yml: {firefox_path}"
                )
            except Exception as e:
                logger.error(f"Failed to update config.yml with new Firefox path: {e}")
        return firefox_path
    else:
        logger.error("Firefox binary not found in typical locations.")
        return None

def get_streamlit_server_address() -> str:
    """Get the address of the Streamlit server."""
    return load_configuration().get("streamlit", {}).get("address", "127.0.0.1")

def get_streamlit_server_port() -> str:
    """Get the port of the Streamlit server."""
    return load_configuration().get("streamlit", {}).get("port", "8501")

def get_streamlit_allow_run_on_save() -> str:
    """Get the allow_run_on_save option of the Streamlit server."""
    run_on_save_value = (
        load_configuration().get("streamlit", {}).get("allow_run_on_save", False)
    )

    if isinstance(run_on_save_value, str):
        return run_on_save_value.lower() == "true"
    return bool(run_on_save_value)


def get_chrome_driver_path() -> str:
    """Get the path to the ChromeDriver."""
    return load_configuration().get("selenium", {}).get("chrome_driver_path")

def get_firefox_driver_path() -> str:
    """Get the path to the FirefoxDriver."""
    return load_configuration().get("selenium", {}).get("firefox_driver_path")

def get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return "config.yml"


def load_configuration() -> Dict[str, Any]:
    """Load the configuration file as a dictionary."""
    config_path = get_config_file_path()
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}
    else:
        logger.error("Configuration file does not exist.")
        config = {}
    return config


def get_headless_mode() -> bool:
    """Retrieve the headless option for Selenium."""
    headless_value = load_configuration().get("selenium", {}).get("headless", False)

    # Normalize the headless value to a boolean
    if isinstance(headless_value, str):
        return headless_value.lower() == "true"
    return bool(headless_value)


def update_configuration(config: Dict[str, Any]) -> None:
    """Update the configuration file.

    Sorts the websites by task ID (index) numerically by default. If `sort_by_task_id` is set to False,
    it will sort the websites alphabetically by their names.

    Args:
        config (Dict[str, Any]): The configuration dictionary to update.
    """
    with open(get_config_file_path(), "w") as file:
        yaml.dump(config, file, sort_keys=False)

