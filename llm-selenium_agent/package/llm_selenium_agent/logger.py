import logging
import os
from datetime import datetime
from .decorator import *

class Logger:
    """A class to handle logging for the llm_selenium_agent Selenium Downloaders. It's a singleton. Note that we cannot use the decorator on the methods of this class because it would create a circular dependency."""

    def __init__(
        self,
        log_format="%(asctime)s - %(levelname)s - %(message)s",
    ):
        self.logger = logging.getLogger("llm_selenium_agentSeleniumDownloadersLogger")
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(log_format)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.clear_existing_handlers()

    def clear_existing_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.add_console_handler()
        self.add_file_handler(".", "llm_selenium_agent.log")
        self.add_file_handler(self.current_date, f"llm_selenium_agent-{self.current_date}.log")

    def add_console_handler(self, level=logging.INFO):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def add_file_handler(self, directory: str, filename: str, level=logging.DEBUG):
        """Add a file handler to the logger. Note that the directory will be created if it doesn't exist. Usually it's here that the Date Folder is created."""
        os.makedirs(directory, exist_ok=True)
        log_file_path = os.path.join(directory, filename)
        try:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f"Failed to create file handler for {log_file_path}: {e}")

    def get_logger(self) -> logging.Logger:
        return self.logger


logger_instance = Logger()
logger = logger_instance.get_logger()

def count_warnings(log_file: str) -> int:
    """Count warnings in a log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            return sum(1 for line in file if "WARNING" in line or "[WARNING]" in line)
    except:
        return 0

def count_errors(log_file: str) -> int:
    """Count errors in a log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            return sum(1 for line in file if "ERROR" in line or "[ERROR]" in line)
    except:
        return 0


def count_info(log_file: str) -> int:
    """Count info in a log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            return sum(1 for line in file if "INFO" in line or "[INFO]" in line)
    except:
        return 0

def count_debug(log_file: str) -> int:
    """Count debug in a log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            return sum(1 for line in file if "DEBUG" in line or "[DEBUG]" in line)
    except:
        return 0
