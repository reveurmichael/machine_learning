import os
import sys
import time
import warnings
import shutil
from datetime import datetime
from datetime import date
from urllib.parse import urljoin
import pandas as pd

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .captcha import *
from .config import *
from .logger import *
from .network import *
from .screenshot import *
from .decorator import *
from .env import initialize_environment

warnings.filterwarnings("ignore")


class BaseSelenium:

    @log_function_call
    def __init__(self):
        self.config = load_configuration()
        self.downloads_dir = self.initialize_tmp_downloads_directory()
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.headless_mode = get_headless_mode()
        self.browser_name = "UnknownBrowser"
        self.twocaptcha_api_key = get_twocaptcha_api_key()
        self.site_name = self.get_site_name()
        self.login_success = False
        self.whole_process_success = False
        

    @log_function_call
    def prepare_environment(self):
        self.task_download_dir = self.initialize_task_download_directory()
        self.setup_logging()
        self.load_secret_credentials()

        self.copy_config_yml_file()
        self.initialize_webdriver()
        self.setup_screenshot_service()
        self.remove_existing_log_file() 

    @log_function_call
    def initialize_tmp_downloads_directory(self) -> str:
        """Initialize a temporary directory for downloads."""
        tmp_dir = os.path.join(os.getcwd(), "tmp_downloads")
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info(f"Temporary downloads directory: {tmp_dir}")
        return tmp_dir

    @log_function_call
    def initialize_task_download_directory(self) -> str:
        """Initialize a directory for task-specific downloads."""
        dir_name = f"{self.current_date}_{self.site_name}"
        download_dir = os.path.join(os.getcwd(), self.current_date, dir_name)
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Task download directory: {download_dir}")
        return download_dir

    @log_function_call
    def load_secret_credentials(self) -> None:
        """Load secret credentials from environment variables."""
        try:
            initialize_environment()
            logger.info("Secret credentials loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load secret credentials: {e}")

    def copy_config_yml_file(self):
        source_path = "config.yml"
        destination_path = os.path.join(self.task_download_dir, "config.yml")

        try:
            shutil.copy(source_path, destination_path)
            logger.info(f"Copied config.yml to {destination_path}")
        except Exception as e:
            logger.error(f"An error occurred while copying config.yml: {e}")

    @log_function_call
    def setup_logging(self) -> None:
        logger_instance.clear_existing_handlers()
        log_filename = f"{self.site_name}.log"
        logger_instance.add_file_handler(self.current_date, log_filename)
        self.logger_file_path = os.path.join(
            os.getcwd(), self.current_date, log_filename
        )

    @log_function_call
    def remove_existing_log_file(self) -> None:
        try:
            if os.path.isfile(self.logger_file_path):
                # Open the file in write mode to empty it
                with open(self.logger_file_path, 'w') as file:
                    pass  # Just opening and closing the file empties it
                logger.info(f"Emptied existing log file: {self.logger_file_path}")
            else:
                logger.warning(f"Warning: log file does not exist: {self.logger_file_path}")
        except Exception as e:
            logger.error(f"Error emptying log file {self.logger_file_path}: {e}")

    @log_function_call
    def get_site_name(self) -> str:
        return self.__class__.__name__

    @log_function_call
    def setup_screenshot_service(self):
        self.screenshot_service = ScreenshotService(self.site_name, self.driver)

    @log_function_call
    def capture_screenshot(self):
        screenshot_dir = self.task_download_dir
        self.screenshot_service.capture_screenshot(screenshot_dir)

    @log_function_call
    def main(self):
        self.prepare_environment()
        self.navigate_to_url()
        self.login()
        self.verify_login_success()
        self.terminate_webdriver()

    @log_function_call
    def navigate_to_url(self):
        self.driver.get(self.url)

    @log_function_call
    def initialize_webdriver_options(self):
        pass

    @log_function_call
    def initialize_webdriver(self):
        pass

    @log_function_call
    def login(self):
        pass

    @log_function_call
    def verify_login_success(self) -> bool:
        pass

    @log_function_call
    def terminate_webdriver(self):
        self.driver.quit()

    @log_function_call
    def navigate_to_relative_url(self, relative_href: str):
        current_url = self.driver.current_url
        logger.info(f"Current URL: {current_url}")

        new_url = urljoin(current_url, relative_href)
        logger.info(f"Navigating to: {new_url}")
        self.driver.get(new_url)

    @log_function_call
    def accept_alert_dialog(self):
        try:
            WebDriverWait(self.driver, 10).until(EC.alert_is_present())
            alert = self.driver.switch_to.alert
            alert.accept()
            logger.info("Accepted alert dialog.")
        except TimeoutException:
            logger.info("No alert dialog appeared within the timeout period.")
        except Exception as e:
            logger.exception(f"Error handling alert dialog: {e}")
