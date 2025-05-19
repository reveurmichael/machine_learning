import os
import inspect
from selenium import webdriver
from .logger import logger
from .decorator import log_function_call

class ScreenshotService:
    def __init__(self, site_name: str = 'UnknownSite', driver: webdriver = None) -> None:
        self.screenshot_count: int = 0
        self.site_name: str = site_name
        self.driver: webdriver = driver

    @log_function_call
    def capture_screenshot(self, screenshot_dir: str) -> None:
        self.screenshot_count += 1
        # Get the name of the calling function
        frame = inspect.currentframe().f_back.f_back.f_back.f_back
        function_name = frame.f_code.co_name

        formatted_count = f"{self.screenshot_count:02}"
        screenshot_filename = f"{self.site_name}_{formatted_count}_{function_name}.png"
        screenshot_path = os.path.join(screenshot_dir, screenshot_filename)

        self.driver.save_screenshot(screenshot_path)
        logger.info(f"Screenshot saved to: {screenshot_path}")
