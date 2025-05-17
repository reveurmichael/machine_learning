"""
Selenium WebDriver wrapper for the code generation system.
Uses llm_selenium_agent BaseSeleniumChrome as the base class.
This provides only the basic WebDriver functionality, as all actions
should come from LLM-generated code.
"""

import os
import time
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from llm_selenium_agent import BaseSeleniumChrome
from colorama import Fore, Style

# Default timeout for Selenium waits
SELENIUM_TIMEOUT = 10

class SeleniumDriver(BaseSeleniumChrome):
    """Selenium driver for web automation using llm_selenium_agent.
    Provides only basic WebDriver functionality - all actions come from LLM-generated code.
    """
    
    def __init__(self, headless: bool = False, website: str = "https://quotes.toscrape.com"):
        """Initialize the Selenium driver.
        
        Args:
            headless: Whether to run the browser in headless mode
            website: The website to interact with
        """
        # Initialize the parent class
        super().__init__()
        
        # Set headless mode
        self.headless_mode = headless
        
        # Set the base URL
        self.base_url = website
        self.url = website
        
        # Create directories for debug info
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.debug_dir = f"selenium_debug_{timestamp}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create directory for screenshots
        self.screenshots_dir = os.path.join(self.debug_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def navigate_to_url(self, url: Optional[str] = None):
        """Navigate to a URL, using the base_url if none is provided.
        
        Args:
            url: The URL to navigate to (defaults to base_url)
        """
        # Set the url attribute expected by the parent class
        self.url = url if url else self.base_url
        # Call the parent method without arguments
        super().navigate_to_url()
        print(Fore.GREEN + f"✅ Navigated to {self.url}")
    
    def get_page_html(self, max_length: int = 30000) -> str:
        """Get the HTML of the current page, cleaned up with BeautifulSoup.
        
        Args:
            max_length: Maximum length of the HTML to return
            
        Returns:
            A string containing the HTML
        """
        try:
            # Get the HTML
            html = self.driver.page_source
            
            # Use BeautifulSoup to clean and format the HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get only the body content to reduce size
            body = soup.find('body')
            if body:
                return str(body)[:max_length]
            else:
                return str(soup)[:max_length]
            
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Error getting page HTML: {e}")
            return "<html><body><p>Error fetching HTML</p></body></html>"
    
    def take_screenshot(self, filename: str) -> None:
        """Take a screenshot.
        
        Args:
            filename: Name of the screenshot file
        """
        try:
            # Ensure filename has .png extension
            if not filename.endswith('.png'):
                filename = f"{filename}.png"
            
            screenshot_path = os.path.join(self.screenshots_dir, filename)
            self.driver.save_screenshot(screenshot_path)
            print(Fore.BLUE + f"📸 Screenshot saved to {screenshot_path}")
            
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Error taking screenshot: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                print(Fore.GREEN + "✅ WebDriver closed successfully")
            except Exception as e:
                print(Fore.YELLOW + f"⚠️ Error closing WebDriver: {e}") 