"""
Example Selenium automation script using llm_selenium_agent.
This demonstrates how to create a simple automation task.
"""

from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class GoogleSearch(BaseSeleniumChrome):
    def __init__(self):
        super().__init__()
        self.url = "https://www.google.com"
        
    def login(self):
        """
        No login needed for Google search example.
        This method is required by the BaseSelenium class.
        """
        pass
        
    def search_for_term(self, search_term):
        """
        Perform a Google search for the given term.
        """
        # Wait for the search box to be present
        search_box = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        
        # Type the search term
        search_box.send_keys(search_term)
        
        # Take a screenshot before submitting
        self.capture_screenshot()
        
        # Submit the search
        search_box.submit()
        
        # Wait for results to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "search"))
        )
        
        # Take a screenshot of the results
        self.capture_screenshot()
        
    def verify_login_success(self):
        """
        No login verification needed for this example.
        This method is required by the BaseSelenium class.
        """
        return True

if __name__ == "__main__":
    # Create an instance of the GoogleSearch class
    automation = GoogleSearch()
    
    try:
        # Set up the environment and navigate to the URL
        automation.prepare_environment()
        automation.navigate_to_url()
        
        # Perform the search
        automation.search_for_term("Selenium WebDriver tutorial")
        
        # Wait a few seconds to see the results
        import time
        time.sleep(5)
        
    finally:
        # Always terminate the webdriver to clean up
        automation.terminate_webdriver()
        
    print("✅ Automation completed successfully!")
    print(f"📸 Screenshots saved in: {automation.task_download_dir}") 