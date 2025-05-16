"""Selenium driver module for web navigation using llm_selenium_agent."""

import os
from datetime import datetime
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from llm_selenium_agent import BaseSeleniumChrome
from config import SELENIUM_TIMEOUT, SCREENSHOTS_ENABLED, LOGIN_CREDENTIALS

class SeleniumDriver(BaseSeleniumChrome):
    """Selenium driver for web navigation using llm_selenium_agent as base."""
    
    def __init__(self, headless: bool = False):
        """Initialize the Selenium driver.
        
        Args:
            headless: Whether to run the browser in headless mode
        """
        # Initialize the parent class
        super().__init__()
        
        # Store headless setting directly to headless_mode attribute used by parent
        self.headless_mode = headless
        self.base_url = "https://quotes.toscrape.com/"
        self.logged_in = False
        
        # Create directories for screenshots and debug info
        self.debug_dir = f"selenium_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        if SCREENSHOTS_ENABLED:
            self.screenshots_dir = os.path.join(self.debug_dir, "screenshots")
            os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def login(self):
        """Override the login method from BaseSeleniumChrome.
        Required by BaseSeleniumChrome but we'll use our own login_to_site method.
        """
        return self.login_to_site()
    
    def verify_login_success(self):
        """Verify if login was successful.
        Required by BaseSeleniumChrome.
        """
        return self.logged_in
    
    def navigate_to_url(self, url: Optional[str] = None):
        """Navigate to a URL, using the base_url if none is provided.
        
        Args:
            url: The URL to navigate to (defaults to base_url)
        """
        # Set the url attribute expected by the parent class
        self.url = url if url else self.base_url
        # Call the parent method without arguments
        super().navigate_to_url()
    
    def get_page_html_snippet(self, max_length: int = 300000) -> str:
        """Get a snippet of the current page's HTML.
        
        Args:
            max_length: Maximum length of the HTML snippet to return
            
        Returns:
            A string containing a portion of the page's HTML
        """
        html = self.driver.page_source
        return html
    
    def take_screenshot(self, filename: str) -> None:
        """Take a screenshot.
        
        Args:
            filename: Name of the screenshot file
        """
        if not SCREENSHOTS_ENABLED:
            return
            
        try:
            screenshot_path = os.path.join(self.screenshots_dir, filename)
            self.driver.save_screenshot(screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
    
    def navigate_to_next_page(self) -> bool:
        """Navigate to the next page of quotes.
        
        Returns:
            Boolean indicating success
        """
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, "li.next > a")
            next_button.click()
            
            # Wait for the new page to load
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Navigated to next page")
            self.take_screenshot("next_page.png")
            return True
        
        except NoSuchElementException:
            print("No 'Next' button found. This appears to be the last page.")
            return False
        
        except Exception as e:
            print(f"Error navigating to next page: {e}")
            return False
    
    def navigate_to_previous_page(self) -> bool:
        """Navigate to the previous page of quotes.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.driver.back()
            
            # Wait for the page to load
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Navigated back to previous page")
            self.take_screenshot("previous_page.png")
            return True
            
        except Exception as e:
            print(f"Error navigating to previous page: {e}")
            return False
    
    def visit_author_page(self, author_name: str) -> bool:
        """Visit an author's page.
        
        Args:
            author_name: The name of the author to visit
            
        Returns:
            Boolean indicating success
        """
        try:
            # Save current URL to return to later
            current_url = self.driver.current_url
            
            # Find the author link
            try:
                author_links = self.driver.find_elements(
                    By.XPATH,
                    f"//small[@class='author' and text()='{author_name}']/following-sibling::a",
                )
                
                if author_links:
                    author_links[0].click()
                else:
                    print(f"Couldn't find link for author: {author_name}")
                    return False
            
            except NoSuchElementException:
                print(f"Couldn't find author link for: {author_name}")
                return False
            
            # Wait for author details to load
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "author-details"))
            )
            
            print(f"Visited author page for: {author_name}")
            
            # Take a screenshot
            self.take_screenshot(f"author_{author_name.replace(' ', '_')}.png")
            
            # Return to the original page
            self.driver.get(current_url)
            
            # Wait for the original page to load
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            return True
                
        except Exception as e:
            print(f"Error visiting author page: {e}")
            return False
    
    def filter_by_tag(self, tag: str) -> bool:
        """Filter quotes by a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            Boolean indicating success
        """
        try:
            # Find the tag link
            tag_links = self.driver.find_elements(
                By.XPATH, f"//a[@class='tag' and text()='{tag}']"
            )
            
            if not tag_links:
                # Try to navigate to the tags page
                try:
                    # Go to the home page first
                    self.driver.get(self.base_url)
                    
                    # Wait for the page to load
                    WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "quote"))
                    )
                    
                    # Now try again to find the tag
                    tag_links = self.driver.find_elements(
                        By.XPATH, f"//a[@class='tag' and text()='{tag}']"
                    )
                    
                    if not tag_links:
                        print(f"Tag '{tag}' not found on the current page.")
                        return False
                    
                except Exception as e:
                    print(f"Error navigating to find tag: {e}")
                    return False
            
            # Click the tag link
            tag_links[0].click()
            
            # Wait for the filtered page to load
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Filtered quotes by tag: {tag}")
            self.take_screenshot(f"tag_{tag}.png")
            return True
            
        except Exception as e:
            print(f"Error filtering by tag: {e}")
            return False
    
    def login_to_site(self) -> bool:
        """Log in to the quotes.toscrape.com website.
        
        Returns:
            Boolean indicating success
        """
        if self.logged_in:
            print("Already logged in.")
            return True
            
        try:
            print("Attempting to login to quotes.toscrape.com...")

            # Navigate to the login page
            login_url = "https://quotes.toscrape.com/login"
            self.driver.get(login_url)

            # Take a screenshot at the login page
            self.take_screenshot("login_page.png")

            # Wait for the login form to be available
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form[action='/login']"))
            )

            # Find username and password fields and submit button
            try:
                username_field = self.driver.find_element(By.ID, "username")
                password_field = self.driver.find_element(By.ID, "password")
                submit_button = self.driver.find_element(
                    By.CSS_SELECTOR, "input[type='submit']"
                )

                print("Login form elements found successfully.")
            except Exception as e:
                print(f"Error finding login form elements: {e}")
                raise

            # Get credentials from config
            username = LOGIN_CREDENTIALS["username"]
            password = LOGIN_CREDENTIALS["password"]

            print(f"Using credentials - Username: {username}, Password: {password}")

            # Enter credentials
            username_field.clear()
            username_field.send_keys(username)
            password_field.clear()
            password_field.send_keys(password)

            # Take a screenshot before submitting
            self.take_screenshot("before_login_submit.png")

            # Submit the form
            print("Submitting login form...")
            submit_button.click()

            # Wait for page to load after login
            try:
                WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "quote"))
                )
                print("Page loaded after login attempt.")
            except Exception as e:
                print(f"Error waiting for page to load after login: {e}")

            # Take a screenshot after form submission
            self.take_screenshot("after_login_submit.png")

            # Verify login success (check for logout link)
            try:
                # Use a more specific XPath to find the logout link
                logout_link = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//a[@href='/logout' and text()='Logout']")),
                    message="Logout link not found"
                )
                
                if logout_link:
                    print("Login successful! Logout link found.")
                    self.logged_in = True

                    # Take a screenshot after successful login
                    self.take_screenshot("login_successful.png")
                    return True
            except Exception as e:
                print(f"Login failed - no logout link found: {e}")
                # Additional information about the page after failed login
                print(f"Current URL after login attempt: {self.driver.current_url}")
                try:
                    body_text = self.driver.find_element(By.TAG_NAME, "body").text
                    print(f"Page content snippet: {body_text[:200]}...")
                except:
                    pass

                self.logged_in = False
                return False

        except Exception as e:
            print(f"Error during login process: {e}")
            self.take_screenshot("login_error.png")
            return False
    
    def logout_from_site(self) -> bool:
        """Log out from the quotes.toscrape.com website.
        
        Returns:
            Boolean indicating success
        """
        if not self.logged_in:
            print("Not currently logged in.")
            return False
            
        try:
            # Find and click the logout link
            print("Attempting to logout...")
            self.take_screenshot("before_logout.png")
            
            # Use a more specific XPath to find the logout link
            logout_link = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@href='/logout' and text()='Logout']")),
                message="Logout link not found or not clickable"
            )
            
            print("Logout link found, clicking it...")
            logout_link.click()
            
            # Wait for the page to load after logout
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            # Verify logout success (check for login link)
            try:
                WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//a[@href='/login' and text()='Login']")),
                    message="Login link not found after logout"
                )
                print("Logout successful! Login link found.")
                self.logged_in = False
                
                # Take a screenshot after successful logout
                self.take_screenshot("after_logout.png")
                return True
            except Exception as e:
                print(f"Logout failed - no login link found: {e}")
                self.logged_in = True
                self.take_screenshot("logout_failed.png")
                return False
            
        except Exception as e:
            print(f"Error during logout process: {e}")
            self.take_screenshot("logout_error.png")
            return False 