"""
LLM-Guided Quotes.toscrape.com Navigator

This script demonstrates how to use a local LLM (via Ollama) to guide
Selenium web navigation on quotes.toscrape.com. The LLM analyzes
the page structure and helps make decisions about navigation.
"""

import os
import time
import random
import argparse
import requests
import sys
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# For output formatting
try:
    from colorama import init, Fore, Style
    import pyfiglet
    FORMATTING_AVAILABLE = True
except ImportError:
    FORMATTING_AVAILABLE = False
    print("For enhanced output formatting, install colorama and pyfiglet:")
    print("pip install colorama pyfiglet")

# Add a function to get the best Ollama model
def get_best_ollama_model(server: str = "localhost") -> str:
    """Get the 'best' (largest) Ollama model available locally.
    
    Args:
        server: IP address or hostname of the Ollama server
        
    Returns:
        The name of the largest model available, or a fallback default
    """
    fallback_model = "llama3.2:1b"  # Fallback default
    
    try:
        # Try to get list of models from Ollama API
        response = requests.get(f"http://{server}:11434/api/tags")
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            # No models available
            if not models:
                print(f"No Ollama models found. Using fallback model: {fallback_model}")
                return fallback_model
            
            # Try to find models with parameter information
            models_with_size = []
            for model in models:
                model_name = model.get('name')
                size_mb = model.get('size') / (1024 * 1024) if model.get('size') else 0
                
                # Look for parameter count in name (like 7b, 13b, 70b, etc.)
                param_size = 0
                name_parts = model_name.lower().replace('-', ' ').replace(':', ' ').split()
                for part in name_parts:
                    if part.endswith('b') and part[:-1].isdigit():
                        try:
                            param_size = int(part[:-1])
                            break
                        except ValueError:
                            pass
                
                models_with_size.append((model_name, param_size, size_mb))
            
            # Sort models by parameter size (primary) and file size (secondary)
            models_with_size.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # Return the largest model
            if models_with_size:
                best_model = models_with_size[0][0]
                print(f"Selected largest available model: {best_model}")
                return best_model
            
            # If we couldn't determine sizes, just return the first model
            print(f"Couldn't determine model sizes. Using first available model: {models[0]['name']}")
            return models[0]['name']
            
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        print(f"Using fallback model: {fallback_model}")
    
    # Only try command line if server is localhost
    if server == "localhost":
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the output to find models
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is the model name
                
                if models:
                    # Sort models by potential size indicators in name
                    # This is a heuristic approach - larger models often have numbers like 70b, 13b, 7b
                    def extract_size(model_name):
                        name = model_name.lower()
                        for size in ['70b', '34b', '13b', '7b', '3b', '1b']:
                            if size in name:
                                return int(size[:-1])  # Convert '7b' to 7
                        return 0
                    
                    models.sort(key=extract_size, reverse=True)
                    best_model = models[0]
                    print(f"Selected model with largest parameter count: {best_model}")
                    return best_model
            
        except Exception as e:
            print(f"Error running 'ollama list': {e}")
    
    return fallback_model

# Configuration
DEFAULT_OLLAMA_MODEL = get_best_ollama_model()  # Get the best available model
PROMPT_TEMPLATE = """
You are an AI assistant tasked with guiding a web navigation on the quotes.toscrape.com website.

Current page HTML snippet:
```html
{html_snippet}
```

Current state:
- URL: {current_url}
- Last action: {last_action}
- Logged in: {logged_in}

Based on the HTML snippet above and the current state, please suggest what actions to take next.
Choose from these possible actions:
1. NAVIGATE_NEXT_PAGE - Go to the next page if available
2. NAVIGATE_PREVIOUS_PAGE - Go back to the previous page if available
3. VISIT_AUTHOR_PAGE - Visit a specific author's page (YOU MUST specify a valid author name from the page)
4. FILTER_BY_TAG - Filter quotes by a specific tag (specify which tag)
5. LOGIN - Log in to the website using credentials
6. LOGOUT - Log out of the website
7. COMPLETE_SCRAPING - Complete the navigation process

Important guidelines:
- Only use LOGIN if we're not already logged in and we need to access restricted content
- Only use LOGOUT if we're currently logged in
- Only use VISIT_AUTHOR_PAGE after identifying specific authors from quotes (e.g., "Albert Einstein", "Jane Austen")
- When using VISIT_AUTHOR_PAGE, you MUST provide a real author name from the quotes on the page
- NAVIGATE_NEXT_PAGE is appropriate after exploring the current page
- NAVIGATE_PREVIOUS_PAGE can be used to revisit previous pages
- FILTER_BY_TAG is useful after identifying interesting tags
- COMPLETE_SCRAPING when you have explored enough of the website
{extra_rules}

Most important rule (user-provided):
{user_rule}

Your response should be structured like this:
ACTION: [chosen action]
REASON: [brief explanation of why this action is appropriate]
DETAILS: [any specific details needed for the action, like author name or tag]

Example responses:
ACTION: NAVIGATE_NEXT_PAGE
REASON: I've examined the current page and should check the next page for additional quotes.
DETAILS: None

ACTION: VISIT_AUTHOR_PAGE
REASON: I'd like to see more information about this author.
DETAILS: Albert Einstein

ACTION: FILTER_BY_TAG
REASON: I want to see quotes related to this popular topic.
DETAILS: love
"""

class LLMGuidedQuoteScraper(BaseSeleniumChrome):
    """A Selenium scraper guided by an LLM for intelligent decision making."""
    
    def __init__(self, ollama_model: str = None, extra_rules: List[str] = None, ollama_server: str = "localhost"):
        """Initialize the LLM-guided scraper.
        
        Args:
            ollama_model: The Ollama model to use for guidance (if None, best model will be selected)
            extra_rules: Additional rules to guide the LLM
            ollama_server: IP address or hostname of the Ollama server
        """
        super().__init__()
        self.url = "https://quotes.toscrape.com/"
        self.ollama_server = ollama_server
        self.ollama_api_url = f"http://{ollama_server}:11434/api/generate"
        
        # If no model specified, get the best available model
        if ollama_model is None:
            self.ollama_model = get_best_ollama_model(self.ollama_server)
        else:
            self.ollama_model = ollama_model
            
        self.last_action = "INITIALIZE"  # Track the last action performed
        self.conversation_history = []
        self.extra_rules = extra_rules or []
        self.user_rule = ""  # Initialize user rule as empty
        self.logged_in = False  # Track login status
        
        # Create a directory for storing LLM responses and debug info
        self.debug_dir = f"llm_scraper_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create screenshots directory within our debug directory
        self.custom_screenshots_dir = os.path.join(self.debug_dir, "screenshots")
        os.makedirs(self.custom_screenshots_dir, exist_ok=True)
        
        # Initialize colorama if available
        if FORMATTING_AVAILABLE:
            init()  # Initialize colorama
    
    def display_intro(self):
        """Display an introduction with information about the scraper."""
        if FORMATTING_AVAILABLE:
            # Clear the screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display title with figlet
            title = pyfiglet.figlet_format("Quotes Navigator", font="slant")
            print(Fore.CYAN + title + Style.RESET_ALL)
            
            # Introduce the scraper
            print(Fore.YELLOW + "LLM-Guided Web Navigator for Quotes" + Style.RESET_ALL)
            print(Fore.GREEN + f"Using model: {self.ollama_model}" + Style.RESET_ALL)
            print(Fore.BLUE + f"Server: {self.ollama_server}" + Style.RESET_ALL)
            print(Fore.CYAN + "\nPreparing to navigate..." + Style.RESET_ALL)
            
            print(Fore.MAGENTA + "\n" + "=" * 60 + Style.RESET_ALL)
        else:
            print("\n===== LLM-Guided Web Navigator for Quotes =====")
            print(f"Using model: {self.ollama_model}")
            print(f"Server: {self.ollama_server}")
            print("\nPreparing to navigate...")
            print("=" * 60)
    
    def login(self):
        """Method required by BaseSelenium but not needed for this site."""
        pass
    
    def verify_login_success(self):
        """Method required by BaseSelenium."""
        return True
    
    def get_page_html_snippet(self, max_length: int = 3000) -> str:
        """Get a snippet of the current page's HTML.
        
        Args:
            max_length: Maximum length of the HTML snippet to return
        
        Returns:
            A string containing a portion of the page's HTML
        """
        html = self.driver.page_source
        
        # Get a reasonable snippet that's not too large
        if len(html) > max_length:
            # Try to find main content area
            try:
                main_content = self.driver.find_element(By.CLASS_NAME, "col-md-8").get_attribute("outerHTML")
                return main_content[:max_length]
            except:
                # Fall back to truncating the whole HTML
                return html[:max_length] + "..."
        
        return html
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the scraping process.
        
        Returns:
            A dictionary containing the current state
        """
        return {
            "current_url": self.driver.current_url,
            "last_action": self.last_action,
            "logged_in": self.logged_in
        }
    
    def consult_llm(self) -> Dict[str, str]:
        """Consult the LLM for guidance on the next action.
        
        Returns:
            A dictionary containing the LLM's guidance
        """
        state = self.get_current_state()
        html_snippet = self.get_page_html_snippet()
        
        # Format extra rules if provided
        extra_rules_text = ""
        if self.extra_rules:
            extra_rules_text = "\n" + "\n".join(f"- {rule}" for rule in self.extra_rules)
        
        # Format user rule if provided
        user_rule_text = ""
        if self.user_rule:
            user_rule_text = self.user_rule
        
        prompt = PROMPT_TEMPLATE.format(
            html_snippet=html_snippet,
            current_url=state["current_url"],
            last_action=state["last_action"],
            logged_in=state["logged_in"],
            extra_rules=extra_rules_text,
            user_rule=user_rule_text
        )
        
        # Save prompt for debugging
        with open(os.path.join(self.debug_dir, f"prompt_{len(self.conversation_history)}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Call Ollama API using the instance's API URL
        response = requests.post(
            self.ollama_api_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        llm_response = response.json()["response"]
        
        # Save response for debugging
        with open(os.path.join(self.debug_dir, f"response_{len(self.conversation_history)}.txt"), "w", encoding="utf-8") as f:
            f.write(llm_response)
        
        # Parse the LLM response
        parsed_response = self.parse_llm_response(llm_response)
        
        # Add to conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": llm_response,
            "parsed_response": parsed_response
        })
        
        return parsed_response
    
    def parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM's response into structured data.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A dictionary containing the parsed response
        """
        # Default values
        parsed = {
            "action": "NAVIGATE_NEXT_PAGE",  # Safe default
            "reason": "Default action",
            "details": "None"
        }
        
        # Extract sections from the response
        for line in response.split('\n'):
            if line.startswith("ACTION:"):
                parsed["action"] = line.replace("ACTION:", "").strip()
            elif line.startswith("REASON:"):
                parsed["reason"] = line.replace("REASON:", "").strip()
            elif line.startswith("DETAILS:"):
                parsed["details"] = line.replace("DETAILS:", "").strip()
        
        return parsed
    
    def execute_action(self, action_data: Dict[str, str]) -> bool:
        """Execute the action suggested by the LLM.
        
        Args:
            action_data: The parsed LLM response containing action details
            
        Returns:
            Boolean indicating success or failure
        """
        action = action_data["action"]
        
        if FORMATTING_AVAILABLE:
            print(Fore.CYAN + f"\n=== Executing Action: {action} ===" + Style.RESET_ALL)
            print(Fore.GREEN + f"Reason: {action_data['reason']}" + Style.RESET_ALL)
        else:
            print(f"\n=== Executing Action: {action} ===")
            print(f"Reason: {action_data['reason']}")
            
        if action == "NAVIGATE_NEXT_PAGE":
            self.last_action = "NAVIGATE_NEXT_PAGE"
            return self.navigate_to_next_page()
            
        elif action == "NAVIGATE_PREVIOUS_PAGE":
            self.last_action = "NAVIGATE_PREVIOUS_PAGE"
            return self.navigate_to_previous_page()
        
        elif action == "VISIT_AUTHOR_PAGE":
            author = action_data["details"]
            if author.lower() == "none" or not author:
                print("No specific author specified. Cannot visit author page.")
                
                # Instead of failing completely, suggest a fallback action
                print("Falling back to NAVIGATE_NEXT_PAGE action.")
                self.last_action = "NAVIGATE_NEXT_PAGE"
                return self.navigate_to_next_page()
            
            self.last_action = "VISIT_AUTHOR_PAGE"
            return self.visit_author_page(author)
        
        elif action == "FILTER_BY_TAG":
            tag = action_data["details"]
            if tag.lower() == "none" or not tag:
                print("No specific tag specified. Cannot filter by tag.")
                
                # Instead of failing completely, suggest a fallback action
                print("Falling back to NAVIGATE_NEXT_PAGE action.")
                self.last_action = "NAVIGATE_NEXT_PAGE"
                return self.navigate_to_next_page()
            
            self.last_action = "FILTER_BY_TAG"
            return self.filter_by_tag(tag)
        
        elif action == "LOGIN":
            if self.logged_in:
                print("Already logged in.")
                return True
            
            self.last_action = "LOGIN"
            return self.login_to_site()
        
        elif action == "LOGOUT":
            if not self.logged_in:
                print("Not currently logged in.")
                return False
                
            self.last_action = "LOGOUT"
            return self.logout_from_site()
        
        elif action == "COMPLETE_SCRAPING":
            self.last_action = "COMPLETE_SCRAPING"
            print("Navigation process complete as suggested by LLM.")
            return False  # Signal to stop the scraping loop
        
        else:
            print(f"Unknown action: {action}")
            # Fallback to a safe action
            print("Falling back to NAVIGATE_NEXT_PAGE action.")
            self.last_action = "NAVIGATE_NEXT_PAGE"
            return self.navigate_to_next_page()
    
    def custom_screenshot(self, filename: str) -> None:
        """Take a screenshot and save it with a custom filename.
        
        Args:
            filename: The name of the screenshot file to save
        """
        try:
            # Take a screenshot directly and save it to our custom directory
            screenshot_path = os.path.join(self.custom_screenshots_dir, filename)
            self.driver.save_screenshot(screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")
        except Exception as e:
            print(f"Error saving custom screenshot: {e}")
    
    def navigate_to_next_page(self) -> bool:
        """Navigate to the next page of quotes.
        
        Returns:
            Boolean indicating success
        """
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, "li.next > a")
            next_button.click()
            
            # Wait for the new page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Navigated to next page")
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
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Navigated back to previous page")
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
            
            # Find the author link if we're on the quotes page
            try:
                # Try to find by the author name in the page
                author_links = self.driver.find_elements(By.XPATH, 
                    f"//small[@class='author' and text()='{author_name}']/following-sibling::a")
                
                if author_links:
                    author_links[0].click()
                else:
                    print(f"Couldn't find link for author: {author_name}")
                    return False
            
            except NoSuchElementException:
                print(f"Couldn't find author link for: {author_name}")
                return False
            
            # Wait for author details to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "author-details"))
            )
            
            print(f"Visited author page for: {author_name}")
            
            # Take a screenshot
            self.custom_screenshot(f"author_{author_name.replace(' ', '_')}.png")
            
            # Return to the original page
            self.driver.get(current_url)
            
            # Wait for the original page to load
            WebDriverWait(self.driver, 10).until(
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
            tag_links = self.driver.find_elements(By.XPATH, f"//a[@class='tag' and text()='{tag}']")
            
            if not tag_links:
                # Try to navigate to the tags page
                try:
                    # Go to the home page first
                    self.driver.get(self.url)
                    
                    # Wait for the page to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "quote"))
                    )
                    
                    # Now try again to find the tag
                    tag_links = self.driver.find_elements(By.XPATH, f"//a[@class='tag' and text()='{tag}']")
                    
                    if not tag_links:
                        print(f"Tag '{tag}' not found on the current page.")
                        return False
                    
                except Exception as e:
                    print(f"Error navigating to find tag: {e}")
                    return False
            
            # Click the tag link
            tag_links[0].click()
            
            # Wait for the filtered page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            print(f"Filtered quotes by tag: {tag}")
            return True
            
        except Exception as e:
            print(f"Error filtering by tag: {e}")
            return False
    
    def login_to_site(self) -> bool:
        """Log in to the quotes.toscrape.com website.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Navigate to the login page
            login_url = "https://quotes.toscrape.com/login"
            self.driver.get(login_url)
            
            # Wait for the login form to be available
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form.form-signin"))
            )
            
            # Find username and password fields and submit button
            username_field = self.driver.find_element(By.ID, "username")
            password_field = self.driver.find_element(By.ID, "password")
            submit_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            
            # Enter credentials
            username = "abcd"  # Hardcoded as specified
            password = "123456"  # Hardcoded as specified
            
            username_field.send_keys(username)
            password_field.send_keys(password)
            
            # Take a screenshot before submitting
            self.custom_screenshot("before_login.png")
            
            # Submit the form
            submit_button.click()
            
            # Wait for page to load after login
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            # Verify login success (check for logout link)
            try:
                logout_link = self.driver.find_element(By.XPATH, "//a[text()='Logout']")
                if logout_link:
                    print("Login successful!")
                    self.logged_in = True
                    
                    # Take a screenshot after successful login
                    self.custom_screenshot("after_login.png")
                    return True
            except NoSuchElementException:
                print("Login failed - no logout link found.")
                self.logged_in = False
                return False
            
        except Exception as e:
            print(f"Error during login: {e}")
            return False
    
    def logout_from_site(self) -> bool:
        """Log out from the quotes.toscrape.com website.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Find and click the logout link
            logout_link = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[text()='Logout']"))
            )
            
            # Take a screenshot before logging out
            self.custom_screenshot("before_logout.png")
            
            # Click the logout link
            logout_link.click()
            
            # Wait for the page to load after logout
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            # Verify logout success (check for login link)
            try:
                login_link = self.driver.find_element(By.XPATH, "//a[text()='Login']")
                if login_link:
                    print("Logout successful!")
                    self.logged_in = False
                    
                    # Take a screenshot after successful logout
                    self.custom_screenshot("after_logout.png")
                    return True
            except NoSuchElementException:
                print("Logout failed - no login link found.")
                self.logged_in = True
                return False
            
        except Exception as e:
            print(f"Error during logout: {e}")
            return False
    
    def get_user_rule(self) -> str:
        """Get a new rule from the user.
        
        Returns:
            The user's input rule
        """
        if FORMATTING_AVAILABLE:
            print(Fore.MAGENTA + "\n=== Waiting for User Input ===" + Style.RESET_ALL)
            print(Fore.YELLOW + "Enter your new rule:" + Style.RESET_ALL)
        else:
            print("\n=== Waiting for User Input ===")
            print("Enter your new rule:")
        
        user_input = input("> ")
        
        # Always pass user input directly to the LLM
        return user_input
    
    def run_llm_guided_navigation(self, max_actions: int = 10) -> None:
        """Run the LLM-guided navigation process.
        
        Args:
            max_actions: Maximum number of actions to take
        """
        if FORMATTING_AVAILABLE:
            print(Fore.CYAN + "\n" + "=" * 60)
            title = pyfiglet.figlet_format("Starting Navigator", font="small")
            print(title)
            print(Fore.YELLOW + f"Beginning navigation process using model: {self.ollama_model}" + Style.RESET_ALL)
            print("=" * 60 + "\n" + Style.RESET_ALL)
        else:
            print(f"Starting LLM-guided website navigation using model: {self.ollama_model}")
        
        self.prepare_environment()
        self.navigate_to_url()
        
        # Take initial screenshot
        self.custom_screenshot("initial_page.png")
        
        # Add initial rules for the first action
        self.user_rule = "For the first action, use NAVIGATE_NEXT_PAGE to explore quotes or FILTER_BY_TAG with a specific tag you can see. Don't use VISIT_AUTHOR_PAGE until you've seen the quotes on the page."
        
        action_count = 0
        while action_count < max_actions:
            if FORMATTING_AVAILABLE:
                progress = int((action_count / max_actions) * 20)
                progress_bar = "#" * progress + "-" * (20 - progress)
                print(Fore.MAGENTA + f"\n=== Progress: [{progress_bar}] [{action_count + 1}/{max_actions}] ===" + Style.RESET_ALL)
            else:
                print(f"\n=== Action {action_count + 1}/{max_actions} ===")
            
            # After every two actions, get new rule from user
            if action_count > 0 and action_count % 2 == 0:
                self.user_rule = self.get_user_rule()
                if FORMATTING_AVAILABLE:
                    print(Fore.CYAN + f"New user rule: {self.user_rule}" + Style.RESET_ALL)
                else:
                    print(f"New user rule: {self.user_rule}")
            
            # Get guidance from LLM
            print("Consulting LLM for next action...")
            guidance = self.consult_llm()
            
            # Execute the suggested action
            if FORMATTING_AVAILABLE:
                print(Fore.YELLOW + f"LLM suggests: {guidance['action']}" + Style.RESET_ALL)
            else:
                print(f"LLM suggests: {guidance['action']}")
                
            success = self.execute_action(guidance)
            
            # Add a small delay to be respectful to the server
            delay = random.uniform(1, 2)
            time.sleep(delay)
            
            action_count += 1
            
            # Check if we should stop
            if not success or guidance["action"] == "COMPLETE_SCRAPING":
                print("Stopping navigation process as recommended by LLM or due to action failure.")
                break
        
        # Navigation complete
        if FORMATTING_AVAILABLE:
            print(Fore.GREEN + "\n" + "=" * 60)
            title = pyfiglet.figlet_format("Navigation Complete", font="small")
            print(title)
            print(Fore.YELLOW + f"Executed {action_count} actions" + Style.RESET_ALL)
            print("=" * 60 + "\n" + Style.RESET_ALL)
        else:
            print("\n=== Navigation Complete ===")
            print(f"Executed {action_count} actions")
        
        # Clean up
        self.terminate_webdriver()


def main():
    """Run the LLM-guided quotes navigator."""
    parser = argparse.ArgumentParser(description='LLM-guided quotes.toscrape.com navigator')
    parser.add_argument('rules', nargs='*', 
                        help='Additional rules to guide the LLM (optional, can provide multiple)')
    parser.add_argument('--model', type=str, default=None,
                        help='Ollama model to use (if not specified, the largest available model will be used)')
    parser.add_argument('--max-actions', type=int, default=50,
                        help='Maximum number of actions to take (default: 50)')
    parser.add_argument('--server', type=str, default='localhost',
                        help='IP address or hostname of the Ollama server (default: localhost)')
    args = parser.parse_args()
    
    scraper = LLMGuidedQuoteScraper(
        ollama_model=args.model, 
        extra_rules=args.rules,
        ollama_server=args.server
    )
    
    try:
        # Display intro before starting
        scraper.display_intro()
        scraper.run_llm_guided_navigation(max_actions=args.max_actions)
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user.")
        scraper.terminate_webdriver()
    except Exception as e:
        print(f"\nError during navigation: {e}")
        # Clean up
        try:
            scraper.terminate_webdriver()
        except:
            pass


if __name__ == "__main__":
    main()
