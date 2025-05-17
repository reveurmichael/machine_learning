"""
LLM-guided code generator for Selenium automation.
This script uses LLMs to generate and execute Selenium code for web automation.
"""

import os
import re
import time
import yaml
import argparse
import shutil
import sys
from io import StringIO
from datetime import datetime
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from colorama import Fore, Style, init as init_colorama

# Import the LLM client and SeleniumDriver
from llm_client import LLMClient
from selenium_driver import SeleniumDriver

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

# Load environment variables
load_dotenv()

# Prompt templates
CODE_GENERATION_PROMPT = """
You are an expert Python programmer specialized in web automation with Selenium and the llm_selenium_agent package.
I need you to generate executable Python code for the following task:

Task: {task}

The code should interact with the website: {website}

Requirements:
- You MUST use the existing driver instance provided as variable 'driver' - this is an instance of llm_selenium_agent.SeleniumDriver
- When taking screenshots, use driver.take_screenshot("filename") rather than direct Selenium methods
- You should leverage the llm_selenium_agent package for common operations (navigation, screenshots, etc.)
- Don't include imports for selenium modules as they're already imported
- Use explicit waits for reliability
- Include proper exception handling
- Add comments to explain key sections
- Structure the code using functions

Important: The driver is ALREADY INITIALIZED. Do not initialize a new driver, just use the existing 'driver' variable.

Please provide ONLY the Python code, no explanations before or after.

Please write a lot of print statements after some key code lines to help me debug the code.
"""

ACTION_SUGGESTION_PROMPT = """
You are an AI assistant tasked with guiding web scraping on quotes.toscrape.com.

Current page HTML snippet:
```html
{html_snippet}
```

Current URL: {current_url}

Based on the HTML snippet and current state, suggest the next action to take.
Choose from these possible actions:
1. NAVIGATE_NEXT_PAGE - Go to the next page of quotes
2. NAVIGATE_PREVIOUS_PAGE - Go back to the previous page
3. FILTER_BY_TAG - Filter quotes by a specific tag
4. VISIT_AUTHOR_PAGE - Visit an author's page to see their details
5. LOGIN - Log in to the website (username: user, password: pass)
6. LOGOUT - Log out from the website
7. SCROLL - Scroll down the page
8. EXTRACT_DATA - Extract and print data from the current page

Your response should be structured like this:
ACTION: [chosen action]
REASON: [brief explanation of why this action is appropriate]
DETAILS: [any specific details needed for the action, like which tag to filter or which author to visit]
CODE: [the specific Python code using Selenium to implement this action]

IMPORTANT: 
- You MUST use the existing driver instance
- When handling elements, make sure they are visible and clickable before interacting
- Use explicit waits for reliability
- Add proper error handling with try/except blocks
- Include scrolling if elements might be outside the viewport
- For clicking elements, use JavaScript execution as backup if direct clicking fails
- For the "next page" button, use a robust selector like "//a[contains(text(), 'Next')]" or "//li[@class='next']/a"

Example response:
ACTION: FILTER_BY_TAG
REASON: I can see several interesting tags on the page, and filtering by 'love' would show us quotes related to this theme.
DETAILS: love
CODE:
```python
def filter_by_tag(driver, tag_name="love"):
    try:
        # Wait for tags to be visible
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tag"))
        )
        
        # Find all tags
        tag_links = driver.find_elements(By.CLASS_NAME, "tag")
        target_link = None
        
        # Look for our specific tag
        for link in tag_links:
            if link.text.strip().lower() == tag_name.lower():
                target_link = link
                break
        
        if not target_link:
            print(f"Tag '{tag_name}' not found on the page.")
            return False
        
        # Scroll the tag into view
        driver.execute_script("arguments[0].scrollIntoView(true);", target_link)
        time.sleep(0.5)  # Small pause after scrolling
        
        # Click the tag, using JavaScript as a fallback
        try:
            target_link.click()
        except Exception as e:
            print(f"Direct click failed, trying JavaScript click: {e}")
            driver.execute_script("arguments[0].click();", target_link)
        
        # Wait for the page to load with filtered results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "quote"))
        )
        
        print(f"Successfully filtered quotes by tag: {tag_name}")
        return True
        
    except Exception as e:
        print(f"Error filtering by tag: {e}")
        return False

# Execute the function with the desired tag
filter_by_tag(driver)
```
"""

class SeleniumCodeGenerator:
    """Generates and executes Selenium code using LLMs."""
    
    def __init__(self, llm_provider="hunyuan", website="https://quotes.toscrape.com", headless=False):
        """Initialize the code generator.
        
        Args:
            llm_provider: The LLM provider to use ("hunyuan" or "ollama")
            website: The website to interact with
            headless: Whether to run the browser in headless mode
        """
        self.llm_client = LLMClient(provider=llm_provider)
        self.website = website
        self.headless = headless
        self.driver = None
        
        # Create debug directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.debug_dir = f"code_generation_{timestamp}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create directories for saving prompts, responses, code
        self.prompts_dir = os.path.join(self.debug_dir, "prompts")
        self.responses_dir = os.path.join(self.debug_dir, "responses")
        self.code_dir = os.path.join(self.debug_dir, "code")
        
        for directory in [self.prompts_dir, self.responses_dir, self.code_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Copy config.yml to the debug directory for llm_selenium_agent to use
        self._copy_config_file()
        
        # Initialize the webdriver using our SeleniumDriver
        self._setup_webdriver()
    
    def _copy_config_file(self):
        """Copy the config.yml file to the generated code directory."""
        try:
            # Source config file (from current directory or parent directories)
            source_config = "config.yml"
            if not os.path.exists(source_config):
                # Try looking in parent directory
                source_config = os.path.join("..", "config.yml")
            
            if os.path.exists(source_config):
                # Copy to the generated code directory
                dest_config = os.path.join(self.debug_dir + "/code", "config.yml")
                shutil.copy2(source_config, dest_config)
                print(Fore.GREEN + f"✅ Copied config.yml to {self.debug_dir}")
            else:
                print(Fore.YELLOW + f"⚠️ Could not find config.yml to copy")
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Error copying config.yml: {e}")
    
    def _setup_webdriver(self):
        """Set up the Selenium WebDriver using our custom driver."""
        try:
            # Create SeleniumDriver instance
            self.driver = SeleniumDriver(headless=self.headless, website=self.website)
            
            # Prepare environment (required by BaseSeleniumChrome)
            self.driver.prepare_environment()
            
            # Also use the driver's screenshot directory
            self.screenshots_dir = self.driver.screenshots_dir
            
            print(Fore.GREEN + "✅ WebDriver initialized successfully")
            
        except Exception as e:
            print(Fore.RED + f"❌ Error initializing WebDriver: {e}")
            raise
    
    def _save_to_file(self, content, directory, filename):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def get_page_html(self, max_length=30000):
        return self.driver.get_page_html(max_length)
    
    def execute_code(self, code, description="Generated code"):
        """Execute the generated code."""
        try:
            # Clean the code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code.split("```python", 1)[1]
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            
            code = code.strip()
            
            # Save the code to a file
            timestamp = datetime.now().strftime('%H%M%S')
            code_filename = f"code_{timestamp}.py"
            code_path = self._save_to_file(code, self.code_dir, code_filename)
            print(Fore.BLUE + f"💾 Code saved to {code_path}")
            
            # Take a screenshot before executing the code
            self.driver.take_screenshot(f"before_{timestamp}")
            
            # Import Selenium classes needed for the wrapper
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.common.action_chains import ActionChains
            
            # Define the driver wrapper class directly in this scope
            class DriverWrapper:
                def __init__(self, selenium_driver):
                    self.selenium_driver = selenium_driver
                    # Access the underlying WebDriver directly for properties
                    self.driver = selenium_driver.driver
                    # Add wait property that many scripts use
                    self.wait = WebDriverWait(self.driver, 10)
                
                @property
                def current_url(self):
                    return self.driver.current_url
                
                def take_screenshot(self, filename):
                    return self.selenium_driver.take_screenshot(filename)
                
                def find_element(self, by, value):
                    print(f"Finding element: {by}={value}")
                    elem = self.driver.find_element(by, value)
                    # Highlight the element briefly
                    self.highlight_element(elem)
                    return elem
                
                def find_elements(self, by, value):
                    print(f"Finding elements: {by}={value}")
                    return self.driver.find_elements(by, value)
                
                def get(self, url):
                    print(f"Navigating to URL: {url}")
                    return self.driver.get(url)
                
                def execute_script(self, script, *args):
                    return self.driver.execute_script(script, *args)
                
                def highlight_element(self, element, duration=0.5):
                    """Highlight an element by changing its border and background color briefly."""
                    try:
                        original_style = element.get_attribute("style")
                        self.driver.execute_script(
                            "arguments[0].setAttribute('style', arguments[1]);", 
                            element, 
                            "border: 2px solid red; background-color: yellow;"
                        )
                        time.sleep(duration)
                        self.driver.execute_script(
                            "arguments[0].setAttribute('style', arguments[1]);", 
                            element, 
                            original_style
                        )
                    except:
                        pass  # Ignore any errors during highlighting
                
                def click(self, element):
                    """Click an element with visual feedback and error handling."""
                    try:
                        print(f"Clicking element: {element.tag_name}")
                        self.highlight_element(element)
                        # Scroll into view first
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(0.3)  # Brief pause after scrolling
                        # Try direct click
                        element.click()
                        print("Element clicked successfully")
                    except Exception as e:
                        print(f"Direct click failed, trying JavaScript click: {e}")
                        # Fall back to JavaScript click
                        self.driver.execute_script("arguments[0].click();", element)
                        print("Element clicked with JavaScript")
                    time.sleep(0.5)  # Small pause after clicking
            
            # Create a wrapper instance
            driver_wrapper = DriverWrapper(self.driver)
            
            # Print execution start
            print(Fore.CYAN + f"🚀 Executing code for: {description}")
            
            # Add direct function calls if needed (some LLMs define but forget to call functions)
            function_pattern = r'def\s+(\w+)\s*\('
            function_calls = re.findall(function_pattern, code)
            
            # If there are function definitions but no calls to them at the end, add calls
            if function_calls:
                main_function = None
                if 'main' in function_calls:
                    main_function = 'main'
                elif function_calls:
                    main_function = function_calls[0]
                
                # Check if the function is already called in the code
                if main_function and f"{main_function}(" not in code.split("def ")[-1]:
                    print(f"Adding explicit call to {main_function}() function")
                    code += f"\n\n# Explicitly added function call\nprint('Calling {main_function}(driver)...')\n{main_function}(driver)"
            
            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            captured_output = StringIO()
            sys.stdout = captured_output
            sys.stderr = captured_output
            
            try:
                # Set up the execution environment
                exec_globals = {
                    'driver': driver_wrapper,
                    'By': By,
                    'WebDriverWait': WebDriverWait,
                    'EC': EC,
                    'Keys': Keys,
                    'ActionChains': ActionChains,
                    'time': time
                }
                
                # Execute the code
                exec(code, exec_globals)
                
                # Look for defined functions that we can call if they weren't called already
                known_functions = [
                    'go_to_next_page', 'navigate_to_next_page', 'click_next_page',
                    'go_to_previous_page', 'navigate_previous_page', 
                    'login', 'logout', 'filter_by_tag', 'visit_author', 'main'
                ]
                
                function_called = False
                
                # Try to call each known function
                for func_name in known_functions:
                    if func_name in exec_globals and callable(exec_globals[func_name]):
                        try:
                            print(f"Calling {func_name}(driver)...")
                            exec_globals[func_name](driver_wrapper)
                            function_called = True
                            break  # Stop after successfully calling one function
                        except Exception as e:
                            print(f"Error calling {func_name}: {e}")
                
                if not function_called:
                    print("No functions were called - code may have executed directly")
                
                # Get the captured output
                output = captured_output.getvalue()
                
                # Print the captured output with proper formatting
                if output.strip():
                    print(Fore.GREEN + "📝 Output from executed code:")
                    print(Fore.WHITE + output.strip())
                else:
                    print(Fore.YELLOW + "⚠️ No output captured from executed code")
                
            finally:
                # Restore stdout and stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # Allow some time for the page to update
            time.sleep(2)
            
            # Take a screenshot after executing the code
            self.driver.take_screenshot(f"after_{timestamp}")
            
            print(Fore.GREEN + "✅ Code executed successfully")
            return True
            
        except Exception as e:
            print(Fore.RED + f"❌ Error executing code: {e}")
            import traceback
            print(Fore.RED + traceback.format_exc())
            # Take a screenshot of the error state
            self.driver.take_screenshot(f"error_{datetime.now().strftime('%H%M%S')}")
            return False
    
    def generate_initial_code(self, task):
        """Generate initial code to navigate to the website and start the task.
        
        Args:
            task: The automation task description
            
        Returns:
            The generated code as a string
        """
        # Get the current page HTML to help with element selection
        html_snippet = self.get_page_html(max_length=10000)
        current_url = self.driver.driver.current_url
        
        # Create a more informative prompt with HTML context
        prompt = f"""
You are an expert Python programmer specialized in web automation with Selenium and the llm_selenium_agent package.
I need you to generate executable Python code for the following task:

Task: {task}

Current URL: {current_url}

Current page HTML snippet:
```html
{html_snippet}
```

The code should interact with the website: {self.website}

Requirements:
- You MUST use the existing driver instance provided as variable 'driver'
- When taking screenshots, use driver.take_screenshot("filename") rather than direct Selenium methods
- Use explicit waits for reliability
- Include proper exception handling
- Add comments to explain key sections
- Structure the code using functions
- IMPORTANT: Make sure to actually call your functions! Don't just define them.

IMPORTANT NOTES FOR ROBUST SELENIUM INTERACTIONS:
- When handling elements, make sure they are visible and clickable before interacting
- Always scroll elements into view before clicking
- For problematic clicks, use JavaScript execution as a backup
- For the "next page" button, use a robust selector like "//li[@class='next']/a"
- Add print statements to show progress for debugging
- Add explicit pauses after major actions (time.sleep(1)) for visibility
- Screenshots should be taken after each significant action

Please provide ONLY the Python code, no explanations before or after.
"""
        
        # Save the prompt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_to_file(prompt, self.prompts_dir, f"initial_prompt_{timestamp}.txt")
        
        # Generate the response
        print(Fore.CYAN + "🤖 Generating initial code...")
        response = self.llm_client.generate_response(prompt, temperature=0.3)
        
        # Save the response
        self._save_to_file(response, self.responses_dir, f"initial_response_{timestamp}.txt")
        
        return response
    
    def get_next_action(self):
        """Get the next action to take based on the current page.
        
        Returns:
            The action response from the LLM
        """
        # Get the current page info
        html_snippet = self.get_page_html()
        current_url = self.driver.driver.current_url
        
        # Create the prompt
        prompt = ACTION_SUGGESTION_PROMPT.format(
            html_snippet=html_snippet,
            current_url=current_url
        )
        
        # Save the prompt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_to_file(prompt, self.prompts_dir, f"action_prompt_{timestamp}.txt")
        
        # Generate the response
        print(Fore.CYAN + "🤖 Suggesting next action...")
        response = self.llm_client.generate_response(prompt)
        
        # Save the response
        self._save_to_file(response, self.responses_dir, f"action_response_{timestamp}.txt")
        
        return response
    
    def _parse_action_response(self, response):
        """Parse the action response from the LLM.
        
        Args:
            response: The LLM's response string
            
        Returns:
            A dictionary with action, reason, details, and code
        """
        result = {
            "action": None,
            "reason": None,
            "details": None,
            "code": None
        }
        
        # Extract action
        action_match = re.search(r"ACTION:\s*(.*?)(?:\n|$)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Extract reason
        reason_match = re.search(r"REASON:\s*(.*?)(?:\n|$)", response)
        if reason_match:
            result["reason"] = reason_match.group(1).strip()
        
        # Extract details
        details_match = re.search(r"DETAILS:\s*(.*?)(?:\n|$)", response)
        if details_match:
            result["details"] = details_match.group(1).strip()
        
        # Extract code
        code_match = re.search(r"CODE:\s*(?:```python)?(.*?)(?:```)?$", response, re.DOTALL)
        if code_match:
            result["code"] = code_match.group(1).strip()
        
        return result
    
    def run_interactive_session(self, initial_task=None, max_actions=100):
        """Run an interactive session with the user.
        
        Args:
            initial_task: An optional initial task to start with
            max_actions: Maximum number of actions to take before stopping
        """
        try:
            # Navigate to the website using the driver's method
            print(Fore.CYAN + f"🌐 Navigating to {self.website}")
            self.driver.navigate_to_url()
            time.sleep(2)
            self.driver.take_screenshot("initial_page")
            
            # If there's an initial task, generate and execute the code
            if initial_task:
                initial_code = self.generate_initial_code(initial_task)
                self.execute_code(initial_code, description=initial_task)
            
            # Interactive session loop
            action_count = 0
            while action_count < max_actions:
                # Take a screenshot of the current state
                self.driver.take_screenshot(f"state_{action_count}")
                
                # Get user input for the next action
                print(Fore.MAGENTA + "\n" + "="*80)
                print(Fore.MAGENTA + "Current URL: " + self.driver.driver.current_url)
                print(Fore.MAGENTA + "="*80)
                print(Fore.GREEN + "What would you like to do next? (Type 'exit' to quit)")
                user_input = input(Fore.GREEN + "> ")
                
                # Check if the user wants to exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(Fore.CYAN + "👋 Exiting interactive session")
                    break
                
                # Generate a suggestion based on the current page or use user input
                if user_input.strip():
                    # If the user provided input, use it to guide the LLM
                    action_code = self.generate_initial_code(user_input)
                    self.execute_code(action_code, description=user_input)
                else:
                    # Get a suggestion from the LLM
                    response = self.get_next_action()
                    parsed = self._parse_action_response(response)
                    
                    # Print the suggestion
                    print(Fore.CYAN + f"Suggested action: {parsed['action']}")
                    print(Fore.CYAN + f"Reason: {parsed['reason']}")
                    if parsed['details'] and parsed['details'].lower() != 'none':
                        print(Fore.CYAN + f"Details: {parsed['details']}")
                    
                    # Ask if the user wants to execute the suggestion
                    print(Fore.GREEN + "Execute this suggestion? (y/n)")
                    execute = input(Fore.GREEN + "> ").lower()
                    
                    if execute in ['y', 'yes']:
                        # Use the code generated by the LLM
                        self.execute_code(parsed['code'], description=parsed['action'])
                
                action_count += 1
                
        except Exception as e:
            print(Fore.RED + f"❌ Error in interactive session: {e}")
        finally:
            # Always take a final screenshot
            self.driver.take_screenshot("final_state")
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.cleanup()

def main():
    """Main function to run the Selenium code generator."""
    parser = argparse.ArgumentParser(description='LLM-guided Selenium code generator')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use (hunyuan or ollama)')
    parser.add_argument('--website', type=str, default='https://quotes.toscrape.com',
                      help='Website to interact with')
    parser.add_argument('--task', type=str,
                      help='Initial task to perform')
    parser.add_argument('--max-actions', type=int, default=100,
                      help='Maximum number of actions to take')
    parser.add_argument('--headless', action='store_true',
                      help='Run the browser in headless mode')
    
    args = parser.parse_args()
    
    try:
        # Create the code generator
        generator = SeleniumCodeGenerator(
            llm_provider=args.provider,
            website=args.website,
            headless=args.headless
        )
        
        # Run the interactive session
        generator.run_interactive_session(
            initial_task=args.task,
            max_actions=args.max_actions
        )
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n⚠️ Session interrupted by user")
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}")
    finally:
        # Clean up
        if 'generator' in locals():
            generator.cleanup()

if __name__ == "__main__":
    main() 