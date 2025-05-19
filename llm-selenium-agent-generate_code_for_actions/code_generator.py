"""
LLM-guided code generator for Selenium automation.
This script uses LLMs to generate and execute Selenium code for web automation.
"""

import os
import re
import time
import argparse
import shutil
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from colorama import Fore, Style, init as init_colorama
import sys
from io import StringIO

# Import the LLM client and SeleniumDriver
from llm_client import LLMClient
from selenium_driver import SeleniumDriver

# Import configuration
from config import ACTION_SUGGESTION_PROMPT, DEFAULT_TEMPERATURE, SELENIUM_TIMEOUT

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

# Load environment variables
load_dotenv()

class SeleniumCodeGenerator:
    """Generates and executes Selenium code using LLMs.
    
    This class provides the core functionality for:
    1. Generating Selenium code using LLMs based on the current page
    2. Executing the generated code in a real browser
    3. Providing visual feedback during execution
    """
    
    def __init__(self, llm_provider="hunyuan", website="https://quotes.toscrape.com", headless=False, model=None):
        """Initialize the code generator.
        
        Args:
            llm_provider: The LLM provider to use ("hunyuan" or "ollama")
            website: The website to interact with
            headless: Whether to run the browser in headless mode
            model: Specific model to use with the provider
        """
        # Initialize the LLM client with the specified provider and model
        self.llm_client = LLMClient(provider=llm_provider, model=model)
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
        
        
        # Initialize the webdriver using our SeleniumDriver
        self._setup_webdriver()
    
    
    
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
        """Save content to a file in the specified directory.
        
        Args:
            content: Content to save
            directory: Directory to save to
            filename: Name of the file
        """
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def get_page_html(self, max_length=30000):
        """Get the HTML of the current page.
        
        Args:
            max_length: Maximum length of the HTML to return
            
        Returns:
            A string containing the HTML
        """
        return self.driver.get_page_html(max_length)
    
    def _extract_code_from_response(self, response):
        """Extract code from an LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            The extracted code or the original response if no code section found
        """
        if "GENERATED_CODE:" in response:
            # Extract everything after the GENERATED_CODE: marker
            code_match = re.search(r"GENERATED_CODE:\s*(.*?)(?:\n\s*(?:###|Note|Notes)\s*:.*)?$", response, re.DOTALL | re.IGNORECASE)
            if code_match:
                code = code_match.group(1).strip()
                
                # Remove any trailing explanatory notes sections
                code = re.sub(r'\n\s*(?:###|Note|Notes)\s*:.*$', '', code, flags=re.DOTALL | re.IGNORECASE)
                
                return code
        return response
    
    def execute_code(self, code, description="Generated code"):
        """Execute the generated code in the browser.
        
        Args:
            code: The Python code to execute
            description: Description of what the code does
            
        Returns:
            Boolean indicating success
        """
        try:
            # Remove any explanatory notes sections
            code = re.sub(r'\n\s*(?:###|Note|Notes)\s*:.*$', '', code, flags=re.DOTALL | re.IGNORECASE)
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
                """Wrapper around the SeleniumDriver to provide enhanced functionality.
                
                This wrapper:
                1. Makes common Selenium operations more visible
                2. Adds visual highlighting when elements are interacted with
                3. Provides robust click operations with fallbacks
                """
                def __init__(self, selenium_driver):
                    self.selenium_driver = selenium_driver
                    # Access the underlying WebDriver directly for properties
                    self.driver = selenium_driver.driver
                    # Add wait property that many scripts use
                    self.wait = WebDriverWait(self.driver, SELENIUM_TIMEOUT)
                
                @property
                def current_url(self):
                    """Get the current URL of the page."""
                    return self.driver.current_url
                
                def take_screenshot(self, filename):
                    """Take a screenshot with a descriptive filename."""
                    print(f"📸 Taking screenshot: {filename}")
                    return self.selenium_driver.take_screenshot(filename)
                
                def find_element(self, by, value):
                    """Find an element and highlight it for visibility."""
                    print(f"🔍 Finding element: {by}={value}")
                    try:
                        # Use explicit wait to find the element
                        element = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                            EC.presence_of_element_located((by, value))
                        )
                        print(f"✅ Element found")
                        
                        # Highlight the element briefly
                        self.highlight_element(element)
                        return element
                    except Exception as e:
                        print(f"❌ Error finding element: {e}")
                        # Re-raise the exception
                        raise
                
                def find_elements(self, by, value):
                    """Find multiple elements matching the selector."""
                    print(f"🔍 Finding elements: {by}={value}")
                    elements = self.driver.find_elements(by, value)
                    print(f"Found {len(elements)} elements")
                    return elements
                
                def get(self, url):
                    """Navigate to a URL."""
                    print(f"🌐 Navigating to URL: {url}")
                    return self.driver.get(url)
                
                def execute_script(self, script, *args):
                    """Execute JavaScript in the browser."""
                    return self.driver.execute_script(script, *args)
                
                def highlight_element(self, element, duration=0.5):
                    """Highlight an element by changing its border and background color briefly."""
                    try:
                        # Check if the element is displayed before highlighting
                        try:
                            if not element.is_displayed():
                                print("⚠️ Element is not displayed, skipping highlight")
                                return
                        except:
                            # If we can't check is_displayed, continue anyway
                            pass
                            
                        # Get original style
                        try:
                            original_style = element.get_attribute("style")
                        except:
                            original_style = ""
                            
                        # Apply highlight style
                        try:
                            self.driver.execute_script(
                                "arguments[0].setAttribute('style', arguments[1]);", 
                                element, 
                                "border: 2px solid red; background-color: yellow;"
                            )
                            time.sleep(duration)
                        except Exception as e:
                            print(f"⚠️ Could not highlight element: {e}")
                            return
                            
                        # Restore original style
                        try:
                            self.driver.execute_script(
                                "arguments[0].setAttribute('style', arguments[1]);", 
                                element, 
                                original_style
                            )
                        except:
                            pass  # Ignore errors when restoring style
                            
                    except Exception as e:
                        print(f"⚠️ Error highlighting element: {e}")
                        # Don't re-raise the exception - highlighting is optional
                
                def click(self, element):
                    """Click an element with visual feedback and error handling."""
                    try:
                        # Get element tag name safely
                        try:
                            tag_name = element.tag_name
                            print(f"👆 Clicking element: {tag_name}")
                        except Exception:
                            # If we can't get the tag name, just use a generic message
                            print(f"👆 Clicking element (unknown tag)")
                        
                        self.highlight_element(element)
                        # Scroll into view first
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(0.3)  # Brief pause after scrolling
                        # Try direct click
                        element.click()
                        print("✅ Element clicked successfully")
                    except Exception as e:
                        print(f"⚠️ Direct click failed, trying JavaScript click: {e}")
                        # Fall back to JavaScript click
                        try:
                            self.driver.execute_script("arguments[0].click();", element)
                            print("✅ Element clicked with JavaScript")
                        except Exception as e:
                            print(f"❌ JavaScript click also failed: {e}")
                            raise
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
    
    def generate_code(self, user_request=None):
        """Generate code based on the current page state and optional user request.
        
        Args:
            user_request: Optional user request to guide the code generation
            
        Returns:
            The generated code as a string
        """
        # Get the current page HTML to help with element selection
        html_snippet = self.get_page_html(max_length=10000)
        current_url = self.driver.driver.current_url
        
        # Format the user request if provided
        user_request_text = ""
        if user_request:
            user_request_text = f"User request: {user_request}\n\n"
        
        # Format the prompt with current page information
        prompt = ACTION_SUGGESTION_PROMPT.format(
            current_url=current_url,
            html_snippet=html_snippet,
            user_request=user_request_text
        )
        
        # Save the prompt for debugging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_to_file(prompt, self.prompts_dir, f"prompt_{timestamp}.txt")
        
        # Generate the response from the LLM
        action_description = "Generating code"
        if user_request:
            action_description = f"Generating code for: {user_request}"
        print(Fore.CYAN + f"🤖 {action_description}...")
        
        response = self.llm_client.generate_response(prompt, temperature=DEFAULT_TEMPERATURE)
        
        # Save the response for debugging
        self._save_to_file(response, self.responses_dir, f"response_{timestamp}.txt")
        
        # Extract code from the response
        return self._extract_code_from_response(response)
    
    def run_interactive_session(self, max_actions=100):
        """Run an interactive session with the user.
        
        Args:
            max_actions: Maximum number of actions to take before stopping
        """
        try:
            # Navigate to the website using the driver's method
            print(Fore.CYAN + f"🌐 Navigating to {self.website}")
            self.driver.navigate_to_url()
            time.sleep(2)
            self.driver.take_screenshot("initial_page")
            
            # Interactive session loop
            action_count = 0
            while action_count < max_actions:
                try:
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
                    
                    # Generate code based on the user's request or get a suggestion
                    code = self.generate_code(user_request=user_input if user_input.strip() else None)
                    
                    # Execute the generated code
                    description = user_input if user_input.strip() else "Suggested action"
                    self.execute_code(code, description=description)
                    
                    action_count += 1
                    
                except Exception as e:
                    # Catch exceptions within the loop to prevent the entire session from crashing
                    print(Fore.RED + f"❌ Error in action: {e}")
                    import traceback
                    print(Fore.RED + traceback.format_exc())
                    # Take a screenshot of the error state
                    self.driver.take_screenshot(f"error_{datetime.now().strftime('%H%M%S')}")
                    
                    # Ask if the user wants to continue
                    print(Fore.YELLOW + "Do you want to continue? (y/n)")
                    continue_input = input(Fore.YELLOW + "> ")
                    if continue_input.lower() not in ['y', 'yes']:
                        print(Fore.CYAN + "👋 Exiting interactive session")
                        break
                
        except Exception as e:
            print(Fore.RED + f"❌ Error in interactive session: {e}")
            import traceback
            print(Fore.RED + traceback.format_exc())
        finally:
            # Always take a final screenshot
            try:
                self.driver.take_screenshot("final_state")
            except:
                pass  # Ignore errors in taking the final screenshot
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.cleanup()

def main():
    """Main function to run the Selenium code generator."""
    parser = argparse.ArgumentParser(description='LLM-guided Selenium code generator')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use (hunyuan or ollama)')
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Model name to use. For Ollama, check first what\'s available on the server. For DeepSeek: "deepseek-chat" or "deepseek-reasoner". For Mistral: "mistral-medium-latest" (default) or "mistral-large-latest"',
    )
    parser.add_argument('--website', type=str, default='https://quotes.toscrape.com',
                      help='Website to interact with')
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
            headless=args.headless,
            model=args.model
        )

        # Run the interactive session
        generator.run_interactive_session(
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
