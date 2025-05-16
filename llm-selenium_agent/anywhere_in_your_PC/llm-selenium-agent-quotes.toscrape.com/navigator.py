"""
LLM-guided Navigator for quotes.toscrape.com website.
"""

import os
import argparse
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from selenium_driver import SeleniumDriver
from llm_client import LLMClient
from config import (
    PROMPT_TEMPLATE, 
    DEFAULT_MAX_ACTIONS, 
    DEFAULT_LLM_PROVIDER,
    LLM_CONFIG
)

# Load environment variables from .env file
load_dotenv()

class QuotesNavigator:
    """LLM-guided navigator for quotes.toscrape.com."""
    
    def __init__(
        self, 
        provider: str = DEFAULT_LLM_PROVIDER,
        headless: bool = False,
        extra_rules: List[str] = None
    ):
        """Initialize the navigator.
        
        Args:
            provider: The LLM provider to use
            headless: Whether to run the browser in headless mode
            extra_rules: Additional rules to guide the LLM
        """
        self.provider = provider
        self.headless = headless
        self.extra_rules = extra_rules or []
        self.user_rule = ""  # Initialize user rule as empty
        self.llm_client = LLMClient(provider=provider)
        self.selenium_driver = None
        self.last_action = "INITIALIZE"
        
        # Create debug directory for prompts and responses
        self.debug_dir = f"selenium_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def start(self):
        """Start the navigator session by initializing the Selenium driver."""
        self.selenium_driver = SeleniumDriver(headless=self.headless)
        self.selenium_driver.prepare_environment()
        # Set the base URL and navigate to it
        self.selenium_driver.navigate_to_url()
        # Use the same debug directory as selenium driver
        self.debug_dir = self.selenium_driver.debug_dir
    
    def stop(self):
        """Stop the navigator session by closing the Selenium driver."""
        if self.selenium_driver:
            self.selenium_driver.terminate_webdriver()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the navigation.
        
        Returns:
            A dictionary containing the current state
        """
        return {
            "current_url": self.selenium_driver.driver.current_url,
            "last_action": self.last_action,
            "logged_in": self.selenium_driver.logged_in,
        }
    
    def get_user_rule(self) -> str:
        """Get a new rule from the user.
        
        Returns:
            The user's input rule
        """
        print("\n=== Waiting for User Input ===")
        print("Enter your new rule (this will be treated as HIGHEST PRIORITY command):")
        
        user_input = input("> ")
        
        # Format the user input to emphasize its importance
        if user_input.lower().strip():
            formatted_input = (
                f"CRITICAL USER COMMAND (YOU MUST FOLLOW THIS EXACTLY): {user_input}"
            )
            
            # If user is asking to login, add extra emphasis
            if "login" in user_input.lower():
                formatted_input = f"EXTREMELY IMPORTANT - YOU MUST LOGIN NOW. USER COMMAND: {user_input} - YOUR NEXT ACTION MUST BE LOGIN."
            
            return formatted_input
        
        return user_input
    
    def save_to_file(self, content: str, round_num: int, filename: str):
        """Save content to a file in the round's debug folder.
        
        Args:
            content: Content to save
            round_num: Round number for folder name
            filename: Name of the file
        """
        # Create round directory
        round_dir = os.path.join(self.debug_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Save content to file
        file_path = os.path.join(round_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Saved {filename} to {file_path}")
    
    def consult_llm(self, current_round: int, max_rounds: int) -> Dict[str, str]:
        """Consult the LLM for guidance on the next action.
        
        Args:
            current_round: The current round of action
            max_rounds: The maximum number of rounds
            
        Returns:
            A dictionary containing the LLM's guidance
        """
        # Get current state
        state = self.get_current_state()
        
        # Get HTML snippet from current page
        html_snippet = self.selenium_driver.get_page_html_snippet()
        
        # Format extra rules if provided
        extra_rules_text = ""
        if self.extra_rules:
            extra_rules_text = "\n" + "\n".join(
                f"- {rule}" for rule in self.extra_rules
            )
        
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            html_snippet=html_snippet,
            current_url=state["current_url"],
            last_action=state["last_action"],
            logged_in=state["logged_in"],
            extra_rules=extra_rules_text,
            user_rule=self.user_rule,
            current_round=current_round,
            max_rounds=max_rounds
        )
        
        # Save prompt to file
        self.save_to_file(prompt, current_round, "prompt.txt")
        
        # Get LLM config based on provider
        llm_kwargs = LLM_CONFIG.get(self.provider, {})
        
        # Get response from LLM
        print("Consulting LLM for next action...")
        response = self.llm_client.generate_response(prompt, **llm_kwargs)
        
        # Save response to file
        self.save_to_file(response, current_round, "response.txt")
        
        # Parse the response
        return self.parse_llm_response(response)
    
    def parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM's response into structured data.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A dictionary containing the parsed response
        """
        # Default values
        parsed = {
            "action": "None",  # Safe default
            "reason": "None",
            "details": "None",
        }
        
        # Extract sections from the response
        for line in response.split("\n"):
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
        
        # Print reason before action execution
        print(f"Reason: {action_data['reason']}")
        print(f"\n=== Executing Action: {action} ===")
        
        # Update last action
        self.last_action = action
        
        # Execute the appropriate action based on the LLM's suggestion
        if action == "NAVIGATE_NEXT_PAGE":
            return self.selenium_driver.navigate_to_next_page()
        
        elif action == "NAVIGATE_PREVIOUS_PAGE":
            return self.selenium_driver.navigate_to_previous_page()
        
        elif action == "VISIT_AUTHOR_PAGE":
            author = action_data["details"]
            if author.lower() == "none" or not author:
                print("No specific author specified. Cannot visit author page.")
                return True
            
            return self.selenium_driver.visit_author_page(author)
        
        elif action == "FILTER_BY_TAG":
            tag = action_data["details"]
            if tag.lower() == "none" or not tag:
                print("No specific tag specified. Cannot filter by tag.")
                return True
            
            return self.selenium_driver.filter_by_tag(tag)
        
        elif action == "LOGIN":
            if self.selenium_driver.logged_in:
                print("Already logged in.")
                return True
            
            # Add special highlighting for login
            print("=" * 50)
            print(" >>> EXECUTING USER-REQUESTED LOGIN <<< ")
            print("=" * 50)
            
            login_success = self.selenium_driver.login()
            
            # Report login result
            print("LOGIN RESULT: " + ("SUCCESSFUL" if login_success else "FAILED"))
            
            return login_success
        
        elif action == "LOGOUT":
            if not self.selenium_driver.logged_in:
                print("Not currently logged in.")
                return False
            
            return self.selenium_driver.logout_from_site()
        else:
            print(f"Unknown action: {action}")
            return False
    
    def run(self, max_actions: int = DEFAULT_MAX_ACTIONS) -> None:
        """Run the LLM-guided navigation process.
        
        Args:
            max_actions: Maximum number of actions to take
        """
        print("\n" + "=" * 60)
        print(f"Starting LLM-guided website navigation using {self.provider}")
        print("=" * 60 + "\n")
        
        try:
            # Start the navigation session
            self.start()
            
            # Take initial screenshot
            self.selenium_driver.take_screenshot("initial_page.png")
            
            # Add initial rules for the first action
            self.user_rule = ""
            
            action_count = 0
            while action_count < max_actions:
                print(f"\n=== Action {action_count + 1}/{max_actions} ===")
                
                # After every two actions, get new rule from user
                if action_count > 0 and action_count % 2 == 0:
                    self.user_rule = self.get_user_rule()
                    print(f"New user rule: {self.user_rule}")
                
                # Get guidance from LLM
                guidance = self.consult_llm(action_count + 1, max_actions)
                
                # Execute the suggested action
                print(f"LLM suggests: {guidance['action']}")
                
                success = self.execute_action(guidance)
                
                # Add a small delay to be respectful to the server
                delay = random.uniform(1, 2)
                time.sleep(delay)
                
                action_count += 1
                
                # Check if we should stop
                if not success:
                    print("Stopping navigation process as recommended by LLM or due to action failure.")
                    break
            
            # Navigation complete
            print("\n=== Navigation Complete ===")
            print(f"Executed {action_count} actions")
            
        except KeyboardInterrupt:
            print("\nNavigation interrupted by user.")
        except Exception as e:
            print(f"\nError during navigation: {e}")
        finally:
            # Always clean up
            self.stop()

def main():
    """Run the LLM-guided quotes navigator."""
    parser = argparse.ArgumentParser(
        description="LLM-guided quotes.toscrape.com navigator"
    )
    parser.add_argument(
        "rules",
        nargs="*",
        help="Additional rules to guide the LLM (optional, can provide multiple)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=["hunyuan", "ollama"],
        help=f"LLM provider to use (default: {DEFAULT_LLM_PROVIDER})",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_MAX_ACTIONS,
        help=f"Maximum number of actions to take (default: {DEFAULT_MAX_ACTIONS})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode",
    )
    args = parser.parse_args()
    
    navigator = QuotesNavigator(
        provider=args.provider,
        headless=args.headless,
        extra_rules=args.rules,
    )
    
    navigator.run(max_actions=args.max_actions)

if __name__ == "__main__":
    main() 