"""
LLM-guided Navigator for YouTube videos with specific view counts.
"""

import os
import argparse
import random
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from colorama import Fore, Back, Style, init
import pyfiglet

from selenium_driver import SeleniumDriver
from llm_client import LLMClient
from config import (
    PROMPT_TEMPLATE,
    DEFAULT_MAX_ACTIONS,
    DEFAULT_MAX_VIDEOS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TARGET_VIEWS,
    LLM_CONFIG,
)

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

class YouTubeNavigator:
    """LLM-guided navigator for finding YouTube videos with specific view counts."""
    
    def __init__(
        self, 
        provider: str = DEFAULT_LLM_PROVIDER,
        target_views: int = DEFAULT_TARGET_VIEWS,
        headless: bool = False,
        ollama_server: str = "localhost",
        extra_rules: List[str] = None
    ):
        """Initialize the navigator.
        
        Args:
            provider: The LLM provider to use ("hunyuan" or "ollama")
            target_views: Target number of views to find
            headless: Whether to run the browser in headless mode
            ollama_server: Hostname or IP of Ollama server if using Ollama
            extra_rules: Additional rules to guide the LLM
        """
        self.provider = provider
        self.target_views = target_views
        self.headless = headless
        self.ollama_server = ollama_server
        self.extra_rules = extra_rules or []
        self.last_action = "INITIALIZE"
        self.llm_client = LLMClient(provider=provider)
        self.selenium_driver = None
        self.conversation_history = []
        
        # Create debug directory for prompts and responses
        self.debug_dir = f"youtube_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def start(self):
        """Start the navigator session by initializing the Selenium driver."""
        print(Fore.BLUE + "Initializing Selenium driver...")
        self.selenium_driver = SeleniumDriver(headless=self.headless, target_views=self.target_views)
        self.selenium_driver.prepare_environment()
        # Use the same debug directory
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
        # Check if videos are present
        videos_present = self.selenium_driver.are_videos_present()
        
        # Get the most recent videos checked (up to 20)
        recent_videos = self.selenium_driver.videos_checked[-20:] if len(self.selenium_driver.videos_checked) > 20 else self.selenium_driver.videos_checked
        
        return {
            "current_url": self.selenium_driver.driver.current_url,
            "current_search": self.selenium_driver.current_search,
            "videos_checked": recent_videos,
            "target_view_videos": self.selenium_driver.target_view_videos,
            "last_action": self.last_action,
            "videos_present": videos_present
        }
    
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
        
        print(Fore.BLUE + f"Saved {filename} to {file_path}")
    
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
        
        # Format videos_checked as text list
        videos_checked_text = ", ".join([f'"{v["title"]}"' for v in state["videos_checked"]]) if state["videos_checked"] else "None"
        
        # Format target_view_videos as text list
        target_view_videos_text = ", ".join([f'"{v["title"]}"' for v in state["target_view_videos"]]) if state["target_view_videos"] else "None"
        
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            html_snippet=html_snippet,
            current_url=state["current_url"],
            current_search=state["current_search"],
            videos_checked=videos_checked_text,
            target_view_videos=target_view_videos_text,
            last_action=state["last_action"],
            target_views=self.target_views,
            videos_present=state["videos_present"],
            extra_rules=extra_rules_text
        )
        
        # Save prompt to file
        self.save_to_file(prompt, current_round, "prompt.txt")
        
        # Get LLM config based on provider
        llm_kwargs = LLM_CONFIG.get(self.provider, {}).copy()
        
        # If using Ollama, add the server parameter
        if self.provider == "ollama":
            llm_kwargs["server"] = self.ollama_server
        
        # Get response from LLM - simple single-turn request
        print(Fore.YELLOW + "Consulting LLM for next action...")
        response = self.llm_client.generate_response(prompt, **llm_kwargs)
        
        # Save response to file
        self.save_to_file(response, current_round, "response.txt")
        
        # Add to conversation history
        self.conversation_history.append({
            "round": current_round,
            "prompt": prompt,
            "response": response,
            "parsed_response": self.parse_llm_response(response)
        })
        
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
            "action": "SEARCH_VIDEOS",  # Safe default
            "reason": "Default action",
            "details": "No search query provided"  # Default search query
        }
        
        # Extract sections from the response
        for line in response.split("\n"):
            if line.startswith("ACTION:"):
                parsed["action"] = line.replace("ACTION:", "").strip()
            elif line.startswith("REASON:"):
                parsed["reason"] = line.replace("REASON:", "").strip()
            elif line.startswith("DETAILS:"):
                # Extract and clean the details field
                details = line.replace("DETAILS:", "").strip()
                
                # For SEARCH_VIDEOS action, ensure we're only extracting the search query
                if parsed["action"] == "SEARCH_VIDEOS":
                    # Check for quoted content
                    import re
                    quoted_content = re.findall(r'"([^"]*)"', details)
                    if quoted_content:
                        details = quoted_content[0]
                    
                    # Clean up the query
                    details = details.strip('"\'')
                    
                    # Remove any instances of action formatting that might have leaked through
                    details = re.sub(r'ACTION:\s*\w+', '', details)
                    details = re.sub(r'REASON:\s*.*', '', details)
                    details = re.sub(r'DETAILS:\s*', '', details)
                    
                    # Make sure it's just the search query
                    if "search" in details.lower() and len(details.split()) > 4:
                        # Likely contains instructions rather than a query
                        # Extract just a potential search term
                        search_terms = re.findall(r'"([^"]*)"', details)
                        if search_terms:
                            details = search_terms[0]
                        else:
                            # Just get the last part which might be the actual query
                            parts = details.split()
                            details = " ".join(parts[-3:])
                    
                    # Limit search query to a reasonable length (50 chars max)
                    if len(details) > 50:
                        details = details[:50]
                
                parsed["details"] = details.strip()
        
        # Final check to ensure we don't pass LLM formatting directly to YouTube
        if parsed["action"] == "SEARCH_VIDEOS":
            if "ACTION:" in parsed["details"] or "REASON:" in parsed["details"]:
                # Still has LLM formatting, use a generic search
                parsed["details"] = "recently uploaded videos with few views"
        
        return parsed
    
    def get_initial_search_query(self) -> str:
        """Generate an initial search query by consulting the LLM.
        
        Returns:
            A search query string
        """
        # Create a simple prompt for the initial search
        prompt = f"""
You are helping with a YouTube search task to find videos with exactly {self.target_views} views.

Please suggest ONE specific search query to find videos that are likely to have {self.target_views} views.
- Newly uploaded videos or obscure content for low view counts
- Niche tutorials or hobbyist content for moderate view counts
- Specific search operators (like "sort by upload date" or similar)

Respond with ONLY the suggested search query as a single line of text, nothing else. 
Do not include labels, formatting, or explanations.
"""
        print(Fore.YELLOW + "Generating initial search query...")
        
        # Get LLM config based on provider
        llm_kwargs = LLM_CONFIG.get(self.provider, {}).copy()
        
        # If using Ollama, add the server parameter
        if self.provider == "ollama":
            llm_kwargs["server"] = self.ollama_server
        
        # Get search query from LLM - simple single-turn request
        response = self.llm_client.generate_response(prompt, **llm_kwargs)
        
        # Clean the response - remove any formatting or instructions
        search_query = response.strip().strip('"\'')
        
        # Remove any instances of LLM formatting that might have leaked through
        import re
        search_query = re.sub(r'ACTION:\s*\w+', '', search_query)
        search_query = re.sub(r'REASON:\s*.*', '', search_query)
        search_query = re.sub(r'DETAILS:\s*', '', search_query)
        
        # If response is too structured/long, extract just the query part
        if len(search_query.split('\n')) > 1 or len(search_query.split()) > 15:
            # Try to extract quoted content first
            quoted = re.findall(r'"([^"]*)"', search_query)
            if quoted:
                search_query = quoted[0]
            else:
                # Just use the first line
                search_query = search_query.split('\n')[0]
        
        # Remove search-related instructions that might have been included
        search_query = re.sub(r'search(\s+for)?:', '', search_query, flags=re.IGNORECASE)
        search_query = re.sub(r'query:', '', search_query, flags=re.IGNORECASE)
        
        # Limit length
        if len(search_query) > 100:
            search_query = search_query[:100]
        
        # Make sure we have a reasonable query
        if len(search_query.strip()) < 3:
            # Fallback if we got something unusable
            if self.target_views == 0:
                search_query = "first upload recently uploaded"
            else:
                search_query = f"niche videos uploaded recently"
        
        print(Fore.GREEN + f"Generated search query: {search_query}")
        return search_query.strip()
    
    def execute_action(self, action_data: Dict[str, str]) -> bool:
        """Execute the action suggested by the LLM.
        
        Args:
            action_data: The parsed LLM response containing action details
            
        Returns:
            Boolean indicating success or failure
        """
        action = action_data["action"]
        
        # Print reason before action execution
        print(Fore.CYAN + f"Reason: {action_data['reason']}")
        
        # Format action as ASCII art
        action_banner = pyfiglet.figlet_format(f"Action: {action}", font="small")
        print(Fore.GREEN + action_banner)
        
        # Update last action
        self.last_action = action
        
        # Execute the appropriate action based on the LLM's suggestion
        if action == "SEARCH_VIDEOS":
            return self.selenium_driver.search_videos(action_data["details"])
        
        elif action == "SCROLL_DOWN":
            success = self.selenium_driver.scroll_down()
            # If scrolling failed, attempt a new search with LLM-generated query
            if not success:
                print(Fore.YELLOW + "No results found. Generating new search query...")
                new_search = self.get_initial_search_query()
                print(Fore.YELLOW + f"Trying alternative search: {new_search}")
                return self.selenium_driver.search_videos(new_search)
            return success
        
        elif action == "OPEN_VIDEO":
            return self.selenium_driver.open_video(action_data["details"])
        
        elif action == "CHECK_RECOMMENDATIONS":
            return self.selenium_driver.check_recommendations()
        
        elif action == "COMPLETE_SEARCH":
            print(Fore.MAGENTA + "Search process complete as suggested by LLM.")
            return False  # Signal to stop the scraping loop
        
        else:
            print(Fore.RED + f"Unknown action: {action}")
            return False
    
    def save_data(self, results_file: str = None) -> None:
        """Save the collected data to a JSON file.
        
        Args:
            results_file: Filename for results data
        """
        # Use default filename that reflects target view count
        if results_file is None:
            results_file = os.path.join(self.debug_dir, f"youtube_{self.target_views}_views_results.json")
            
        # Enhanced data saving: include more information about the videos
        target_view_videos_enhanced = []
        for video in self.selenium_driver.target_view_videos:
            # Make a copy of the video data
            enhanced_video = video.copy()
            # Add additional metadata if available
            if "metadata" not in enhanced_video:
                enhanced_video["metadata"] = {
                    "target_views": self.target_views,
                    "matched_exactly": enhanced_video.get("views") == self.target_views,
                    "found_on_date": datetime.now().strftime("%Y-%m-%d"),
                    "found_with_search": self.selenium_driver.current_search,
                    "found_at_url": enhanced_video.get("url", ""),
                    "view_difference": abs(enhanced_video.get("views", 0) - self.target_views)
                }
            target_view_videos_enhanced.append(enhanced_video)
            
        # Prepare data to save
        results = {
            "target_view_videos": target_view_videos_enhanced,
            "videos_checked": self.selenium_driver.videos_checked,
            "search_queries_used": list(set([h.get("parsed_response", {}).get("details") 
                                for h in self.conversation_history 
                                if h.get("parsed_response", {}).get("action") == "SEARCH_VIDEOS"])),
            "target_views": self.target_views,
            "timestamp": datetime.now().isoformat(),
            "total_actions_executed": len(self.conversation_history),
            "total_videos_checked": len(self.selenium_driver.videos_checked),
            "success_rate": len(target_view_videos_enhanced) / len(self.selenium_driver.videos_checked) if self.selenium_driver.videos_checked else 0,
            "run_duration": {
                "start_time": self.conversation_history[0]["round"] if self.conversation_history else None,
                "end_time": datetime.now().isoformat(),
            },
            "system_info": {
                "llm_provider": self.provider,
                "headless_mode": self.headless,
                "browser": "Chrome",
            }
        }
        
        # Save to file
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(Fore.GREEN + f"Saved enhanced results to {results_file}")
        
        # Also save conversation history
        conversation_file = os.path.join(self.debug_dir, "conversation_history.json")
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        
        print(Fore.GREEN + f"Saved conversation history to {conversation_file}")
    
    def run(self, max_actions: int = DEFAULT_MAX_ACTIONS, max_target_videos: int = DEFAULT_MAX_VIDEOS) -> None:
        """Run the LLM-guided navigation process.
        
        Args:
            max_actions: Maximum number of actions to take
            max_target_videos: Stop after finding this many videos with target view count
        """
        # Create fancy title
        title = pyfiglet.figlet_format("YouTube View Finder", font="slant")
        print(Fore.MAGENTA + title)
        print(Fore.CYAN + "=" * 60)
        print(Fore.YELLOW + f"Starting LLM-guided YouTube navigation using {self.provider}")
        print(Fore.YELLOW + f"Target view count: {self.target_views}")
        print(Fore.CYAN + "=" * 60 + "\n")
        
        try:
            # Start the navigation session
            self.start()
            
            # Make sure the driver was successfully initialized
            if not hasattr(self.selenium_driver, 'driver') or self.selenium_driver.driver is None:
                print(Fore.RED + "Failed to initialize Selenium driver. Exiting.")
                return
            
            # Generate initial search query using the LLM
            initial_search = self.get_initial_search_query()
            search_successful = False
            try:
                search_successful = self.selenium_driver.search_videos(initial_search)
            except Exception as e:
                print(Fore.RED + f"Error during initial search: {e}")
                
            if not search_successful:
                print(Fore.RED + "Failed to perform initial search. Generating recovery search query...")
                recovery_search = self.get_initial_search_query()
                try:
                    search_successful = self.selenium_driver.search_videos(recovery_search)
                except Exception as e:
                    print(Fore.RED + f"Error during recovery search: {e}")
                    
                if not search_successful:
                    print(Fore.RED + "Recovery search also failed. Exiting.")
                    self.stop()
                    return
            
            action_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while action_count < max_actions:
                # Display progress
                progress = f"Action {action_count + 1}/{max_actions}"
                print(Fore.CYAN + "\n" + "=" * 60)
                print(Fore.YELLOW + Style.BRIGHT + progress.center(60))
                print(Fore.CYAN + "=" * 60)
                
                # Get guidance from LLM
                guidance = self.consult_llm(action_count + 1, max_actions)
                
                # Execute the suggested action
                print(Fore.YELLOW + f"LLM suggests: {guidance['action']}")
                
                success = self.execute_action(guidance)
                
                # Handle success/failure
                if success:
                    consecutive_failures = 0  # Reset failure counter on success
                    action_count += 1
                else:
                    consecutive_failures += 1
                    print(Fore.RED + f"Action failed. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print(Fore.RED + "Too many consecutive failures. Trying to recover...")
                        
                        # Try recovery actions
                        try:
                            # Refresh page and try a new search
                            self.selenium_driver.driver.refresh()
                            time.sleep(3)
                            
                            # Generate a new recovery search query
                            recovery_search = self.get_initial_search_query()
                            if self.selenium_driver.search_videos(recovery_search):
                                print(Fore.GREEN + "Recovery successful!")
                                consecutive_failures = 0
                            else:
                                print(Fore.RED + "Recovery failed. Stopping navigation process.")
                                break
                                
                        except Exception as e:
                            print(Fore.RED + f"Error during recovery: {e}")
                            break
                    
                    # Skip counting this as an action if it failed
                    continue
                
                # Add a small delay to be respectful to the server
                delay = random.uniform(1, 2)
                time.sleep(delay)
                
                # Check if we should stop
                if guidance["action"] == "COMPLETE_SEARCH":
                    print(Fore.MAGENTA + "Stopping navigation process as recommended by LLM.")
                    break
                
                # Stop if we've found enough videos with the target view count
                if len(self.selenium_driver.target_view_videos) >= max_target_videos:
                    print(Fore.GREEN + f"Success! Found {len(self.selenium_driver.target_view_videos)} videos with {self.target_views} views. Stopping search.")
                    break
            
            # Navigation complete
            completion_banner = pyfiglet.figlet_format("Done!", font="big")
            print(Fore.GREEN + completion_banner)
            print(Fore.YELLOW + f"Executed {action_count} actions")
            print(Fore.YELLOW + f"Checked {len(self.selenium_driver.videos_checked)} videos")
            print(Fore.YELLOW + f"Found {len(self.selenium_driver.target_view_videos)} videos with {self.target_views} views")
            
            # Save the data
            self.save_data()
            
            # Print videos found with target view count
            if self.selenium_driver.target_view_videos:
                result_banner = pyfiglet.figlet_format(f"Found {len(self.selenium_driver.target_view_videos)} videos!", font="small")
                print(Fore.GREEN + result_banner)
                for i, video in enumerate(self.selenium_driver.target_view_videos, 1):
                    print(Fore.CYAN + f"{i}. {video['title']}")
                    print(Fore.YELLOW + f"   URL: {video['url']}")
                    print(Fore.BLUE + f"   Found at: {video['timestamp']}")
                    print()
            else:
                print(Fore.YELLOW + f"\nNo videos with exactly {self.target_views} views were found.")
            
        except KeyboardInterrupt:
            print(Fore.RED + "\nNavigation interrupted by user.")
        except Exception as e:
            print(Fore.RED + f"\nError during navigation: {e}")
        finally:
            # Always clean up
            self.stop()

def main():
    """Run the LLM-guided YouTube video finder."""
    parser = argparse.ArgumentParser(
        description="LLM-guided YouTube video finder with specific view counts"
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
        "--target-views",
        type=int,
        default=DEFAULT_TARGET_VIEWS,
        help=f"Target number of views to find (default: {DEFAULT_TARGET_VIEWS})",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_MAX_ACTIONS,
        help=f"Maximum number of actions to take (default: {DEFAULT_MAX_ACTIONS})",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help=f"Stop after finding this many target videos (default: {DEFAULT_MAX_VIDEOS})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode",
    )
    parser.add_argument(
        "--ollama-server",
        type=str,
        default="localhost",
        help="IP address or hostname of the Ollama server (default: localhost)",
    )
    args = parser.parse_args()
    
    navigator = YouTubeNavigator(
        provider=args.provider,
        target_views=args.target_views,
        headless=args.headless,
        ollama_server=args.ollama_server,
        extra_rules=args.rules,
    )
    
    navigator.run(max_actions=args.max_actions, max_target_videos=args.max_videos)

if __name__ == "__main__":
    main() 