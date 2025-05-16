#!/usr/bin/env python
"""
LLM-Guided YouTube Zero Views Finder

This script demonstrates how to use a local LLM (via Ollama) to guide
Selenium in searching for YouTube videos with zero views. The LLM analyzes
the page structure, makes navigation decisions, and helps determine search strategies.
"""

import os
import json
import time
import random
import argparse
import requests
import sys
import subprocess
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException

# Get the best Ollama model
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
PROMPT_TEMPLATE = """
You are an AI assistant tasked with guiding a web scraping operation to find YouTube videos with exactly {target_views} views.

Current page HTML snippet:
```html
{html_snippet}
```

Current state:
- URL: {current_url}
- Current search query: {current_search}
- Videos checked: {videos_checked}
- Target view videos found: {target_view_videos}
- Last action: {last_action}
- Target view count: {target_views}
- Videos present on page: {videos_present}

Based on the HTML snippet above and the current state, please suggest what actions to take next.
Choose from these possible actions:
1. SEARCH_VIDEOS - Search for videos using a specific query (provide a search query focused on finding videos likely to have {target_views} views)
2. SCROLL_DOWN - Scroll down to load more videos
3. OPEN_VIDEO - Open a specific video from the current page (specify which one by position or title)
4. CHECK_RECOMMENDATIONS - Check video recommendations on the current video page
5. BACK_TO_RESULTS - Go back to search results
6. REFINE_SEARCH - Modify the current search query to better target low-view videos
7. COMPLETE_SEARCH - End the search process (if we've found videos with {target_views} views or exhausted options)

Important guidelines:
- ONLY use SEARCH_VIDEOS action when there are no videos visible on the current page to click on (videos_present = False).
- When videos are visible (videos_present = True), always prefer to OPEN_VIDEO or SCROLL_DOWN to find more videos rather than performing a new search.
- When on a search results page, immediately look for videos with view counts close to the target of {target_views} views.
- Use "closest views" or "best match" as the identifier to find videos with view counts nearest to {target_views}.
- Look for and prioritize opening videos with view counts near our target of {target_views}.
- If you see any videos with view counts close to {target_views} in the snippet, immediately suggest opening that video.
- If on a video page and no suitable videos are found, check recommendations or go back to search results.
- For search queries, use YouTube-specific search operators like "uploaded today", "uploaded within hour", or "uploaded minutes ago"
- Add "sort by upload date" to your search queries to prioritize newest content
- After checking about 10-15 videos with no success, consider changing the search query completely
- If "No results found" appears, immediately suggest a new and different search query
{extra_rules}

Your response should be structured like this:
ACTION: [chosen action]
REASON: [brief explanation of why this action is appropriate]
DETAILS: [any specific details needed for the action, like search query or which video to open]

Example response:
ACTION: OPEN_VIDEO
REASON: I see a video with a view count that could potentially have {target_views} views.
DETAILS: closest views
"""

class LLMGuidedYouTubeScraper:
    """A Selenium scraper guided by an LLM for finding YouTube videos with a specific view count."""
    
    def __init__(self, ollama_model: str = None, extra_rules: List[str] = None, ollama_server: str = "localhost", target_views: int = 0):
        """Initialize the LLM-guided scraper.
        
        Args:
            ollama_model: The Ollama model to use for guidance (if None, best model will be selected)
            extra_rules: Additional rules to guide the LLM
            ollama_server: IP address or hostname of the Ollama server
            target_views: Target number of views to find
        """
        self.base_url = "https://www.youtube.com/"
        self.ollama_server = ollama_server
        self.ollama_api_url = f"http://{ollama_server}:11434/api/generate"
        
        # If no model specified, get the best available model
        if ollama_model is None:
            self.ollama_model = get_best_ollama_model(self.ollama_server)
        else:
            self.ollama_model = ollama_model
            
        # Initialize state
        self.current_search = ""
        self.videos_checked = []
        self.target_view_videos = []  # Renamed from zero_view_videos
        self.last_action = "INITIALIZE"
        self.conversation_history = []
        self.extra_rules = extra_rules or []
        self.target_views = target_views
        
        # Create a directory for storing LLM responses and debug info
        self.debug_dir = f"youtube_scraper_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create screenshots directory within our debug directory
        self.screenshots_dir = os.path.join(self.debug_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Set up the webdriver
        self.setup_webdriver()
    
    def setup_webdriver(self):
        """Set up the Selenium WebDriver with appropriate options."""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-notifications")
        options.add_argument("--mute-audio")
        # Uncomment the line below if you want to run in headless mode
        # options.add_argument("--headless")
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)
    
    def get_page_html_snippet(self, max_length: int = 5000) -> str:
        """Get a snippet of the current page's HTML.
        
        Args:
            max_length: Maximum length of the HTML snippet to return
        
        Returns:
            A string containing a portion of the page's HTML
        """
        html = self.driver.page_source
        
        # Get a reasonable snippet that's not too large
        if len(html) > max_length:
            # Try to find main content area based on YouTube's layout
            try:
                if "youtube.com/results" in self.driver.current_url:
                    # For search results page
                    main_content = self.driver.find_element(By.ID, "contents").get_attribute("outerHTML")
                elif "youtube.com/watch" in self.driver.current_url:
                    # For video page
                    main_content = self.driver.find_element(By.ID, "primary").get_attribute("outerHTML")
                else:
                    # Default to body content
                    main_content = self.driver.find_element(By.TAG_NAME, "body").get_attribute("outerHTML")
                
                # Enhance the snippet with view count information
                if "youtube.com/results" in self.driver.current_url:
                    try:
                        main_content = self.enhance_html_with_view_counts(main_content)
                    except Exception as e:
                        print(f"Error enhancing HTML with view counts: {e}")
                
                return main_content[:max_length]
            except:
                # Fall back to truncating the whole HTML
                return html[:max_length] + "..."
        
        return html
    
    def enhance_html_with_view_counts(self, html_content: str) -> str:
        """Parse the HTML to extract and highlight videos with low view counts.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            Enhanced HTML content with view count information
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all video renderer elements
        video_renderers = soup.find_all('div', id=lambda x: x and 'video-renderer' in x)
        
        for renderer in video_renderers:
            try:
                # Try to find the view count
                view_count_element = renderer.select_one('span.style-scope.ytd-video-meta-block:not(#metadata-line)')
                
                if view_count_element:
                    view_count_text = view_count_element.text.strip()
                    
                    # Extract the numeric view count
                    view_count = 0
                    if 'No views' in view_count_text or '0 views' in view_count_text:
                        view_count = 0
                    else:
                        # Parse the number from strings like "123 views" or "1.2K views"
                        numeric_part = re.search(r'([\d,\.]+)', view_count_text)
                        if numeric_part:
                            number_str = numeric_part.group(1).replace(',', '')
                            if 'K' in view_count_text:
                                view_count = int(float(number_str) * 1000)
                            elif 'M' in view_count_text:
                                view_count = int(float(number_str) * 1000000)
                            else:
                                view_count = int(float(number_str))
                    
                    # Get video title and add view count as a comment
                    title_element = renderer.select_one('a#video-title')
                    if title_element and view_count is not None:
                        title_element.string = f"{title_element.text.strip()} <!-- VIEW_COUNT: {view_count} -->"
                        
                        # Add visual indicator for videos with view count close to target
                        # Check if view count is within 5% or ±5 views of target (whichever is greater)
                        close_range = max(5, int(self.target_views * 0.05))
                        if abs(view_count - self.target_views) <= close_range:
                            # Add a special marker for view counts close to target
                            title_element['style'] = "border: 2px solid red; background-color: yellow;"
                            # Add a comment to highlight to the LLM
                            comment = soup.new_string(f" (POTENTIAL MATCH: {view_count} views, target: {self.target_views}) ")
                            title_element.append(comment)
            except Exception as e:
                print(f"Error processing video renderer: {e}")
        
        return str(soup)
    
    def are_videos_present(self) -> bool:
        """Check if there are any videos present on the current page.
        
        Returns:
            Boolean indicating whether videos are present
        """
        try:
            # First check for "No results found" message
            try:
                no_results = self.driver.find_element(By.XPATH, 
                    "//*[contains(text(), 'No results found')]")
                if no_results:
                    return False
            except NoSuchElementException:
                pass  # Element not found means we might have results
            
            # Look for video elements
            video_elements = self.driver.find_elements(By.ID, "video-title")
            return len(video_elements) > 0
            
        except Exception as e:
            print(f"Error checking for videos: {e}")
            return False  # Assume no videos on error
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the scraping process.
        
        Returns:
            A dictionary containing the current state
        """
        # Check if videos are present on the page
        videos_present = self.are_videos_present()
        
        return {
            "current_url": self.driver.current_url,
            "current_search": self.current_search,
            "videos_checked": self.videos_checked[-20:] if len(self.videos_checked) > 20 else self.videos_checked,  # Limit to last 20 for readability
            "target_view_videos": self.target_view_videos,
            "last_action": self.last_action,
            "videos_present": videos_present
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
        
        prompt = PROMPT_TEMPLATE.format(
            html_snippet=html_snippet,
            current_url=state["current_url"],
            current_search=state["current_search"],
            videos_checked=", ".join([f'"{v["title"]}"' for v in state["videos_checked"]]) if state["videos_checked"] else "None",
            target_view_videos=", ".join([f'"{v["title"]}"' for v in state["target_view_videos"]]) if state["target_view_videos"] else "None",
            last_action=state["last_action"],
            target_views=self.target_views,
            videos_present=state["videos_present"],
            extra_rules=extra_rules_text
        )
        
        # Save prompt for debugging
        with open(os.path.join(self.debug_dir, f"prompt_{len(self.conversation_history)}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Call Ollama API
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
        # Default values with better initial search
        parsed = {
            "action": "SEARCH_VIDEOS",  # Safe default
            "reason": "Default action",
            "details": "uploaded within hour"  # Better default search term
        }
        
        # Extract sections from the response
        for line in response.split('\n'):
            if line.startswith("ACTION:"):
                parsed["action"] = line.replace("ACTION:", "").strip()
            elif line.startswith("REASON:"):
                parsed["reason"] = line.replace("REASON:", "").strip()
            elif line.startswith("DETAILS:"):
                # Extract and clean the details field
                details = line.replace("DETAILS:", "").strip()
                
                # For SEARCH_VIDEOS action, ensure we're only extracting the search query
                if parsed["action"] == "SEARCH_VIDEOS":
                    # First check for content in quotes, which is often the actual search query
                    quoted_content = re.findall(r'"([^"]*)"', details)
                    if quoted_content:
                        details = quoted_content[0]
                    else:
                        # If no quotes, try to extract just the search term

                        # Common phrases that indicate explanatory text
                        explanation_patterns = [
                            "Search for", "Look for", "Try searching for", "We should search for",
                            "I recommend", "This will", "Let's search", "This query", "This search",
                            "to find", "to target", "to focus", "to prioritize", "to help",
                            "which will", "that will", "as this", "because", "since", "so that",
                            "in order to", "this combines", "this targets", "this focuses"
                        ]
                        
                        # Try to remove explanatory text
                        for pattern in explanation_patterns:
                            if pattern.lower() in details.lower():
                                parts = details.lower().split(pattern.lower(), 1)
                                if len(parts) > 1:
                                    if parts[0].strip() == "":
                                        # Pattern was at the beginning, keep what comes after
                                        details = parts[1].strip()
                                    else:
                                        # Pattern was in the middle, keep what comes before
                                        details = parts[0].strip()
                        
                        # Remove any trailing explanatory text after certain punctuation
                        details = re.split(r'[.,:;]', details)[0].strip()
                        
                        # Remove common end phrases
                        end_phrases = [
                            "to find", "to help", "to increase", "to target", "to focus",
                            "for better", "for finding", "which should", "that should"
                        ]
                        for phrase in end_phrases:
                            if f" {phrase}" in details.lower():
                                details = details.lower().split(f" {phrase}")[0].strip()
                
                # Final cleanup: remove any remaining quotes and limit length
                details = details.strip('"\'')
                
                # Limit search query to a reasonable length (50 chars max)
                if parsed["action"] == "SEARCH_VIDEOS" and len(details) > 50:
                    details = details[:50]
                
                parsed["details"] = details
        
        return parsed
    
    def execute_action(self, action_data: Dict[str, str]) -> bool:
        """Execute the action suggested by the LLM.
        
        Args:
            action_data: The parsed LLM response containing action details
            
        Returns:
            Boolean indicating success or failure
        """
        action = action_data["action"]
        print(f"\n=== Executing Action: {action} ===")
        print(f"Reason: {action_data['reason']}")
        
        try:
            if action == "SEARCH_VIDEOS":
                return self.search_videos(action_data["details"])
            
            elif action == "SCROLL_DOWN":
                success = self.scroll_down()
                if not success and self.last_action == "REFINE_SEARCH":
                    # If scrolling failed due to no results, force a new search
                    print("No results found. Attempting a new search...")
                    alternative_searches = [
                        "uploaded within hour",
                        "uploaded today sort by upload date",
                        "first upload new channel",
                        "test video uploaded minutes ago",
                        "livestream starting soon",
                        "0 views challenge uploaded today",
                        "my first youtube video uploaded this week sort by upload date",
                        "niche tutorial uploaded minutes",
                        "testing camera uploaded within hour"
                    ]
                    new_search = random.choice(alternative_searches)
                    print(f"Trying alternative search: {new_search}")
                    return self.search_videos(new_search)
                return success
            
            elif action == "OPEN_VIDEO":
                return self.open_video(action_data["details"])
            
            elif action == "CHECK_RECOMMENDATIONS":
                return self.check_recommendations()
            
            elif action == "BACK_TO_RESULTS":
                return self.back_to_results()
            
            elif action == "REFINE_SEARCH":
                return self.search_videos(action_data["details"])
            
            elif action == "COMPLETE_SEARCH":
                print("Search process complete as suggested by LLM.")
                self.last_action = "COMPLETE_SEARCH"
                return False  # Signal to stop the scraping loop
            
            else:
                print(f"Unknown action: {action}")
                return False
        
        except Exception as e:
            print(f"Error executing action: {e}")
            self.take_screenshot(f"error_{action.lower()}.png")
            return False
    
    def search_videos(self, query: str) -> bool:
        """Search for videos using the specified query.
        
        Args:
            query: The search query to use
            
        Returns:
            Boolean indicating success
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Navigate to YouTube if not already there
                if not self.driver.current_url.startswith("https://www.youtube.com"):
                    self.driver.get(self.base_url)
                    time.sleep(2)  # Wait for page to load
                
                # Find and clear the search box
                search_box = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "search_query"))
                )
                search_box.clear()
                
                # Enter the search query and submit
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                
                # Wait for search results to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "contents"))
                )
                
                # Update state
                self.current_search = query
                self.last_action = "SEARCH_VIDEOS"
                
                # Take a screenshot
                self.take_screenshot(f"search_{query.replace(' ', '_')[:30]}.png")
                
                print(f"Searched for: {query}")
                return True
                
            except (ConnectionAbortedError, ConnectionResetError, ConnectionError) as e:
                retry_count += 1
                print(f"Network error (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    # Wait before retrying with exponential backoff
                    sleep_time = 2 ** retry_count
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    
                    # Try to refresh the driver connection
                    try:
                        self.driver.refresh()
                        time.sleep(2)
                    except:
                        pass
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return False
                    
            except Exception as e:
                print(f"Error searching for videos: {e}")
                return False
        
        return False
    
    def scroll_down(self, scroll_amount: int = 3) -> bool:
        """Scroll down to load more videos.
        
        Args:
            scroll_amount: Number of times to scroll
            
        Returns:
            Boolean indicating success
        """
        try:
            # First check if there are no results
            try:
                no_results = self.driver.find_element(By.XPATH, 
                    "//*[contains(text(), 'No results found')]")
                if no_results:
                    print("No results found on the page. Suggesting search refinement.")
                    self.last_action = "REFINE_SEARCH"
                    return False
            except NoSuchElementException:
                pass  # Element not found means we have results
            
            # If we have results, proceed with scrolling
            for _ in range(scroll_amount):
                self.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(1)  # Wait for content to load
            
            self.last_action = "SCROLL_DOWN"
            print(f"Scrolled down {scroll_amount} times to load more videos")
            
            # Take a screenshot after scrolling
            self.take_screenshot("after_scroll.png")
            
            return True
            
        except Exception as e:
            print(f"Error scrolling down: {e}")
            return False
    
    def open_video(self, video_identifier: str) -> bool:
        """Open a specific video from the current page.
        
        Args:
            video_identifier: Description of which video to open (position or title keywords)
            
        Returns:
            Boolean indicating success
        """
        try:
            video_index = None
            
            # Check if this is a request for the video with view count closest to target
            if video_identifier.lower() in ["lowest views", "lowest view count", "least views", "fewest views", 
                                         "target views", "closest views", "best match"]:
                # Try to find the video with view count closest to target
                best_match_info = self.find_video_with_closest_views()
                if best_match_info:
                    video_identifier, view_count = best_match_info
                    print(f"Found video with view count closest to target: {video_identifier} ({view_count} views)")
            
            # Different strategies for identifying the video
            if video_identifier.lower().startswith(("first", "1st")):
                video_index = 0
            elif video_identifier.lower().startswith(("second", "2nd")):
                video_index = 1
            elif video_identifier.lower().startswith(("third", "3rd")):
                video_index = 2
            elif video_identifier.lower().startswith(("fourth", "4th")):
                video_index = 3
            elif video_identifier.lower().startswith(("fifth", "5th")):
                video_index = 4
            elif video_identifier.isdigit():
                video_index = int(video_identifier) - 1
            else:
                # Try to find by title keywords
                video_elements = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.ID, "video-title"))
                )
                
                for i, video in enumerate(video_elements):
                    title = video.get_attribute("title")
                    if title and all(keyword.lower() in title.lower() for keyword in video_identifier.split()):
                        video_index = i
                        break
                
                if video_index is None:
                    print(f"Could not find video matching: {video_identifier}")
                    # Default to the first video if none found matching the description
                    video_index = 0
            
            # Get video elements and click on the selected one
            video_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.ID, "video-title"))
            )
            
            if video_index >= len(video_elements):
                video_index = 0  # Default to first if index is out of range
            
            # Get the title before clicking
            video_title = video_elements[video_index].get_attribute("title")
            
            # Try to click the video
            try:
                video_elements[video_index].click()
            except ElementClickInterceptedException:
                # If direct click fails, try with JavaScript
                self.driver.execute_script("arguments[0].click();", video_elements[video_index])
            
            # Wait for video page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "movie_player"))
            )
            
            # Take a screenshot of the video page
            self.take_screenshot(f"video_{video_title[:30].replace(' ', '_')}.png")
            
            # Check the view count
            views = self.get_view_count()
            
            # Record this video in our checked list
            video_data = {
                "title": video_title,
                "url": self.driver.current_url,
                "views": views,
                "timestamp": datetime.now().isoformat()
            }
            
            self.videos_checked.append(video_data)
            
            # If view count matches our target, add to our target videos list
            if views == self.target_views:
                self.target_view_videos.append(video_data)
                print(f"SUCCESS! Found a video with exactly {self.target_views} views: {video_title}")
            else:
                print(f"Opened video: {video_title} (Views: {views}, Target: {self.target_views})")
            
            self.last_action = "OPEN_VIDEO"
            return True
            
        except Exception as e:
            print(f"Error opening video: {e}")
            return False
    
    def get_view_count(self) -> int:
        """Extract the view count from the current video page.
        
        Returns:
            Number of views, or 0 if unable to determine
        """
        try:
            # Wait for view count to be available
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span.view-count"))
            )
            
            # Get the view count text
            view_count_element = self.driver.find_element(By.CSS_SELECTOR, "span.view-count")
            view_count_text = view_count_element.text
            
            # Extract the number from text like "123,456 views"
            view_count = 0
            if view_count_text:
                # Remove commas and non-numeric characters
                numeric_text = re.sub(r'[^0-9]', '', view_count_text)
                if numeric_text:
                    view_count = int(numeric_text)
            
            return view_count
            
        except TimeoutException:
            # If view count element doesn't appear, might be a brand new video with no views
            try:
                # Look for alternative view count displays
                info_text = self.driver.find_element(By.ID, "info-text").text
                if "No views" in info_text or "0 views" in info_text:
                    return 0
            except:
                pass
            
            print("Could not determine view count, assuming it's new with 0 views")
            return 0
            
        except Exception as e:
            print(f"Error getting view count: {e}")
            return 999  # Default to non-zero to avoid false positives
    
    def check_recommendations(self) -> bool:
        """Check video recommendations on the current video page.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Make sure we're on a video page
            if "youtube.com/watch" not in self.driver.current_url:
                print("Not on a video page, cannot check recommendations")
                return False
            
            # Scroll down to make sure recommendations are loaded
            self.driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2)
            
            # Look for recommendation elements
            recommendations = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#related #dismissible"))
            )
            
            print(f"Found {len(recommendations)} recommended videos")
            
            # Take a screenshot of recommendations
            self.take_screenshot("recommendations.png")
            
            self.last_action = "CHECK_RECOMMENDATIONS"
            return True
            
        except Exception as e:
            print(f"Error checking recommendations: {e}")
            return False
    
    def back_to_results(self) -> bool:
        """Go back to search results page.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.driver.back()
            
            # Wait for search results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "contents"))
            )
            
            print("Navigated back to search results")
            self.last_action = "BACK_TO_RESULTS"
            return True
            
        except Exception as e:
            print(f"Error going back to results: {e}")
            return False
    
    def take_screenshot(self, filename: str) -> None:
        """Take a screenshot and save it with a custom filename.
        
        Args:
            filename: The name of the screenshot file to save
        """
        try:
            # Take a screenshot and save it to our directory
            screenshot_path = os.path.join(self.screenshots_dir, filename)
            self.driver.save_screenshot(screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def save_data(self, results_file: str = None) -> None:
        """Save the collected data to a JSON file.
        
        Args:
            results_file: Filename for results data
        """
        # Use default filename that reflects target view count
        if results_file is None:
            results_file = f"youtube_{self.target_views}_views_results.json"
            
        # Prepare data to save
        results = {
            "target_view_videos": self.target_view_videos,
            "videos_checked": self.videos_checked,
            "search_queries": [h.get("parsed_response", {}).get("details") 
                             for h in self.conversation_history 
                             if h.get("parsed_response", {}).get("action") in ("SEARCH_VIDEOS", "REFINE_SEARCH")],
            "target_views": self.target_views,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {results_file}")
        
        # Also save conversation history
        with open(os.path.join(self.debug_dir, "conversation_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def run_llm_guided_scraping(self, max_actions: int = 20, max_target_videos: int = 3) -> None:
        """Run the LLM-guided scraping process.
        
        Args:
            max_actions: Maximum number of actions to take
            max_target_videos: Stop after finding this many videos with target view count
        """
        print(f"Starting LLM-guided YouTube scraping using model: {self.ollama_model}")
        print(f"Target view count: {self.target_views}")
        
        # Add special rules to guide the LLM
        self.extra_rules.append("IMPORTANT: Only use SEARCH_VIDEOS when there are no videos visible on the page to interact with")
        self.extra_rules.append("When you see <!-- VIEW_COUNT: X --> or (POTENTIAL MATCH) in the HTML, prioritize opening those videos immediately")
        self.extra_rules.append("Always use 'closest views' as the DETAILS when suggesting OPEN_VIDEO to target videos with view counts nearest to our target")
        self.extra_rules.append("If you see 'No results found' or empty search results, only then should you suggest a new search query")
        
        # Initialize YouTube session with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Navigate to YouTube
                self.driver.get(self.base_url)
                time.sleep(2)  # Wait for page to load
                
                # Accept cookies if the dialog appears
                try:
                    accept_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Accept all']"))
                    )
                    accept_button.click()
                    time.sleep(1)
                except:
                    pass  # No cookie dialog or different structure
                
                # Take initial screenshot
                self.take_screenshot("initial_page.png")
                
                # Start with a basic search to get some results
                initial_searches = [
                    "uploaded today sort by upload date",
                    "first upload test video", 
                    "just uploaded",
                    "0 views",
                    "new channel first video"
                ]
                initial_search = random.choice(initial_searches)
                if not self.search_videos(initial_search):
                    if attempt < max_retries - 1:
                        print(f"Initial search failed, retrying... (attempt {attempt+1}/{max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print("Failed to initialize YouTube session after multiple attempts")
                        return
                
                # Successfully initialized
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error initializing YouTube session: {e}")
                    print(f"Retrying... (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to initialize YouTube session after {max_retries} attempts: {e}")
                    return
        
        action_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while action_count < max_actions:
            print(f"\n=== Action {action_count + 1}/{max_actions} ===")
            
            try:
                # Get guidance from LLM
                print("Consulting LLM for next action...")
                guidance = self.consult_llm()
                
                # Execute the suggested action
                print(f"LLM suggests: {guidance['action']}")
                success = self.execute_action(guidance)
                
                # Handle success/failure
                if success:
                    consecutive_failures = 0  # Reset failure counter on success
                    action_count += 1
                else:
                    consecutive_failures += 1
                    print(f"Action failed. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive failures. Trying to recover...")
                        
                        # Try recovery actions
                        try:
                            # Refresh page and try a new search
                            self.driver.refresh()
                            time.sleep(3)
                            
                            recovery_searches = [
                                "uploaded today",
                                "test video new channel",
                                "first video upload"
                            ]
                            recovery_search = random.choice(recovery_searches)
                            if self.search_videos(recovery_search):
                                print("Recovery successful!")
                                consecutive_failures = 0
                            else:
                                print("Recovery failed. Stopping scraping process.")
                                break
                                
                        except Exception as e:
                            print(f"Error during recovery: {e}")
                            break
                    
                    # Skip counting this as an action if it failed
                    continue
                
                # Add a small delay to be respectful to the server
                delay = random.uniform(1, 2)
                time.sleep(delay)
                
                # Check if we should stop due to LLM recommendation
                if guidance["action"] == "COMPLETE_SEARCH":
                    print("Stopping scraping process as recommended by LLM.")
                    break
                
                # Stop if we've found enough videos with the target view count
                if len(self.target_view_videos) >= max_target_videos:
                    print(f"Success! Found {len(self.target_view_videos)} videos with target view count ({self.target_views}). Stopping search.")
                    break
                    
            except Exception as e:
                print(f"Error during scraping process: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive errors. Stopping scraping process.")
                    break
                time.sleep(2)  # Pause before continuing
        
        # Scraping complete
        print("\n=== Scraping Complete ===")
        print(f"Executed {action_count} actions")
        print(f"Checked {len(self.videos_checked)} videos")
        print(f"Found {len(self.target_view_videos)} videos with target view count ({self.target_views})")
        
        # Save the data
        self.save_data()
        
        # Print videos found with target view count
        if self.target_view_videos:
            print(f"\n=== Videos with {self.target_views} Views Found ===")
            for i, video in enumerate(self.target_view_videos, 1):
                print(f"{i}. {video['title']}")
                print(f"   URL: {video['url']}")
                print(f"   Found at: {video['timestamp']}")
                print()
        else:
            print(f"\nNo videos with exactly {self.target_views} views were found.")
        
        # Clean up
        self.driver.quit()

    def find_video_with_closest_views(self) -> Optional[Tuple[str, int]]:
        """Find the video with view count closest to target on the current page.
        
        Returns:
            Tuple of (video_position_or_title, view_count) or None if no videos found
        """
        try:
            # Get the page source
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all video renderer elements
            videos = []
            
            # First try to get all video renderers
            video_renderers = soup.find_all('div', id=lambda x: x and 'video-renderer' in x)
            
            for i, renderer in enumerate(video_renderers):
                try:
                    # Get title
                    title_element = renderer.select_one('a#video-title')
                    title = title_element.text.strip() if title_element else f"Video {i+1}"
                    
                    # Try to find the view count
                    view_count_element = renderer.select_one('span.style-scope.ytd-video-meta-block:not(#metadata-line)')
                    
                    view_count = 999999  # Default high number
                    if view_count_element:
                        view_count_text = view_count_element.text.strip()
                        
                        # Parse the view count
                        if 'No views' in view_count_text or '0 views' in view_count_text:
                            view_count = 0
                        else:
                            # Parse the number from strings like "123 views" or "1.2K views"
                            numeric_part = re.search(r'([\d,\.]+)', view_count_text)
                            if numeric_part:
                                number_str = numeric_part.group(1).replace(',', '')
                                if 'K' in view_count_text:
                                    view_count = int(float(number_str) * 1000)
                                elif 'M' in view_count_text:
                                    view_count = int(float(number_str) * 1000000)
                                else:
                                    view_count = int(float(number_str))
                    
                    videos.append((i, title, view_count))
                    
                except Exception as e:
                    print(f"Error parsing video {i}: {e}")
            
            # If targeting 0 views, sort by lowest first
            if self.target_views == 0:
                videos.sort(key=lambda x: x[2])
            else:
                # Sort by how close the view count is to our target
                videos.sort(key=lambda x: abs(x[2] - self.target_views))
            
            # Return the position/title of the video with closest view count to target
            if videos:
                best_match = videos[0]
                position = best_match[0] + 1  # 1-indexed position
                title = best_match[1]
                view_count = best_match[2]
                
                print(f"Best match: '{title}' with {view_count} views (target: {self.target_views})")
                
                # If title is unique enough, use it; otherwise use position
                if any(v[1] == title and v[0] != best_match[0] for v in videos):
                    return (str(position), view_count)  # Use position if title isn't unique
                else:
                    return (title, view_count)
            
            return None
            
        except Exception as e:
            print(f"Error finding video with target views: {e}")
            return None

def main():
    """Run the LLM-guided YouTube zero views scraper."""
    parser = argparse.ArgumentParser(description='LLM-guided YouTube low-view video finder')
    parser.add_argument('rules', nargs='*', 
                        help='Additional rules to guide the LLM (optional, can provide multiple)')
    parser.add_argument('--model', type=str, default=None,
                        help='Ollama model to use (if not specified, the largest available model will be used)')
    parser.add_argument('--max-actions', type=int, default=20,
                        help='Maximum number of actions to take (default: 20)')
    parser.add_argument('--max-videos', type=int, default=3,
                        help='Stop after finding this many videos with target view count (default: 3)')
    parser.add_argument('--server', type=str, default='localhost',
                        help='IP address or hostname of the Ollama server (default: localhost)')
    parser.add_argument('--target-views', type=int, default=0,
                        help='Target number of views to find (default: 0)')
    args = parser.parse_args()
    
    scraper = LLMGuidedYouTubeScraper(
        ollama_model=args.model,
        extra_rules=args.rules,
        ollama_server=args.server,
        target_views=args.target_views
    )
    
    try:
        scraper.run_llm_guided_scraping(
            max_actions=args.max_actions,
            max_target_videos=args.max_videos
        )
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        # Save whatever data we've collected
        scraper.save_data()
        scraper.driver.quit()
    except Exception as e:
        print(f"\nError during scraping: {e}")
        # Try to save the data and clean up
        try:
            scraper.save_data()
            scraper.driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()
