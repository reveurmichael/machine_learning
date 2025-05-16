"""
Selenium driver module for YouTube video navigation and interaction.
"""

import os
import re
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementClickInterceptedException, 
    StaleElementReferenceException
)
from bs4 import BeautifulSoup

from llm_selenium_agent import BaseSeleniumChrome
from config import SELENIUM_TIMEOUT, SCREENSHOTS_ENABLED, HEADLESS_MODE

class SeleniumDriver(BaseSeleniumChrome):
    """Selenium driver for YouTube video navigation and interaction."""
    
    def __init__(self, headless: bool = HEADLESS_MODE, target_views: int = 0):
        """Initialize the Selenium driver.
        
        Args:
            headless: Whether to run the browser in headless mode
            target_views: Target number of views to find
        """
        # Initialize the parent class
        super().__init__()
        
        self.base_url = "https://www.youtube.com/"
        self.url = self.base_url  # Required by BaseSeleniumChrome
        self.headless_mode = headless  # Use the attribute name expected by BaseSeleniumChrome
        self.target_views = target_views
        self.logged_in = False  # YouTube doesn't require login, but keep for consistency
        
        # Override site_name to help with directory naming
        self.site_name = "YouTubeNavigator"
        
        # State tracking
        self.current_search = ""
        self.videos_checked = []
        self.target_view_videos = []
        
        # Create debug directory with timestamped name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.debug_dir = f"youtube_debug_{timestamp}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create screenshots directory within our debug directory
        if SCREENSHOTS_ENABLED:
            self.screenshots_dir = os.path.join(self.debug_dir, "screenshots")
            os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def login(self):
        """Override the login method from BaseSeleniumChrome.
        Required by BaseSeleniumChrome but we don't need login for YouTube.
        """
        # YouTube doesn't require login for viewing videos
        return True
    
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
    
    def prepare_environment(self):
        """Prepare the environment for YouTube navigation."""
        try:
            # Call the parent class's prepare_environment method
            super().prepare_environment()
            
            # Navigate to YouTube
            self.navigate_to_url()
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
            
            return True
        except Exception as e:
            print(f"Error preparing environment: {e}")
            return False
    
    def get_page_html_snippet(self, max_length: int = 50000) -> str:
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
        """Parse the HTML to extract and highlight videos with specific view counts.
        
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
                                # Handle cases like 1.2K = 1,200 or 22K = 22,000
                                view_count = int(float(number_str) * 1000)
                            elif 'M' in view_count_text:
                                # Handle cases like 1.2M = 1,200,000 or 22M = 22,000,000
                                view_count = int(float(number_str) * 1000000)
                            elif 'B' in view_count_text:
                                # Handle billion view counts
                                view_count = int(float(number_str) * 1000000000)
                            else:
                                view_count = int(float(number_str))
                    
                    # Get video title and add view count as a comment
                    title_element = renderer.select_one('a#video-title')
                    if title_element and view_count is not None:
                        title_element.string = f"{title_element.text.strip()} <!-- VIEW_COUNT: {view_count} -->"
                        
                        # Add visual indicator for videos with view count close to target
                        # Use adaptive range based on target view count
                        if self.target_views < 100:
                            # For very low view counts, use a smaller absolute range
                            close_range = 5
                        elif self.target_views < 1000:
                            # For double-digit counts, use 10% range
                            close_range = max(5, int(self.target_views * 0.1))
                        elif self.target_views < 10000:
                            # For triple-digit counts, use 5% range
                            close_range = max(10, int(self.target_views * 0.05))
                        else:
                            # For larger counts, use 2% range
                            close_range = max(20, int(self.target_views * 0.02))
                            
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
    
    def take_screenshot(self, filename: str) -> None:
        """Take a screenshot.
        
        Args:
            filename: Name of the screenshot file
        """
        if not SCREENSHOTS_ENABLED:
            return
            
        try:
            # Try to use the screenshot service if it exists
            if hasattr(self, 'screenshot_service') and self.screenshot_service:
                self.screenshot_service.capture_screenshot(self.screenshots_dir, filename)
                print(f"Saved screenshot using service to {os.path.join(self.screenshots_dir, filename)}")
            else:
                # Fall back to direct screenshot
                screenshot_path = os.path.join(self.screenshots_dir, filename)
                self.driver.save_screenshot(screenshot_path)
                print(f"Saved screenshot to {screenshot_path}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
    
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
                search_box = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.NAME, "search_query"))
                )
                search_box.clear()
                
                # Enter the search query and submit
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                
                # Wait for search results to load
                WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.ID, "contents"))
                )
                
                # Update state
                self.current_search = query
                
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
                    print("No results found on the page.")
                    return False
            except NoSuchElementException:
                pass  # Element not found means we have results
            
            # If we have results, proceed with scrolling
            for _ in range(scroll_amount):
                self.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(1)  # Wait for content to load
            
            print(f"Scrolled down {scroll_amount} times to load more videos")
            
            # Take a screenshot after scrolling
            self.take_screenshot("after_scroll.png")
            
            return True
            
        except Exception as e:
            print(f"Error scrolling down: {e}")
            return False
    
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
                    
                    view_count = None  # Default to None to identify unprocessed videos
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
                                elif 'B' in view_count_text:
                                    view_count = int(float(number_str) * 1000000000)
                                else:
                                    view_count = int(float(number_str))
                    
                    # Calculate upload recency score (0-10) based on upload time text if available
                    recency_score = 0
                    upload_time_element = renderer.select_one('span.style-scope.ytd-video-meta-block')
                    if upload_time_element:
                        upload_text = upload_time_element.text.lower()
                        if 'minute' in upload_text or 'second' in upload_text:
                            recency_score = 10
                        elif 'hour' in upload_text and 'ago' in upload_text:
                            # Extract the number of hours
                            hours_match = re.search(r'(\d+)\s+hour', upload_text)
                            if hours_match and int(hours_match.group(1)) <= 3:
                                recency_score = 8
                            else:
                                recency_score = 6
                        elif 'day' in upload_text and 'ago' in upload_text:
                            days_match = re.search(r'(\d+)\s+day', upload_text)
                            if days_match and int(days_match.group(1)) <= 1:
                                recency_score = 5
                            else:
                                recency_score = 3
                        elif 'today' in upload_text:
                            recency_score = 7
                        elif 'yesterday' in upload_text:
                            recency_score = 4
                    
                    # Only include videos where we could parse a view count
                    if view_count is not None:
                        # Calculate a combined score based on closeness to target views and recency
                        if self.target_views == 0:
                            # For 0 target views, prioritize absolute lowest view counts
                            closeness_score = 10 if view_count == 0 else (10 / (view_count + 1))
                        else:
                            # For non-zero targets, calculate relative closeness
                            # Lower difference = higher score (max 10)
                            view_difference = abs(view_count - self.target_views)
                            # Scale the difference relative to the target
                            relative_difference = view_difference / max(1, self.target_views)
                            closeness_score = 10 * (1 / (1 + relative_difference))
                        
                        # Combined score: 70% closeness to target, 30% recency
                        combined_score = (closeness_score * 0.7) + (recency_score * 0.3)
                        
                        videos.append({
                            "position": i+1,
                            "title": title,
                            "view_count": view_count,
                            "view_difference": abs(view_count - self.target_views),
                            "recency_score": recency_score,
                            "closeness_score": closeness_score,
                            "combined_score": combined_score
                        })
                    
                except Exception as e:
                    print(f"Error parsing video {i}: {e}")
            
            # Log the video data for debugging
            if videos:
                print(f"Found {len(videos)} videos with parseable view counts")
                for video in sorted(videos, key=lambda x: x["combined_score"], reverse=True)[:3]:
                    print(f"Title: {video['title'][:30]}... | Views: {video['view_count']} | " 
                          f"Diff: {video['view_difference']} | Score: {video['combined_score']:.2f}")
            else:
                print("No videos with parseable view counts found on this page")
            
            # Sort by combined score (descending)
            videos.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Return the best match
            if videos:
                best_match = videos[0]
                position = best_match["position"]
                title = best_match["title"]
                view_count = best_match["view_count"]
                
                print(f"Best match: '{title}' with {view_count} views (target: {self.target_views})")
                
                # If title is unique enough, use it; otherwise use position
                title_counts = {}
                for v in videos:
                    if v["title"] in title_counts:
                        title_counts[v["title"]] += 1
                    else:
                        title_counts[v["title"]] = 1
                
                if title_counts.get(title, 0) > 1:
                    # Title isn't unique, use position
                    return (str(position), view_count)
                else:
                    # Title is unique, use it
                    return (title, view_count)
            
            return None
            
        except Exception as e:
            print(f"Error finding video with target views: {e}")
            return None
    
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
                video_elements = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
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
            video_elements = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
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
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
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
                if 'No views' in view_count_text or '0 views' in view_count_text:
                    view_count = 0
                else:
                    # Parse the number from strings like "123,456 views" or "1.2K views"
                    numeric_part = re.search(r'([\d,\.]+)', view_count_text)
                    if numeric_part:
                        number_str = numeric_part.group(1).replace(',', '')
                        if 'K' in view_count_text:
                            # Handle cases like 1.2K = 1,200 or 22K = 22,000
                            view_count = int(float(number_str) * 1000)
                        elif 'M' in view_count_text:
                            # Handle cases like 1.2M = 1,200,000 or 22M = 22,000,000
                            view_count = int(float(number_str) * 1000000)
                        elif 'B' in view_count_text:
                            # Handle billion view counts
                            view_count = int(float(number_str) * 1000000000)
                        else:
                            view_count = int(float(number_str))
            
            print(f"Extracted view count: {view_count} from text: '{view_count_text}'")
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
            try:
                recommendations = WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#related #dismissible"))
                )
                
                print(f"Found {len(recommendations)} recommended videos")
                
                # Analyze recommendations for view counts
                recommendation_data = []
                
                for i, rec in enumerate(recommendations[:20]):  # Limit to first 20 recommendations
                    try:
                        # Extract title
                        title_elem = rec.find_element(By.ID, "video-title")
                        title = title_elem.get_attribute("title") or title_elem.text
                        
                        # Extract view count
                        metadata_elements = rec.find_elements(By.CSS_SELECTOR, "#metadata-line span")
                        view_count = None
                        
                        for elem in metadata_elements:
                            text = elem.text
                            if "view" in text.lower():
                                # Parse the view count
                                if 'No views' in text or '0 views' in text:
                                    view_count = 0
                                else:
                                    # Parse the number from strings like "123 views" or "1.2K views"
                                    numeric_part = re.search(r'([\d,\.]+)', text)
                                    if numeric_part:
                                        number_str = numeric_part.group(1).replace(',', '')
                                        if 'K' in text:
                                            view_count = int(float(number_str) * 1000)
                                        elif 'M' in text:
                                            view_count = int(float(number_str) * 1000000)
                                        elif 'B' in text:
                                            view_count = int(float(number_str) * 1000000000)
                                        else:
                                            view_count = int(float(number_str))
                        
                        # Calculate age/recency score if available
                        recency_score = 0
                        for elem in metadata_elements:
                            text = elem.text.lower()
                            if any(time_unit in text for time_unit in ["second", "minute", "hour", "day", "week", "month", "year"]):
                                if 'minute' in text or 'second' in text:
                                    recency_score = 10
                                elif 'hour' in text:
                                    hours_match = re.search(r'(\d+)\s+hour', text)
                                    if hours_match and int(hours_match.group(1)) <= 3:
                                        recency_score = 8
                                    else:
                                        recency_score = 6
                                elif 'day' in text:
                                    days_match = re.search(r'(\d+)\s+day', text)
                                    if days_match and int(days_match.group(1)) <= 1:
                                        recency_score = 5
                                    else:
                                        recency_score = 3
                                elif 'week' in text:
                                    recency_score = 2
                                else:
                                    recency_score = 1
                        
                        # Include if we could parse a view count
                        if view_count is not None:
                            # Calculate a score based on closeness to target views and recency
                            if self.target_views == 0:
                                # For 0 target views, prioritize absolute lowest view counts
                                closeness_score = 10 if view_count == 0 else (10 / (view_count + 1))
                            else:
                                # For non-zero targets, calculate relative closeness
                                view_difference = abs(view_count - self.target_views)
                                relative_difference = view_difference / max(1, self.target_views)
                                closeness_score = 10 * (1 / (1 + relative_difference))
                            
                            # Combined score: 80% closeness to target, 20% recency
                            combined_score = (closeness_score * 0.8) + (recency_score * 0.2)
                            
                            href = title_elem.get_attribute("href")
                            
                            recommendation_data.append({
                                "position": i+1,
                                "title": title,
                                "view_count": view_count,
                                "view_difference": abs(view_count - self.target_views),
                                "recency_score": recency_score,
                                "closeness_score": closeness_score,
                                "combined_score": combined_score,
                                "url": href
                            })
                    except Exception as e:
                        print(f"Error parsing recommendation {i}: {e}")
                
                # Sort recommendations by combined score
                recommendation_data.sort(key=lambda x: x["combined_score"], reverse=True)
                
                # Log the top recommendations
                if recommendation_data:
                    print(f"\n=== Top Recommended Videos (Target Views: {self.target_views}) ===")
                    for i, rec in enumerate(recommendation_data[:5]):
                        print(f"{i+1}. '{rec['title'][:40]}...' | Views: {rec['view_count']} | "
                              f"Diff: {rec['view_difference']} | Score: {rec['combined_score']:.2f}")
                    
                    # If we found one very close to our target, click it
                    best_rec = recommendation_data[0]
                    
                    # Set threshold based on target views
                    if self.target_views < 100:
                        threshold = 10  # Within 10 views for very low targets
                    elif self.target_views < 1000:
                        threshold = 50  # Within 50 views for 3-digit targets
                    else:
                        threshold = int(self.target_views * 0.05)  # Within 5% for larger targets
                    
                    if best_rec["view_difference"] <= threshold:
                        print(f"Found a recommended video very close to target views: {best_rec['view_count']} (target: {self.target_views})")
                        print(f"Auto-clicking: {best_rec['title']}")
                        
                        try:
                            # Find the element again and click it
                            i = best_rec["position"] - 1
                            rec_elem = recommendations[i].find_element(By.ID, "video-title")
                            rec_elem.click()
                            
                            # Wait for video to load
                            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                                EC.presence_of_element_located((By.ID, "movie_player"))
                            )
                            
                            # Check the view count of the opened video
                            views = self.get_view_count()
                            
                            # Record this video in our checked list
                            video_data = {
                                "title": best_rec['title'],
                                "url": self.driver.current_url,
                                "views": views,
                                "timestamp": datetime.now().isoformat(),
                                "found_via": "recommendations"
                            }
                            
                            self.videos_checked.append(video_data)
                            
                            # If it matches our target exactly, add to target videos list
                            if views == self.target_views:
                                self.target_view_videos.append(video_data)
                                print(f"SUCCESS! Found a recommended video with exactly {self.target_views} views!")
                            
                            return True
                        except Exception as e:
                            print(f"Error auto-clicking recommendation: {e}")
                else:
                    print("No recommendations with parseable view counts")
                
            except Exception as e:
                print(f"Error analyzing recommendations: {e}")
            
            # Take a screenshot of recommendations
            self.take_screenshot("recommendations.png")
            
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
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.ID, "contents"))
            )
            
            print("Navigated back to search results")
            return True
            
        except Exception as e:
            print(f"Error going back to results: {e}")
            return False
    
    def terminate_webdriver(self) -> None:
        """Terminate the WebDriver session."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                print("WebDriver terminated successfully")
        except Exception as e:
            print(f"Error terminating WebDriver: {e}") 