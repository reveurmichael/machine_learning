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
    
    def get_page_html_snippet(self, max_length: int = 30000) -> str:
        """Get a simplified HTML snippet of the current page.
        
        Args:
            max_length: Maximum length of the HTML snippet to return
            
        Returns:
            A simplified HTML string with only relevant video information
        """
        # Get the page source
        html = self.driver.page_source
        
        # Parse with BeautifulSoup to extract only relevant information
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check if we're on a search results page or video page
        if "/watch" in self.driver.current_url:
            # We're on a video page - extract only relevant information about the current video
            # and video recommendations
            
            # Current video information
            current_video_info = {}
            try:
                # Get video title
                title_element = soup.select_one('h1.ytd-watch-metadata yt-formatted-string')
                current_video_info['title'] = title_element.text.strip() if title_element else "Unknown title"
                
                # Get view count
                view_count_element = soup.select_one('span.view-count')
                current_video_info['views'] = view_count_element.text.strip() if view_count_element else "Unknown views"
                
                # Get channel name
                channel_element = soup.select_one('ytd-channel-name yt-formatted-string a')
                current_video_info['channel'] = channel_element.text.strip() if channel_element else "Unknown channel"
            except Exception as e:
                print(f"Error extracting current video info: {e}")
            
            # Get recommended videos
            recommended_videos = []
            try:
                # Find recommendation sections
                recommendations = soup.select('ytd-compact-video-renderer')
                
                # Process up to 10 recommendations to keep the size reasonable
                for i, video in enumerate(recommendations[:10]):
                    video_data = {}
                    
                    # Title
                    title_element = video.select_one('span#video-title')
                    video_data['title'] = title_element.text.strip() if title_element else f"Video {i+1}"
                    
                    # View count - various ways YouTube might display this
                    view_element = video.select_one('span.ytd-video-meta-block')
                    if view_element:
                        video_data['views'] = view_element.text.strip()
                    else:
                        # Try alternative selector
                        view_element = video.select_one('.metadata-snippet-container span')
                        video_data['views'] = view_element.text.strip() if view_element else "Unknown views"
                    
                    recommended_videos.append(video_data)
            except Exception as e:
                print(f"Error extracting recommendations: {e}")
            
            # Create a simplified HTML structure with the extracted information
            simplified_html = f"""
            <div class="current-video">
                <h1>{current_video_info.get('title', 'Unknown title')}</h1>
                <div class="video-info">
                    <span class="view-count">{current_video_info.get('views', 'Unknown views')}</span>
                    <span class="channel">{current_video_info.get('channel', 'Unknown channel')}</span>
                </div>
            </div>
            <div class="recommended-videos">
                <h2>Recommended Videos</h2>
                <ul>
            """
            
            # Add recommended videos
            for i, video in enumerate(recommended_videos):
                simplified_html += f"""
                    <li class="recommendation" id="recommendation-{i+1}">
                        <div class="video-title">{video.get('title', f'Video {i+1}')}</div>
                        <div class="video-views">{video.get('views', 'Unknown views')}</div>
                    </li>
                """
            
            simplified_html += """
                </ul>
            </div>
            """
            
        else:
            # We're on a search results or home page - extract video results
            videos = []
            try:
                # Find video elements - this selector may need adjustment based on YouTube's structure
                video_elements = soup.select('ytd-video-renderer')
                
                # Process up to 20 videos to keep the response size reasonable
                for i, video in enumerate(video_elements[:20]):
                    video_data = {}
                    
                    # Title
                    title_element = video.select_one('a#video-title')
                    video_data['title'] = title_element.text.strip() if title_element else f"Video {i+1}"
                    
                    # URL
                    if title_element and title_element.has_attr('href'):
                        video_data['url'] = title_element['href']
                    
                    # View count - try different possible selectors
                    view_element = video.select_one('span.style-scope.ytd-video-meta-block')
                    if view_element:
                        video_data['views'] = view_element.text.strip()
                    else:
                        meta_block = video.select_one('#metadata-line')
                        if meta_block:
                            spans = meta_block.select('span')
                            if len(spans) >= 1:
                                video_data['views'] = spans[0].text.strip()
                    
                    # Channel name
                    channel_element = video.select_one('#channel-name yt-formatted-string')
                    video_data['channel'] = channel_element.text.strip() if channel_element else "Unknown channel"
                    
                    # Upload time
                    upload_element = video.select_one('#metadata-line span:nth-child(2)')
                    video_data['uploaded'] = upload_element.text.strip() if upload_element else "Unknown upload time"
                    
                    videos.append(video_data)
            except Exception as e:
                print(f"Error extracting search results: {e}")
            
            # Create a simplified HTML structure with the extracted information
            simplified_html = f"""
            <div class="search-results">
                <h2>Search Results</h2>
                <ul>
            """
            
            # Add videos
            for i, video in enumerate(videos):
                simplified_html += f"""
                    <li class="video-result" id="video-{i+1}">
                        <a href="{video.get('url', '#')}" class="video-title">{video.get('title', f'Video {i+1}')}</a>
                        <div class="video-info">
                            <span class="video-views">{video.get('views', 'Unknown views')}</span>
                            <span class="video-channel">{video.get('channel', 'Unknown channel')}</span>
                            <span class="video-uploaded">{video.get('uploaded', 'Unknown upload time')}</span>
                        </div>
                    </li>
                """
            
            simplified_html += """
                </ul>
            </div>
            """
        
        # Return the simplified HTML, constrained by max_length
        return simplified_html[:max_length]
    
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
                # Fix: Create full path and save directly instead of using service
                screenshot_path = os.path.join(self.screenshots_dir, filename)
                self.driver.save_screenshot(screenshot_path)
                print(f"Saved screenshot to {screenshot_path}")
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
            # Get the page source directly instead of parsing it again
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all video renderer elements
            videos = []
            video_renderers = []
            
            # First try to find the main video elements
            video_elements = soup.select('ytd-video-renderer')
            
            # If we don't find any with that selector, try other common selectors
            if not video_elements:
                # Try for grid-style videos
                video_elements = soup.select('ytd-rich-grid-media')
                
                # If still no results, try another pattern for video elements
                if not video_elements:
                    video_elements = soup.select('ytd-compact-video-renderer')
            
            # Now process all the video elements we found
            for i, video in enumerate(video_elements):
                try:
                    # Get title
                    title_element = video.select_one('a#video-title')
                    if not title_element:
                        title_element = video.select_one('#video-title')
                    
                    title = title_element.text.strip() if title_element else f"Video {i+1}"
                    
                    # Get the view count - try multiple possible selectors since YouTube has many formats
                    view_count_text = None
                    
                    # First try the metadata line
                    meta_block = video.select_one('#metadata-line')
                    if meta_block:
                        spans = meta_block.select('span')
                        if spans:
                            for span in spans:
                                if 'view' in span.text.lower():
                                    view_count_text = span.text.strip()
                                    break
                    
                    # If that didn't work, try other common selectors
                    if not view_count_text:
                        view_element = video.select_one('.ytd-video-meta-block')
                        if view_element:
                            view_count_text = view_element.text.strip()
                    
                    # If still no view count, try one more selector
                    if not view_count_text:
                        view_element = video.select_one('span:contains("views")')
                        if view_element:
                            view_count_text = view_element.text.strip()
                    
                    # Parse the view count
                    view_count = None
                    if view_count_text:
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
                    
                    # Skip if we couldn't determine the view count
                    if view_count is not None:
                        videos.append({
                            'position': i + 1,
                            'title': title,
                            'view_count': view_count
                        })
                        
                except Exception as e:
                    print(f"Error processing video {i+1}: {e}")
                    continue
            
            # If no videos were found or processed successfully, fallback to direct DOM access
            if not videos:
                print("No videos found using BeautifulSoup parsing, falling back to direct DOM access")
                
                # Try to get video elements directly from DOM
                video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer, ytd-rich-grid-media, ytd-compact-video-renderer')
                
                for i, video_element in enumerate(video_elements):
                    try:
                        # Get title
                        title_element = video_element.find_element(By.ID, "video-title")
                        title = title_element.text.strip() if title_element else f"Video {i+1}"
                        
                        # Try to find view count
                        try:
                            metadata = video_element.find_element(By.ID, "metadata-line")
                            spans = metadata.find_elements(By.TAG_NAME, "span")
                            if spans and len(spans) > 0:
                                view_count_text = spans[0].text
                            else:
                                view_count_text = None
                        except:
                            view_count_text = None
                        
                        # Parse the view count
                        view_count = None
                        if view_count_text:
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
                        
                        # Skip if we couldn't determine the view count
                        if view_count is not None:
                            videos.append({
                                'position': i + 1,
                                'title': title,
                                'view_count': view_count
                            })
                            
                    except Exception as e:
                        print(f"Error processing DOM video {i+1}: {e}")
                        continue
            
            # If we have videos with view counts, find the one closest to our target
            if videos:
                # Sort videos by how close they are to the target view count
                videos.sort(key=lambda x: abs(x['view_count'] - self.target_views))
                
                # Get the closest match
                closest_video = videos[0]
                
                # Log the closest video
                print(f"Found video closest to {self.target_views} views: '{closest_video['title']}' with {closest_video['view_count']} views")
                
                # Return the position and view count
                return (f"Video {closest_video['position']}", closest_video['view_count'])
            else:
                print("No videos with view counts found on this page")
                return None
                
        except Exception as e:
            print(f"Error finding video with closest views: {e}")
            return None
    
    def open_video(self, video_identifier: str) -> bool:
        """Open a specific video from the current page.
        
        Args:
            video_identifier: Description of which video to open (position, title keywords, or "closest views")
            
        Returns:
            Boolean indicating success
        """
        try:
            video_index = None
            video_elements = []
            
            # Check if this is a request for the video with view count closest to target
            if video_identifier.lower() in ["lowest views", "lowest view count", "least views", "fewest views", 
                                         "target views", "closest views", "best match"]:
                # Try to find the video with view count closest to target
                best_match_info = self.find_video_with_closest_views()
                if best_match_info:
                    video_identifier, view_count = best_match_info
                    print(f"Found video with view count closest to target: {video_identifier} ({view_count} views)")
            
            # Try multiple selectors to find video elements
            selectors = [
                "a#video-title",  # Standard search results
                "#video-title",    # Alternative format
                "ytd-compact-video-renderer a",  # Recommended videos
                "ytd-grid-video-renderer a"      # Grid layout videos
            ]
            
            # Try each selector until we find elements
            for selector in selectors:
                try:
                    video_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if video_elements and len(video_elements) > 0:
                        print(f"Found {len(video_elements)} videos using selector: {selector}")
                        break
                except Exception as e:
                    print(f"Error with selector {selector}: {e}")
            
            if not video_elements:
                print("Could not find any video elements using standard selectors")
                # Last resort - try any clickable link
                try:
                    video_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.yt-simple-endpoint")
                    print(f"Found {len(video_elements)} generic video links")
                except:
                    pass
            
            if not video_elements:
                print("Failed to find any videos to click")
                return False
            
            # Different strategies for identifying the video
            if video_identifier.lower().startswith("video "):
                # Handle "Video N" format from our find_video_with_closest_views function
                try:
                    num_part = video_identifier.lower().replace("video ", "").strip()
                    if num_part.isdigit():
                        video_index = int(num_part) - 1
                except:
                    pass
            elif video_identifier.lower().startswith(("first", "1st")):
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
                for i, video in enumerate(video_elements):
                    try:
                        title = video.get_attribute("title") or video.get_attribute("aria-label") or video.text
                        if title and all(keyword.lower() in title.lower() for keyword in video_identifier.split()):
                            video_index = i
                            break
                    except:
                        continue
            
            # If we still haven't found a specific video, default to the first one
            if video_index is None or video_index < 0:
                print(f"Could not identify specific video for: {video_identifier}, defaulting to first video")
                video_index = 0
            
            # Make sure we don't exceed the number of available videos
            if video_index >= len(video_elements):
                video_index = len(video_elements) - 1
            
            # Get the title before clicking (try multiple approaches)
            video_title = "Unknown Title"
            try:
                title_attrs = ["title", "aria-label", "text", "textContent", "innerHTML"]
                for attr in title_attrs:
                    try:
                        if attr in ["text", "textContent", "innerHTML"]:
                            video_title = getattr(video_elements[video_index], attr, "")
                        else:
                            video_title = video_elements[video_index].get_attribute(attr)
                        
                        if video_title and len(video_title) > 0:
                            # Clean up common patterns in aria-label that include duration
                            if " by " in video_title:
                                video_title = video_title.split(" by ")[0]
                            if " - Duration: " in video_title:
                                video_title = video_title.split(" - Duration: ")[0]
                            break
                    except:
                        continue
            except:
                video_title = f"Video {video_index + 1}"
            
            print(f"Attempting to click video: {video_title}")
            
            # Try to click the video with multiple strategies
            click_success = False
            click_strategies = [
                # Strategy 1: Regular click
                lambda elem: elem.click(),
                # Strategy 2: JavaScript click
                lambda elem: self.driver.execute_script("arguments[0].click();", elem),
                # Strategy 3: Navigate to href
                lambda elem: self.driver.get(elem.get_attribute("href")),
                # Strategy 4: Send enter key
                lambda elem: elem.send_keys(Keys.ENTER)
            ]
            
            for strategy_index, click_strategy in enumerate(click_strategies):
                try:
                    print(f"Trying click strategy {strategy_index + 1}...")
                    click_strategy(video_elements[video_index])
                    # Wait a moment to see if navigation happens
                    time.sleep(2)
                    # Check if we've navigated to a video page
                    if "/watch" in self.driver.current_url:
                        click_success = True
                        break
                except Exception as e:
                    print(f"Click strategy {strategy_index + 1} failed: {e}")
                    continue
            
            if not click_success:
                print("All click strategies failed to open video")
                return False
            
            # Wait for video page to load
            try:
                WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.ID, "movie_player"))
                )
            except:
                print("Warning: Could not detect movie player, but URL suggests we're on a video page")
            
            # Take a screenshot of the video page
            safe_title = ''.join(c if c.isalnum() or c == ' ' else '_' for c in video_title[:30]).strip()
            self.take_screenshot(f"video_{safe_title}.png")
            
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
            # Multiple attempts to find view count with different selectors
            view_count_text = None
            
            # Try approach 1: Standard view-count span
            try:
                # Wait for view count to be available
                view_count_element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "span.view-count"))
                )
                view_count_text = view_count_element.text
                print(f"Found view count using standard selector: {view_count_text}")
            except:
                pass
                
            # Try approach 2: Info text for new videos
            if not view_count_text:
                try:
                    info_element = self.driver.find_element(By.ID, "info-text")
                    info_text = info_element.text
                    if "views" in info_text.lower():
                        view_parts = info_text.split("\n")
                        for part in view_parts:
                            if "views" in part.lower():
                                view_count_text = part.strip()
                                print(f"Found view count in info text: {view_count_text}")
                                break
                except:
                    pass
            
            # Try approach 3: Count info in metadata
            if not view_count_text:
                try:
                    meta_elements = self.driver.find_elements(By.CSS_SELECTOR, "#metadata-line span")
                    for elem in meta_elements:
                        if "view" in elem.text.lower():
                            view_count_text = elem.text
                            print(f"Found view count in metadata: {view_count_text}")
                            break
                except:
                    pass
            
            # Try approach 4: View count in video description
            if not view_count_text:
                try:
                    # Parse with BeautifulSoup for more flexibility
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for elements containing view count text
                    view_elements = soup.find_all(string=lambda text: isinstance(text, str) and 'views' in text.lower())
                    
                    if view_elements:
                        # Get the first element that looks like a view count
                        for elem in view_elements:
                            elem_text = elem.strip()
                            if re.search(r'\d+\s+views', elem_text):
                                view_count_text = elem_text
                                print(f"Found view count with BeautifulSoup: {view_count_text}")
                                break
                except Exception as e:
                    print(f"BeautifulSoup view count extraction failed: {e}")
            
            # If we still don't have a view count, check the title attribute
            if not view_count_text:
                try:
                    title_element = self.driver.find_element(By.CSS_SELECTOR, ".title.ytd-video-primary-info-renderer")
                    title_attrs = [title_element.get_attribute("title"), title_element.get_attribute("aria-label")]
                    
                    for attr in title_attrs:
                        if attr and "views" in attr.lower():
                            # Extract the part with views
                            view_pattern = re.search(r'(\d[\d,\.]*\s+views)', attr)
                            if view_pattern:
                                view_count_text = view_pattern.group(1)
                                print(f"Found view count in title attributes: {view_count_text}")
                                break
                except:
                    pass
            
            # If we still don't have a view count, but it's a new video, it might have 0 views
            if not view_count_text:
                try:
                    # Check for "new" badge or recent upload indicators
                    new_indicators = ["seconds ago", "minutes ago", "minute ago", "hour ago", "hours ago", "today"]
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                    
                    for indicator in new_indicators:
                        if indicator in page_text:
                            print(f"Found '{indicator}' in page, assuming this could be a new video with 0 views")
                            view_count_text = "0 views"
                            break
                except:
                    pass
            
            # Parse the view count from text
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
            
            # Scroll down 1-3 times to ensure more recommendations load
            scroll_count = random.randint(1, 3)
            print(f"Scrolling down {scroll_count} times to load more recommendations...")
            for i in range(scroll_count):
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(1)  # Give time for recommendations to load
                print(f"Scroll {i+1}/{scroll_count} completed")
            
            # Take a screenshot
            self.take_screenshot("recommendations.png")
            
            # Use BeautifulSoup to extract recommendation info more reliably
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for recommended videos using multiple possible selectors
            recommended_videos = []
            
            # Try different selectors for recommendation areas
            selectors = [
                'ytd-compact-video-renderer',  # Standard recommendation format
                'ytd-watch-next-secondary-results-renderer ytd-compact-video-renderer',  # Specific location
                'ytd-shelf-renderer ytd-compact-video-renderer',  # Another possible location
                '.ytd-watch-next-secondary-results-renderer a#thumbnail'  # Thumbnail links
            ]
            
            for selector in selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        print(f"Found {len(elements)} recommended videos using selector: {selector}")
                        
                        # Process up to 10 recommendations
                        for i, video in enumerate(elements[:10]):
                            video_data = {}
                            
                            # Extract title
                            title_elem = video.select_one('#video-title') or video.select_one('span.title')
                            if title_elem:
                                video_data['title'] = title_elem.text.strip()
                            else:
                                video_data['title'] = f"Recommended Video {i+1}"
                            
                            # Extract view count if available
                            view_elem = None
                            for span in video.select('span'):
                                if span.text and 'views' in span.text.lower():
                                    view_elem = span
                                    break
                                    
                            if view_elem:
                                video_data['views'] = view_elem.text.strip()
                            else:
                                video_data['views'] = "Unknown views"
                            
                            # Add to our list
                            recommended_videos.append(video_data)
                        
                        # Don't try more selectors if we found videos
                        break
                except Exception as e:
                    print(f"Error finding recommendations with selector {selector}: {e}")
            
            # If we couldn't extract recommendations with BeautifulSoup, fall back to direct Selenium
            if not recommended_videos:
                print("Falling back to direct Selenium extraction for recommendations")
                
                # Try direct Selenium extraction
                try:
                    # Look for recommended video elements
                    video_elements = self.driver.find_elements(By.CSS_SELECTOR, "ytd-compact-video-renderer")
                    
                    for i, video in enumerate(video_elements[:10]):
                        video_data = {}
                        
                        try:
                            # Extract title
                            title_elem = video.find_element(By.ID, "video-title")
                            video_data['title'] = title_elem.text.strip()
                        except:
                            video_data['title'] = f"Recommended Video {i+1}"
                        
                        try:
                            # Extract view count
                            meta_line = video.find_element(By.ID, "metadata-line")
                            spans = meta_line.find_elements(By.TAG_NAME, "span")
                            for span in spans:
                                if 'views' in span.text.lower():
                                    video_data['views'] = span.text.strip()
                                    break
                        except:
                            video_data['views'] = "Unknown views"
                        
                        # Add to our list
                        recommended_videos.append(video_data)
                        
                except Exception as e:
                    print(f"Error with direct Selenium extraction: {e}")
            
            if recommended_videos:
                print(f"Found {len(recommended_videos)} recommended videos")
                
                # Look for potentially zero-view videos
                potential_zero_videos = []
                for video in recommended_videos:
                    views_text = video.get('views', '').lower()
                    title = video.get('title', '')
                    
                    # Check for likely low view count videos
                    if (
                        "no views" in views_text or 
                        "0 views" in views_text or 
                        "1 view" in views_text or 
                        "2 views" in views_text or
                        "few views" in views_text
                    ):
                        potential_zero_videos.append(video)
                        print(f"Potential zero-view video: {title} - {views_text}")
                
                # Log all recommendations
                print("\nAll recommendations:")
                for i, video in enumerate(recommended_videos):
                    print(f"{i+1}. {video.get('title', 'Untitled')} - {video.get('views', 'Unknown views')}")
                    
                # First try to click on a video with view count close to our target
                video_to_click = None
                
                # If we're looking for zero-view videos, prioritize those
                if self.target_views == 0 and potential_zero_videos:
                    print("\nAttempting to click on a potential zero-view video...")
                    video_to_click = potential_zero_videos[0]
                else:
                    # For non-zero targets, or if no potential zero-view videos, find the closest match
                    # or just pick a random recommendation
                    videos_with_counts = []
                    for video in recommended_videos:
                        views_text = video.get('views', '').lower()
                        if views_text != "unknown views":
                            # Try to parse the view count
                            try:
                                if 'no views' in views_text or '0 views' in views_text:
                                    view_count = 0
                                else:
                                    # Parse the number from strings like "123 views" or "1.2K views"
                                    numeric_part = re.search(r'([\d,\.]+)', views_text)
                                    if numeric_part:
                                        number_str = numeric_part.group(1).replace(',', '')
                                        if 'K' in views_text:
                                            view_count = int(float(number_str) * 1000)
                                        elif 'M' in views_text:
                                            view_count = int(float(number_str) * 1000000)
                                        elif 'B' in views_text:
                                            view_count = int(float(number_str) * 1000000000)
                                        else:
                                            view_count = int(float(number_str))
                                
                                videos_with_counts.append((video, view_count))
                            except:
                                pass
                    
                    if videos_with_counts:
                        # Sort by how close they are to our target and pick the best match
                        videos_with_counts.sort(key=lambda x: abs(x[1] - self.target_views))
                        video_to_click = videos_with_counts[0][0]
                        print(f"\nSelected recommendation with views closest to target: {video_to_click.get('title')} - {videos_with_counts[0][1]} views")
                    else:
                        # If we couldn't parse any view counts, just pick a random recommendation
                        video_to_click = random.choice(recommended_videos)
                        print(f"\nSelected random recommendation: {video_to_click.get('title')}")
                
                # Now try to click the selected video
                if video_to_click:
                    title = video_to_click.get('title', '')
                    print(f"Attempting to click recommended video: {title}")
                    
                    # Multiple approaches to find and click the video
                    click_success = False
                    
                    # Approach 1: Try to find by exact title match
                    try:
                        title_elements = self.driver.find_elements(By.XPATH, 
                            f"//a[@id='video-title' and contains(text(), '{title}')]")
                        
                        if title_elements and len(title_elements) > 0:
                            print(f"Found video by title: {title}")
                            
                            # Try multiple click strategies
                            for click_strategy in [
                                lambda e: e.click(),
                                lambda e: self.driver.execute_script("arguments[0].click();", e),
                                lambda e: e.send_keys(Keys.ENTER),
                                lambda e: self.driver.get(e.get_attribute("href"))
                            ]:
                                try:
                                    click_strategy(title_elements[0])
                                    # Wait for video to load
                                    time.sleep(2)
                                    if "/watch" in self.driver.current_url:
                                        click_success = True
                                        break
                                except Exception as e:
                                    print(f"Click strategy failed: {e}")
                                    continue
                    except Exception as e:
                        print(f"Error finding video by title: {e}")
                    
                    # Approach 2: If title match failed, try finding any video title
                    if not click_success:
                        try:
                            print("Trying to click any recommended video...")
                            video_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                                "ytd-compact-video-renderer a#video-title, ytd-compact-video-renderer a.yt-simple-endpoint")
                            
                            if video_elements and len(video_elements) > 0:
                                # Pick a random video (variety is good for discovery)
                                random_index = random.randint(0, min(5, len(video_elements)-1))
                                video_elem = video_elements[random_index]
                                random_title = video_elem.text or video_elem.get_attribute("title") or f"Video {random_index+1}"
                                print(f"Clicking random recommended video: {random_title}")
                                
                                # Try click strategies
                                for click_strategy in [
                                    lambda e: e.click(),
                                    lambda e: self.driver.execute_script("arguments[0].click();", e),
                                    lambda e: self.driver.get(e.get_attribute("href"))
                                ]:
                                    try:
                                        click_strategy(video_elem)
                                        # Wait for video to load
                                        time.sleep(2)
                                        if "/watch" in self.driver.current_url:
                                            click_success = True
                                            title = random_title  # Update title for record-keeping
                                            break
                                    except Exception as e:
                                        print(f"Click strategy failed: {e}")
                                        continue
                        except Exception as e:
                            print(f"Error with backup click approach: {e}")
                    
                    # If we successfully clicked a video, process it
                    if click_success:
                        print(f"Successfully clicked on recommended video: {title}")
                        
                        # Check the view count
                        views = self.get_view_count()
                        
                        # Record this video in our checked list
                        video_data = {
                            "title": title,
                            "url": self.driver.current_url,
                            "views": views,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        self.videos_checked.append(video_data)
                        
                        # If view count matches our target, add to our target videos list
                        if views == self.target_views:
                            self.target_view_videos.append(video_data)
                            print(f"SUCCESS! Found a recommendation with exactly {self.target_views} views: {title}")
                        else:
                            print(f"Opened recommendation: {title} (Views: {views}, Target: {self.target_views})")
                        
                        return True
                    else:
                        print("Failed to click any recommended videos after multiple attempts")
                        return False
                else:
                    print("No suitable recommendations to click")
                    return False
            else:
                print("No recommended videos found")
                return False
                
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