"""
Configuration settings for the LLM-guided YouTube video finder.
"""

import os
from typing import Dict, Any, List

# Navigator settings
DEFAULT_MAX_ACTIONS = 500
DEFAULT_MAX_VIDEOS = 300
DEFAULT_LLM_PROVIDER = "hunyuan"  # "hunyuan" or "ollama"
DEFAULT_TARGET_VIEWS = 0

# Selenium settings
SELENIUM_TIMEOUT = 10
SCREENSHOTS_ENABLED = True
HEADLESS_MODE = False

# Prompt template
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
5. COMPLETE_SEARCH - End the search process (if we've found videos with {target_views} views)

Important guidelines:
- CRITICAL PRIORITY: Your primary strategy should be to OPEN_VIDEO followed by CHECK_RECOMMENDATIONS
- Only use SCROLL_DOWN 1-2 times before attempting to OPEN_VIDEO
- If on a search results page, OPEN_VIDEO should be your default action
- After opening a video, always CHECK_RECOMMENDATIONS rather than going back to search results
- If the current URL contains "watch" (meaning we're on a video page), you must use CHECK_RECOMMENDATIONS
- Use SEARCH_VIDEOS only when absolutely necessary (no results or after checking many recommendation chains)
- Look for and prioritize opening videos with view counts near our target of {target_views} views
- If you see any videos with view counts close to {target_views} in the snippet, immediately suggest opening that video
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


# LLM settings for different providers
LLM_CONFIG = {
    "hunyuan": {
        "model": "hunyuan-turbos-latest",
        "temperature": 0.7,
        "max_tokens": 1024,
        "enable_enhancement": True,
    },
    "ollama": {
        "model": "llama3.2",  # Will be overridden by available model selection
        "temperature": 0.7
    }
} 
