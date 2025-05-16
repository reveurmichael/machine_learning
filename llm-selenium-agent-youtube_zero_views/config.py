"""
Configuration settings for the LLM-guided YouTube video finder.
"""

import os
from typing import Dict, Any, List

# Navigator settings
DEFAULT_MAX_ACTIONS = 500
DEFAULT_MAX_VIDEOS = 300
DEFAULT_LLM_PROVIDER = "yuanbao"  # "yuanbao" or "ollama"
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
5. COMPLETE_SEARCH - End the search process (if we've found videos with {target_views} views or exhausted options)

Important guidelines:
- ONLY use SEARCH_VIDEOS action when there are no videos visible on the current page to click on (videos_present = False).
- When videos are visible (videos_present = True), always prefer to OPEN_VIDEO or SCROLL_DOWN to find more videos rather than performing a new search.
- When on a search results page, immediately look for videos with view counts close to the target of {target_views} views.
- Use "closest views" or "best match" as the identifier to find videos with view counts nearest to {target_views}.
- Look for and prioritize opening videos with view counts near our target of {target_views}.
- If you see any videos with view counts close to {target_views} in the snippet, immediately suggest opening that video.
- If on a video page and no suitable videos are found, check recommendations.
- For search queries, use YouTube-specific search operators like "uploaded today", "uploaded within hour", or "uploaded minutes ago"
- Add "sort by upload date" to your search queries to prioritize newest content
- After checking about 10-15 videos with no success, consider suggesting a new search with SEARCH_VIDEOS
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

# Initial search queries optimized for finding low-view videos
INITIAL_SEARCH_QUERIES = [
    "first video uploaded today sort by upload date",
    "test video first upload", 
    "just uploaded minutes ago",
    "0 views challenge",
    "new channel first video no subscribers",
    "practice video testing microphone",
    "channel trailer unpopular",
    "tutorial niche topic uploaded today",
    "obscure hobby tutorial uploaded recently",
    "testing new camera uploaded minutes ago",
    "livestream starting soon test"
]

# Recovery search queries if the regular search fails
RECOVERY_SEARCH_QUERIES = [
    "uploaded today hour sort by upload date",
    "test video new channel no subscribers",
    "first upload help me grow",
    "school project uploaded recently",
    "practicing editing software tutorial",
    "unpopular topic explained",
    "small channel trailer"
]

# Alternative search queries
ALTERNATIVE_SEARCH_QUERIES = [
    "uploaded within hour sort by upload date",
    "first upload never shared",
    "test video uploaded minutes ago",
    "livestream setup test",
    "0 views challenge uploaded today",
    "my first youtube video uploaded today sort by upload date",
    "niche tutorial uploaded minutes ago",
    "testing camera mic quality uploaded today",
    "learning to edit videos first attempt",
    "uncommon language tutorial",
    "how to unpopular software tutorial",
    "obscure game walkthrough part 1",
    "no commentary gameplay unknown game",
    "unboxing rare item uploaded recently",
    "foreign language first video"
]

# LLM settings for different providers
LLM_CONFIG = {
    "yuanbao": {
        "model": "Llama3-8B",  # Default model
        "temperature": 0.7,
        "max_tokens": 1024,
    },
    "ollama": {
        "model": "llama3.2",  # Will be overridden by available model selection
        "temperature": 0.7
    }
} 