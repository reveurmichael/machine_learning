"""
Configuration settings for the LLM-guided web navigator.
"""

import os
from typing import Dict, Any, List

# Navigator settings
DEFAULT_MAX_ACTIONS = 50
DEFAULT_LLM_PROVIDER = "hunyuan"  # "hunyuan" or "ollama"

# Prompt template
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
- CURRENT ROUND: {current_round}/{max_rounds}

MOST IMPORTANT RULE (USER COMMAND) - YOU MUST PRIORITIZE THIS ABOVE ALL ELSE (RULE NO.0001):
{user_rule} . However, if you don't know what action to take, just go for NAVIGATE_NEXT_PAGE

Based on the HTML snippet above and the current state, AND PRIMARILY GUIDED BY THE USER COMMAND ABOVE, please suggest what actions to take next.
Choose from these possible actions:
1. NAVIGATE_NEXT_PAGE - Go to the next page if available
2. NAVIGATE_PREVIOUS_PAGE - Go back to the previous page if available
3. VISIT_AUTHOR_PAGE - Visit a specific author's page (YOU MUST specify a valid author name from the page)
4. FILTER_BY_TAG - Filter quotes by a specific tag (specify which tag)
5. LOGIN - Log in to the website using credentials
6. LOGOUT - Log out of the website

Important guidelines:
- The user command above MUST be your primary consideration, though RULE NO.0001 is by far the most most most important of all rules - follow it exactly
- If this is ROUND 1, your action should be NAVIGATE_NEXT_PAGE to begin exploring the site
- If the user asks you to login, YOUR NEXT ACTION MUST BE LOGIN
- If the user asks you to visit an author, YOUR NEXT ACTION MUST BE VISIT_AUTHOR_PAGE
- If the user asks you to filter by a tag, YOUR NEXT ACTION MUST BE FILTER_BY_TAG
- Only use LOGIN if we're not already logged in and we need to access restricted content
- Only use LOGOUT if we're currently logged in
- Only use VISIT_AUTHOR_PAGE after identifying specific authors from quotes (e.g., "Albert Einstein", "Jane Austen")
- When using VISIT_AUTHOR_PAGE, you MUST provide a real author name from the quotes on the page
- When using FILTER_BY_TAG, you MUST provide a real tag from the quotes on the page
- If you don't know what action to take, just go for NAVIGATE_NEXT_PAGE
{extra_rules}

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

ACTION: LOGIN
REASON: The user has requested that I log in to the website.
DETAILS: None
"""

# Login credentials
LOGIN_CREDENTIALS = {
    "username": "123456",
    "password": "12345678"
}

# Selenium settings
SELENIUM_TIMEOUT = 10
SCREENSHOTS_ENABLED = True

# LLM settings for different providers
LLM_CONFIG = {
    "hunyuan": {
        "model": "hunyuan-turbos-latest",
        "temperature": 0.7,
        "max_tokens": 1024,
        "enable_enhancement": True,
    },
    "ollama": {
        "temperature": 0.7
    }
} 
