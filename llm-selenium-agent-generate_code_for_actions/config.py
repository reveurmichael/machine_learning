"""
Configuration settings for the Selenium code generator.
Contains prompt templates and other configuration values.
"""

# Default timeout for Selenium operations
SELENIUM_TIMEOUT = 10

# Default temperature for LLM generation
DEFAULT_TEMPERATURE = 0.2

# Prompt template for generating Selenium code
ACTION_SUGGESTION_PROMPT = """
You are an AI assistant tasked with generating Selenium code for web automation on {current_url}

Current page HTML snippet:
```html
{html_snippet}
```

Current URL: {current_url}

{user_request}Based on the HTML snippet and current state, generate Python code to perform a useful action on this page.

Your response should be structured like this:
ACTION: [briefly describe the action(s) you'll implement]
REASON: [brief explanation of why this action is appropriate or useful]
GENERATED_CODE: [the Python code using Selenium to implement this action]

IMPORTANT GUIDELINES FOR SELENIUM CODE: 
- You MUST use the existing driver instance
- When handling elements, make sure they are visible and clickable before interacting
- Use explicit waits for reliability
- Add proper error handling with try/except blocks
- Include scrolling if elements might be outside the viewport
- For clicking elements, use JavaScript execution as backup if direct clicking fails
- Add print statements to show progress for debugging
- Add explicit pauses after major actions (time.sleep(1)) for visibility
- DO NOT include any markdown formatting like triple backticks (```) in your code
- DO NOT include any explanatory notes or comments after the code
- ONLY provide valid Python code that can be directly executed
- End your response immediately after the last line of code

Example response:
ACTION: Filter quotes by the "love" tag
REASON: I can see several tags on the page, and filtering by a specific tag will show related quotes.
GENERATED_CODE:
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
            return False
        
        # Scroll the tag into view
        driver.execute_script("arguments[0].scrollIntoView(true);", target_link)
        time.sleep(0.5)  # Small pause after scrolling
        
        # Click the tag, using JavaScript as a fallback
        try:
            target_link.click()
        except Exception as e:
            driver.execute_script("arguments[0].click();", target_link)
        
        # Wait for the page to load with filtered results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "quote"))
        )
        
        return True
        
    except Exception as e:
        return False

# Execute the function
filter_by_tag(driver)
"""
