"""
Utility module for text processing.
Handles formatting and processing of text responses from LLMs.
"""

import re
import json

def process_response_for_display(response):
    """Process an LLM response for display in the UI.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Processed version of the response for display, with code blocks 
        and reasoning formatted for readability
    """
    processed = response
    
    # Extract and format code blocks
    processed = format_code_blocks(processed)
    
    # Limit to a reasonable display length
    max_display_length = 2000
    if len(processed) > max_display_length:
        processed = processed[:max_display_length] + "...[truncated]"
    
    return processed

def format_code_blocks(text):
    """Format code blocks in text for better display.
    
    Args:
        text: Text containing code blocks
        
    Returns:
        Text with formatted code blocks
    """
    # Look for code blocks with triple backticks
    code_block_pattern = r'```(?:json)?(.*?)```'
    
    # Function to process each match
    def process_code_block(match):
        code_content = match.group(1).strip()
        
        # Try to parse as JSON for better formatting
        try:
            # Check if it looks like JSON (starts with { or [)
            if code_content.strip().startswith('{') or code_content.strip().startswith('['):
                json_obj = json.loads(code_content)
                formatted_json = json.dumps(json_obj, indent=2)
                return f"```json\n{formatted_json}\n```"
        except:
            # If JSON parsing fails, return as is
            pass
            
        # Return original code block if no special formatting applied
        return f"```\n{code_content}\n```"
    
    # Replace code blocks with formatted versions
    formatted_text = re.sub(code_block_pattern, process_code_block, text, flags=re.DOTALL)
    
    return formatted_text 