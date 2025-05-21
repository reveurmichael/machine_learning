"""
Utility module for text processing in the Snake game.
Handles processing and formatting of text responses.
"""

def process_response_for_display(response):
    """Process the LLM response for display purposes.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Processed response text ready for display
    """
    try:
        processed = response
            
        # Limit to a reasonable length for display
        if len(processed) > 1500:
            processed = processed[:1500] + "...\n(response truncated)"
            
        return processed
    except Exception as e:
        print(f"Error processing response for display: {e}")
        return "Error processing response" 