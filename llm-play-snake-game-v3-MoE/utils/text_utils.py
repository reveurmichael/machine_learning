"""Small text-processing helpers used by dashboards and logging.

The functions here intentionally avoid any heavy NLP libraries so they remain
cheap to import.  They focus on lightweight formatting tasks only.
"""

from __future__ import annotations

import json
import re
from typing import Final

__all__ = [
    "process_response_for_display",
    "format_code_blocks",
]

# Pre-compiled regex – reused for every call so we don't recompile thousands of
# times when streaming logs in a long experiment.

_CODE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?(.*?)```", re.DOTALL,
)

def process_response_for_display(response: str) -> str:
    """Process an LLM response for display in the UI.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Processed version of the response for display, with code blocks 
        and reasoning formatted for readability
    """    
    # Extract and format code blocks
    response = format_code_blocks(response)

    # Limit to a reasonable display length
    max_display_length: int = 2_000
    if len(response) > max_display_length:
        response = f"{response[:max_display_length]}...[truncated]"

    return response

def format_code_blocks(text: str) -> str:
    """Format code blocks in text for better display.
    
    Args:
        text: Text containing code blocks
        
    Returns:
        Text with formatted code blocks
    """
    # Helper applied to every regex match
    def _process(match: re.Match[str]) -> str:
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
    
    # Substitute using the pre-compiled regex for performance
    return _CODE_BLOCK_RE.sub(_process, text)
