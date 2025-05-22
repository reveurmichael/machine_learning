"""
Utility module for JSON processing in the AgenticVM.
Consolidates JSON extraction and validation functions used by multiple components.
"""

import json
import re
import logging

# Set up logging
logger = logging.getLogger("JSONUtils")
logger.setLevel(logging.INFO)

def extract_valid_json(text):
    """Extract valid JSON data from text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    try:
        # Handle the common "RESPONSE:" pattern seen in logs
        if text.startswith("RESPONSE:"):
            # Remove the "RESPONSE:" prefix and try to parse the rest
            response_text = text[len("RESPONSE:"):].strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, continue with other methods
                pass
                
        # First try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract just the JSON part if there's additional text
        try:
            # Find the first opening brace and last closing brace
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                # Extract just the JSON part
                json_part = text[first_brace:last_brace+1]
                return json.loads(json_part)
        except json.JSONDecodeError:
            pass
            
        # Try with our preprocessing for single quotes and unquoted keys
        try:
            preprocessed_text = preprocess_json_string(text)
            return json.loads(preprocessed_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code block
        json_data = extract_json_from_code_block(text)
        if json_data:
            return json_data
            
        # Try extracting from regular text
        json_data = extract_json_from_text(text)
        if json_data:
            return json_data
                
    return None

def preprocess_json_string(json_str):
    """Preprocess a JSON string to handle common LLM-generated issues.
    
    Args:
        json_str: JSON string that may contain syntax issues
        
    Returns:
        Preprocessed JSON string that's more likely to be valid
    """
    if not json_str or not isinstance(json_str, str):
        return json_str
        
    # Handle markdown code blocks if they're still present
    json_str = re.sub(r'```[a-z]*\n?', '', json_str)
    json_str = json_str.replace('```', '')
    
    # Handle triple backticks - common in LLM outputs
    json_str = json_str.replace('```json', '').replace('```', '')
    
    # Find potential multiline string values in the JSON
    # This is especially common for "notes" fields
    # The strategy is to identify and preserve them before other processing
    multiline_strings = {}
    counter = 0
    
    # Look for patterns that might be multiline strings
    # For example: "notes": "line1\nline2\nline3"
    for match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]*(?:\n[^"]*)+)"', json_str):
        key = match.group(1)
        value = match.group(2)
        if '\n' in value:
            placeholder = f"__MULTILINE_STRING_{counter}__"
            multiline_strings[placeholder] = value
            # Replace the multiline string with a placeholder
            json_str = json_str.replace(f'"{key}": "{value}"', f'"{key}": "{placeholder}"')
            counter += 1
    
    # Handle single quotes
    # First, escape any existing double quotes to avoid conflicts
    processed = json_str.replace('\\"', '_ESCAPED_DOUBLE_QUOTE_')
    # Convert single quotes to double quotes (but not those within already double-quoted strings)
    processed = re.sub(r"(?<!\\)'", '"', processed)
    # Restore escaped double quotes
    processed = processed.replace('_ESCAPED_DOUBLE_QUOTE_', '\\"')
    
    # Handle unquoted keys (word characters followed by colon)
    processed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', processed)
    
    # Remove trailing commas in objects and arrays
    processed = re.sub(r',\s*([}\]])', r'\1', processed)
    
    # Fix newlines within string literals - more comprehensive approach
    # First replace all newlines in multiline strings with a space
    processed = re.sub(r':\s*"([^"]*?)[\n\r]+([^"]*?)"', r': "\1 \2"', processed)
    processed = re.sub(r'"([^"]*?)[\n\r]+([^"]*?)"', r'"\1 \2"', processed)
    
    # Handle common formatting errors
    # Fix multiple colons
    processed = re.sub(r':\s*:', r':', processed)
    
    # Fix missing comma between array items
    processed = re.sub(r'(["}\]])\s*({")', r'\1,\2', processed)
    
    # Fix missing comma between object properties
    processed = re.sub(r'(["}\]])\s*"([^"]+)":', r'\1,"\\2":', processed)
    
    # Clean up excessive whitespace
    processed = re.sub(r'\s+', ' ', processed)
    
    # One final pass to remove any invalid control characters
    processed = re.sub(r'[\x00-\x1F\x7F]', ' ', processed)
    
    # Restore multiline strings
    for placeholder, value in multiline_strings.items():
        # Escape newlines for JSON
        escaped_value = value.replace('\n', '\\n')
        processed = processed.replace(f'"{placeholder}"', f'"{escaped_value}"')
    
    return processed

def validate_json_format(json_data):
    """Validate that JSON data follows the required format for snake moves.
    
    Args:
        json_data: Parsed JSON data
        
    Returns:
        Boolean indicating if the JSON follows the required format
    """
    # Check that json_data is a dict with the required keys
    if not isinstance(json_data, dict):
        return False
        
    # Check for required keys
    if "moves" not in json_data or "reasoning" not in json_data:
        return False
        
    # Check that moves is a list
    if not isinstance(json_data["moves"], list):
        return False
        
    # Check that each move is a valid direction
    valid_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for move in json_data["moves"]:
        if not isinstance(move, str) or move.upper() not in valid_directions:
            return False
            
    # Check that reasoning is a string
    if not isinstance(json_data["reasoning"], str):
        return False
        
    return True

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    try:
        # First check for explicitly marked JSON code blocks
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
        
        if not json_block_match:
            # Try with more generic backtick block
            json_block_match = re.search(r'```(?:json)?[\s\n]*({[\s\S]*?})[\s\n]*```', response, re.DOTALL)
        
        if not json_block_match:
            # Try alternate syntax (no opening/closing delimiters)
            json_block_match = re.search(r'(?:^|\n)({[\s\S]*?})(?:\n|$)', response, re.DOTALL)
            
        if not json_block_match:
            return None
        
        # Get the content inside the code block
        json_str = json_block_match.group(1)
        
        # For explicitly tagged JSON blocks, we might need to extract just the JSON part
        if not json_str.startswith('{'):
            # Try to find valid JSON inside the block
            inner_json_match = re.search(r'({[\s\S]*})', json_str, re.DOTALL)
            if inner_json_match:
                json_str = inner_json_match.group(1)
        
        # Use our comprehensive preprocessing function
        json_str = preprocess_json_string(json_str)
        
        # Log what we're trying to parse for debugging
        logger.debug(f"Attempting to parse JSON from code block: {json_str[:100]}...")
        
        data = json.loads(json_str)
        logger.info("Successfully parsed JSON from code block")
        return data
    except Exception as e:
        logger.debug(f"Failed to parse JSON code block: {e}")
        return None

def extract_json_from_text(response):
    """Extract JSON data directly from text without code block.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    try:
        # Check for common pattern: JSON followed by explanatory text
        # Extract from beginning of response to first double newline (empty line)
        parts = response.split('\n\n', 1)
        if len(parts) > 1:
            try:
                # Try parsing the first part
                data = json.loads(parts[0].strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
                
        # First try direct JSON parsing (for clean JSON responses)
        try:
            response_stripped = response.strip()
            data = json.loads(response_stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        
        # Handle common cases where JSON is surrounded by explanation text
        # Try to find any complete JSON object with curly braces
        possible_jsons = re.findall(r'({[\s\S]*?})', response, re.DOTALL)
        
        for json_str in possible_jsons:
            try:
                # Preprocess to handle common JSON issues
                processed = preprocess_json_string(json_str)
                data = json.loads(processed)
                
                # If it has steps, it's likely what we're looking for
                if isinstance(data, dict) and "steps" in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        # Try to match JSON objects with steps array specifically
        json_match = re.search(r'\{[\s\S]*?"steps"\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        
        # If that fails, try a more general pattern for any JSON object
        if not json_match:
            json_match = re.search(r'\{[\s\S]*?\}', response, re.DOTALL)
            
        if not json_match:
            return None
            
        json_str = json_match.group(0)
        # Use our comprehensive preprocessing function
        json_str = preprocess_json_string(json_str)
        
        # Now try to parse the cleaned JSON
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            # Try to fix multiline strings before giving up
            try:
                # Replace newlines in multiline strings with spaces
                fixed_json = json_str
                fixed_json = re.sub(r':\s*"([^"]*?)\n([^"]*?)"', r': "\1 \2"', fixed_json)
                # Also fix newlines in the middle of string values
                fixed_json = re.sub(r'"([^"]*?)\n([^"]*?)"', r'"\1 \2"', fixed_json)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # If we still can't parse it, try to extract only the part that looks like JSON
                # This handles cases where the JSON is malformed but still contains valuable information
                logger.debug("Attempting final cleanup of malformed JSON")
                cleaned_str = re.sub(r'[^{}[\],:"\'0-9a-zA-Z_\-\s.]', '', json_str)
                try:
                    return json.loads(cleaned_str)
                except:
                    logger.warning("Failed to parse JSON even after cleaning")
                    return None
    except Exception as e:
        logger.error(f"Error in JSON parsing: {e}")
        return None

def extract_moves_from_arrays(response):
    """Extract moves from array notation in text.
    
    Args:
        response: LLM response text
        
    Returns:
        List of extracted moves, or empty list if none found
    """
    moves = []
    
    # First try a more comprehensive approach to find arrays of direction strings
    # Support both single and double quotes
    array_match = re.search(r'\[\s*(["\'](?:UP|DOWN|LEFT|RIGHT)["\'](?:\s*,\s*["\'](?:UP|DOWN|LEFT|RIGHT)["\'])*)\s*\]', 
                          response, re.IGNORECASE | re.DOTALL)
    
    if array_match:
        # Extract all quoted strings (both single and double quotes)
        directions = re.findall(r'["\']([^"\']+)["\']', array_match.group(1))
        if directions:
            moves = [move.upper() for move in directions 
                   if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            return moves
    
    # Fallback: Look for arrays of directions in quotes (original method extended for single quotes)
    move_arrays = re.findall(r'\[\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*\s*\]', response)
    if move_arrays:
        for move_group in move_arrays:
            for move in move_group:
                if move and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    moves.append(move.upper())
                    
    return moves 

def validate_steps_format(json_data):
    """Validate that JSON data follows the required format for command steps.
    
    Args:
        json_data: Parsed JSON data
        
    Returns:
        Boolean indicating if the JSON follows the required format
    """
    # Check that json_data is a dict with the required keys
    if not isinstance(json_data, dict):
        logger.warning("JSON is not a dictionary")
        return False
        
    # Check for required keys
    if "steps" not in json_data:
        logger.warning("JSON does not contain 'steps' key")
        return False
        
    # Check that steps is a list
    if not isinstance(json_data["steps"], list):
        logger.warning("'steps' is not a list")
        return False
        
    # Check that each step has the required fields
    for i, step in enumerate(json_data["steps"]):
        if not isinstance(step, dict):
            logger.warning(f"Step {i} is not a dictionary")
            return False
            
        if "explanation" not in step:
            logger.warning(f"Step {i} does not have 'explanation'")
            return False
            
        if "command" not in step:
            logger.warning(f"Step {i} does not have 'command'")
            return False
            
        if not isinstance(step["explanation"], str):
            logger.warning(f"Step {i} 'explanation' is not a string")
            return False
            
        if not isinstance(step["command"], str):
            logger.warning(f"Step {i} 'command' is not a string")
            return False
    
    # Additional fields like "notes" are allowed in various formats
    if "notes" in json_data:
        # Notes can be either a list of strings or a single string
        if isinstance(json_data["notes"], list):
            for i, note in enumerate(json_data["notes"]):
                if not isinstance(note, str):
                    logger.warning(f"Note {i} is not a string")
                    return False
        elif not isinstance(json_data["notes"], str):
            logger.warning("'notes' is present but not a list or string")
            return False
    
    return True

def parse_command_response(response):
    """Parse a response from the LLM to extract commands in the JSON steps format.
    
    Args:
        response: LLM response text that may contain JSON with steps
        
    Returns:
        Tuple of (success, result) where:
            - success is a boolean indicating if valid JSON was found
            - result is either the parsed steps or an error message
    """
    try:
        # Log the first 100 characters of the response for debugging
        logger.debug(f"Parsing response starting with: {response[:100]}...")
        
        # Handle the specific pattern seen in logs: "RESPONSE:" followed by a JSON code block
        if "RESPONSE:" in response:
            # Split at RESPONSE: and take everything after it
            parts = response.split("RESPONSE:", 1)
            if len(parts) > 1:
                response_part = parts[1].strip()
                
                # Look for JSON code block
                code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_part, re.DOTALL)
                if code_block_match:
                    try:
                        json_str = code_block_match.group(1)
                        # Process the JSON string
                        json_str = preprocess_json_string(json_str)
                        data = json.loads(json_str)
                        
                        if validate_steps_format(data):
                            logger.info("Successfully parsed JSON from RESPONSE: code block")
                            return True, data
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON from RESPONSE: code block: {e}")
        
        # Try extracting just the JSON part if there's additional text
        try:
            # Find the first opening brace and last closing brace
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                # Extract just the JSON part
                json_part = response[first_brace:last_brace+1]
                data = json.loads(json_part)
                if validate_steps_format(data):
                    logger.info("Successfully parsed JSON by extracting JSON part")
                    return True, data
        except json.JSONDecodeError:
            logger.debug("JSON part extraction failed")
            pass
        
        # First try parsing directly - handle clean JSON without extraction
        try:
            cleaned_response = response.strip()
            data = json.loads(cleaned_response)
            if validate_steps_format(data):
                logger.info("Successfully parsed direct JSON response")
                return True, data
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying extraction methods")
            pass
        
        # Extract JSON from the response
        json_data = extract_valid_json(response)
        
        if not json_data:
            logger.debug("extract_valid_json failed, trying aggressive extraction")
            # Try an even more aggressive approach for JSON extraction
            # Look for everything between the first { and last } in the text
            matches = re.search(r'(\{[\s\S]*?\})(?:\s*\n\s*\n|$)', response, re.DOTALL)
            if matches:
                try:
                    potential_json = matches.group(1)
                    # Clean up any markdown code block markers
                    potential_json = re.sub(r'```[a-z]*\n?', '', potential_json)
                    potential_json = potential_json.replace('```', '')
                    # Process and try to parse
                    processed_json = preprocess_json_string(potential_json)
                    json_data = json.loads(processed_json)
                    
                    if validate_steps_format(json_data):
                        logger.info("Successfully parsed JSON using aggressive extraction")
                        return True, json_data
                except Exception as e:
                    logger.debug(f"Failed aggressive JSON extraction: {e}")
            
            logger.warning(f"No valid JSON found in response: {response[:50]}...")
            return False, "No valid JSON found in response"
        
        # Validate the JSON structure
        if not validate_steps_format(json_data):
            logger.warning(f"Invalid JSON format: {json.dumps(json_data)[:100]}...")
            return False, "Invalid JSON format: should contain 'steps' array with 'explanation' and 'command' for each step"
        
        # Return the parsed steps
        logger.info("Successfully parsed JSON response with valid steps format")
        return True, json_data
    except Exception as e:
        logger.error(f"Error in parse_command_response: {str(e)}")
        return False, f"Error parsing response: {str(e)}" 