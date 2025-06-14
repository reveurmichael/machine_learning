"""
Utility module for logging and data formatting.
Handles saving and formatting of game logs, LLM responses, and game summaries.
"""

import json

def format_raw_llm_response(raw_response, request_time, response_time, model_info):
    """Format a raw LLM response with metadata.
    
    Args:
        raw_response: Raw response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_info: Dictionary containing model details:
            - model_name: Name of the model
            - provider: Provider of the model
            - parser_model: Name of the parser model (optional)
            - parser_provider: Provider of the parser model (optional)
            - response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_info.get("model_name"),
        "provider": model_info.get("provider"),
        "response_duration_seconds": model_info.get("response_duration")
    }
    
    if "parser_model" in model_info:
        metadata["parser_model"] = model_info["parser_model"]
    
    if "parser_provider" in model_info:
        metadata["parser_provider"] = model_info["parser_provider"]
        
    # Format as metadata block followed by raw response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- RESPONSE ---\n{raw_response}"
    
    return formatted_response

def format_parsed_llm_response(parsed_response, request_time, response_time, model_info):
    """Format a parsed LLM response with metadata.
    
    Args:
        parsed_response: Parsed response from the LLM
        request_time: Time the request was made
        response_time: Time the response was received
        model_info: Dictionary containing model details:
            - model_name: Name of the model
            - provider: Provider of the model
            - response_duration: Duration of the response in seconds (optional)
        
    Returns:
        Formatted response with metadata
    """
    metadata = {
        "request_time": request_time,
        "response_time": response_time,
        "model": model_info.get("model_name"),
        "provider": model_info.get("provider"),
        "response_duration_seconds": model_info.get("response_duration")
    }
    
    # Format as metadata block followed by parsed response
    metadata_str = json.dumps(metadata, indent=2)
    formatted_response = f"--- METADATA ---\n{metadata_str}\n\n--- PARSED RESPONSE ---\n{parsed_response}"
    
    return formatted_response
