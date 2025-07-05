"""
JSONL Utilities for Heuristics Extensions

This module provides JSONL-specific formatting utilities for generating
rich JSONL entries from game states and agent explanations.

Design Philosophy:
- Single responsibility: Only handles JSONL formatting
- Agent-agnostic: Works with any agent that provides explanations
- Rich content: Extracts full explanations and metrics for fine-tuning
- Consistent schema: Standardized JSONL output format across all agents

Usage:
    # Create a JSONL record
    jsonl_record = create_jsonl_record(game_state, move, explanation, metrics, agent_name)
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.print_utils import print_info, print_warning, print_error


def append_jsonl_records(jsonl_file_path: str, records: List[Dict[str, Any]], overwrite: bool = False) -> bool:
    """
    Append JSONL records to a file.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        records: List of record dictionaries with 'prompt' and 'completion' keys
        overwrite: If True, overwrite the file instead of appending
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(jsonl_file_path), exist_ok=True)
        
        # Choose file mode based on overwrite flag
        mode = 'w' if overwrite else 'a'
        
        # Write records to file
        with open(jsonl_file_path, mode, encoding='utf-8') as f:
            for record in records:
                if 'prompt' in record and 'completion' in record:
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
                else:
                    print_warning(f"[JSONLUtils] Skipping invalid record: {record}")
        
        action = "Overwrote" if overwrite else "Appended"
        print_info(f"[JSONLUtils] {action} {len(records)} records to {jsonl_file_path}")
        return True
        
    except Exception as e:
        print_error(f"[JSONLUtils] Error writing to JSONL file: {e}")
        return False


def format_jsonl_record(prompt: str, completion: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a JSONL record with prompt and completion.
    
    Args:
        prompt: Natural language prompt
        completion: Completion dictionary to be JSON serialized
        
    Returns:
        Formatted JSONL record
    """
    return {
        "prompt": prompt,
        "completion": json.dumps(completion, ensure_ascii=False)
    }


def validate_jsonl_record(record: Dict[str, Any]) -> bool:
    """
    Validate that a JSONL record has the correct format.
    
    Args:
        record: Record dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(record, dict):
        return False
    
    if 'prompt' not in record or 'completion' not in record:
        return False
    
    if not isinstance(record['prompt'], str):
        return False
    
    # Try to parse completion as JSON if it's a string
    if isinstance(record['completion'], str):
        try:
            json.loads(record['completion'])
        except json.JSONDecodeError:
            return False
    
    return True


def read_jsonl_file(jsonl_file_path: str) -> List[Dict[str, Any]]:
    """
    Read all records from a JSONL file.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        
    Returns:
        List of record dictionaries
    """
    records = []
    
    try:
        if not os.path.exists(jsonl_file_path):
            print_warning(f"[JSONLUtils] JSONL file does not exist: {jsonl_file_path}")
            return records
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    if validate_jsonl_record(record):
                        records.append(record)
                    else:
                        print_warning(f"[JSONLUtils] Invalid record at line {line_num}")
                except json.JSONDecodeError as e:
                    print_warning(f"[JSONLUtils] JSON decode error at line {line_num}: {e}")
        
        print_info(f"[JSONLUtils] Read {len(records)} records from {jsonl_file_path}")
        
    except Exception as e:
        print_error(f"[JSONLUtils] Error reading JSONL file: {e}")
    
    return records


def get_jsonl_stats(jsonl_file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a JSONL file.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        
    Returns:
        Dictionary with file statistics
    """
    stats = {
        'total_records': 0,
        'valid_records': 0,
        'invalid_records': 0,
        'file_size_bytes': 0,
        'exists': False
    }
    
    try:
        if not os.path.exists(jsonl_file_path):
            return stats
        
        stats['exists'] = True
        stats['file_size_bytes'] = os.path.getsize(jsonl_file_path)
        
        records = read_jsonl_file(jsonl_file_path)
        stats['total_records'] = len(records)
        stats['valid_records'] = len([r for r in records if validate_jsonl_record(r)])
        stats['invalid_records'] = stats['total_records'] - stats['valid_records']
        
    except Exception as e:
        print_error(f"[JSONLUtils] Error getting JSONL stats: {e}")
    
    return stats


def create_jsonl_record(game_state: Dict[str, Any], move: str, explanation: Any, 
                       metrics: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Create a rich JSONL record with natural language prompt and completion.
    The 'explanation' field in the JSONL output is always a rich, human-readable string (never a nested dict).
    Metrics are only in the 'metrics' field.
    
    Args:
        game_state: Game state dictionary
        move: Move made by the agent
        explanation: Agent explanation (dict or string)
        metrics: Agent metrics dictionary
        agent_name: Name of the agent
        
    Returns:
        JSONL record dictionary with 'prompt' and 'completion' fields
    """
    from jsonl_prompt_formatter import format_prompt_for_jsonl
    
    prompt = format_prompt_for_jsonl(game_state, agent_name)
    
    # Always flatten explanation for JSONL output
    explanation_text = flatten_explanation_for_jsonl(explanation)
    
    # Create rich completion with proper metrics
    completion = {
        "move": move,
        "algorithm": agent_name.lower(),
        "metrics": format_metrics_for_jsonl(metrics),
        "explanation": explanation_text
    }
    
    return {"prompt": prompt, "completion": json.dumps(completion, ensure_ascii=False)}


def flatten_explanation_for_jsonl(explanation: Any) -> str:
    """
    Convert a structured explanation dict to a rich, human-readable string for JSONL output.
    If already a string, return as-is. If dict, use 'natural_language_summary' and 'explanation_steps'.
    
    Args:
        explanation: The explanation object to flatten
        
    Returns:
        Rich, human-readable explanation string
    """
    if isinstance(explanation, str):
        return explanation
    if isinstance(explanation, dict):
        # Prefer natural_language_summary + explanation_steps
        summary = explanation.get('natural_language_summary', '')
        steps = explanation.get('explanation_steps', [])
        if steps and isinstance(steps, list):
            steps_text = '\n'.join(steps)
        else:
            steps_text = ''
        # Compose
        if summary and steps_text:
            return f"{steps_text}\n\n{summary}"
        elif steps_text:
            return steps_text
        elif summary:
            return summary
        # Fallback: join all string fields in the dict
        return '\n'.join(str(v) for v in explanation.values() if isinstance(v, str))
    # Fallback: just str()
    return str(explanation)


def format_metrics_for_jsonl(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for JSONL completion, ensuring all values are JSON serializable.
    
    Args:
        metrics: Raw metrics dictionary
        
    Returns:
        Formatted metrics dictionary
    """
    if not metrics:
        return {}
    
    # Map metric names to match the expected format
    formatted_metrics = {}
    
    # Common metric mappings
    metric_mappings = {
        'manhattan_distance': 'manhattan_distance',
        'obstacles_near_path': 'obstacles_near_path',
        'remaining_free_cells': 'remaining_free_cells',
        'valid_moves': 'valid_moves',
        'final_chosen_direction': 'chosen_direction',
        'apple_path_length': 'path_length',
        'apple_path_safe': 'apple_path_safe',
        'fallback_used': 'fallback_used',
        # Position and game state metrics
        'head_position': 'head_position',
        'apple_position': 'apple_position',
        'snake_length': 'snake_length',
        'grid_size': 'grid_size'
    }
    
    for old_key, new_key in metric_mappings.items():
        if old_key in metrics:
            value = metrics[old_key]
            # Convert numpy types to Python types for JSON serialization
            if hasattr(value, 'item'):  # numpy scalar
                formatted_metrics[new_key] = value.item()
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples that might contain numpy types
                formatted_metrics[new_key] = [
                    item.item() if hasattr(item, 'item') else item 
                    for item in value
                ]
            elif isinstance(value, dict):
                # Handle nested dictionaries
                formatted_metrics[new_key] = format_metrics_for_jsonl(value)
            else:
                formatted_metrics[new_key] = value
    
    return formatted_metrics 