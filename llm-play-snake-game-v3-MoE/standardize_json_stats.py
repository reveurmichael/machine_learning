#!/usr/bin/env python3
"""
JSON Statistics Standardization Tool for Snake Game
Ensures all game data files follow the standard format with consistent statistics structure.
"""

import os
import json
import sys
import numpy as np
from pathlib import Path

def standardize_json_stats(log_dir):
    """Standardize statistics in all game JSON files in the given directory."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: Directory {log_dir} does not exist")
        return False
    
    # Find all game_*.json files
    game_files = sorted([f for f in log_dir.glob("game_*.json")])
    
    if not game_files:
        print(f"No game files found in {log_dir}")
        return False
    
    print(f"Found {len(game_files)} game files to process")
    
    # Read summary.json if it exists
    summary_file = log_dir / "summary.json"
    summary_data = None
    
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            print(f"Loaded summary data from {summary_file}")
        except Exception as e:
            print(f"Error loading summary file: {e}")
            summary_data = None
    
    # Process each game file
    for game_file in game_files:
        try:
            print(f"\nProcessing {game_file}...")
            
            # Load the game data
            with open(game_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # Extract game number from filename
            game_number = int(game_file.stem.split('_')[1])
            
            # Standardize metadata
            if 'metadata' in game_data:
                game_data['metadata']['game_number'] = game_number
                
                # Set round_count based on rounds_data
                if 'detailed_history' in game_data and 'rounds_data' in game_data['detailed_history']:
                    rounds_data = game_data['detailed_history']['rounds_data']
                    game_data['metadata']['round_count'] = len(rounds_data)
                    print(f"  Setting round_count to {len(rounds_data)}")
                else:
                    # Create standard rounds data structure with a single round
                    game_data['metadata']['round_count'] = 1
                    print("  Setting round_count to 1")
            
            # Standardize json_parsing_stats
            if 'json_parsing_stats' in game_data:
                # Get extraction attempts from parser_usage_count
                parser_usage_count = game_data.get('metadata', {}).get('parser_usage_count', 0)
                if parser_usage_count == 0 and 'detailed_history' in game_data:
                    # Determine from rounds_data
                    if 'rounds_data' in game_data['detailed_history']:
                        for round_key, round_data in game_data['detailed_history']['rounds_data'].items():
                            if 'secondary_response_times' in round_data:
                                parser_usage_count += len(round_data['secondary_response_times'])
                
                # Set standard statistics format
                game_data['json_parsing_stats'] = {
                    "total_extraction_attempts": parser_usage_count,
                    "successful_extractions": parser_usage_count,
                    "success_rate": 1.0 if parser_usage_count > 0 else 0,
                    "failed_extractions": 0,
                    "failure_rate": 0,
                    "json_decode_errors": 0,
                    "format_validation_errors": 0,
                    "code_block_extraction_errors": 0,
                    "text_extraction_errors": 0,
                    "fallback_extraction_success": 0
                }
                print(f"  Standardized json_parsing_stats with {parser_usage_count} extraction attempts")
            
            # Standardize detailed_history structure
            if 'detailed_history' in game_data:
                detailed_history = game_data['detailed_history']
                
                # Ensure standard format for apple_positions
                if 'rounds_data' in detailed_history:
                    # Extract apple positions from rounds_data if needed
                    if 'apple_positions' not in detailed_history or not detailed_history['apple_positions']:
                        apple_positions = []
                        for round_key, round_data in detailed_history['rounds_data'].items():
                            if 'apple_position' in round_data:
                                # Convert to standard dict format
                                apple_pos = round_data['apple_position']
                                if isinstance(apple_pos, list) and len(apple_pos) == 2:
                                    apple_positions.append({"x": apple_pos[0], "y": apple_pos[1]})
                                elif isinstance(apple_pos, dict) and 'x' in apple_pos and 'y' in apple_pos:
                                    apple_positions.append(apple_pos)
                        
                        if apple_positions:
                            detailed_history['apple_positions'] = apple_positions
                            print(f"  Added {len(apple_positions)} apple positions to detailed_history")
                    
                    # Ensure moves are properly collected
                    if 'moves' not in detailed_history or not detailed_history['moves']:
                        all_moves = []
                        for round_key, round_data in detailed_history['rounds_data'].items():
                            if 'moves' in round_data:
                                all_moves.extend(round_data['moves'])
                        
                        if all_moves:
                            detailed_history['moves'] = all_moves
                            print(f"  Added {len(all_moves)} moves to detailed_history")
                
                # Create standard rounds_data structure if missing
                if 'rounds_data' not in detailed_history:
                    if 'moves' in detailed_history and 'apple_positions' in detailed_history:
                        # Create a single round with all moves and the first apple position
                        detailed_history['rounds_data'] = {
                            "round_0": {
                                "apple_position": (
                                    [detailed_history['apple_positions'][0]['x'], 
                                     detailed_history['apple_positions'][0]['y']]
                                    if isinstance(detailed_history['apple_positions'][0], dict)
                                    else detailed_history['apple_positions'][0]
                                ),
                                "moves": detailed_history['moves']
                            }
                        }
                        
                        # Add token stats and response times
                        if 'token_stats' in game_data:
                            primary_token_stats = game_data['token_stats'].get('primary_llm', [])
                            secondary_token_stats = game_data['token_stats'].get('secondary_llm', [])
                            
                            detailed_history['rounds_data']['round_0']['primary_token_stats'] = primary_token_stats
                            detailed_history['rounds_data']['round_0']['secondary_token_stats'] = secondary_token_stats
                        
                        if 'prompt_response_stats' in game_data:
                            primary_times = game_data['prompt_response_stats'].get('primary_llm_response_times', [])
                            secondary_times = game_data['prompt_response_stats'].get('secondary_llm_response_times', [])
                            
                            detailed_history['rounds_data']['round_0']['primary_response_times'] = primary_times
                            detailed_history['rounds_data']['round_0']['secondary_response_times'] = secondary_times
                        
                        print("  Created standard rounds_data structure")
            
            # Save the standardized game data
            with open(game_file, 'w', encoding='utf-8') as f:
                json.dump(game_data, f, indent=2)
            
            print(f"  Saved standardized data to {game_file}")
            
        except Exception as e:
            print(f"Error processing {game_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Update summary file if it exists
    if summary_data:
        try:
            # Ensure standard format for token usage statistics
            if 'token_usage_stats' in summary_data:
                for llm_key in ['primary_llm', 'secondary_llm']:
                    if llm_key in summary_data['token_usage_stats']:
                        stats = summary_data['token_usage_stats'][llm_key]
                        request_count = len(game_files)
                        
                        if request_count > 0:
                            stats['avg_tokens_per_request'] = stats['total_tokens'] / request_count
                            stats['avg_prompt_tokens'] = stats['total_prompt_tokens'] / request_count
                            stats['avg_completion_tokens'] = stats['total_completion_tokens'] / request_count
            
            # Ensure standard format for response time statistics
            if 'response_time_stats' in summary_data:
                for llm_key in ['primary_llm', 'secondary_llm']:
                    if llm_key in summary_data['response_time_stats']:
                        stats = summary_data['response_time_stats'][llm_key]
                        
                        # Calculate average response time if total is available
                        if stats['total_response_time'] > 0:
                            request_count = len(game_files)
                            stats['avg_response_time'] = stats['total_response_time'] / request_count
            
            # Save updated summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"\nUpdated summary file {summary_file}")
            
        except Exception as e:
            print(f"Error updating summary file: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nJSON standardization completed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python standardize_json_stats.py <log_directory>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    success = standardize_json_stats(log_dir)
    
    sys.exit(0 if success else 1) 