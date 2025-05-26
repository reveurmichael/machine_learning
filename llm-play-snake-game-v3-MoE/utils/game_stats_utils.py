"""
Utility module for game statistics and visualization.
Handles game statistics processing and visualization for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
from colorama import Fore
import plotly.express as px

def create_display_dataframe(stats_df):
    """Create a display dataframe with selected columns.
    
    Args:
        stats_df: DataFrame containing game statistics
        
    Returns:
        DataFrame with selected columns for display
    """
    if stats_df is None or len(stats_df) == 0:
        return pd.DataFrame()
    
    # Create a copy of the dataframe
    display_df = stats_df.copy()
    
    # Extract LLM information
    display_df['LLM'] = display_df['primary_llm'].apply(lambda x: x.split('-')[0].strip() if isinstance(x, str) else 'Unknown')
    display_df['Model'] = display_df['primary_llm'].apply(lambda x: x.split('-')[1].strip() if isinstance(x, str) and '-' in x else 'Unknown')
    
    # Create a dataframe with selected columns
    columns = [
        'date', 
        'LLM', 
        'Model', 
        'total_games', 
        'total_score', 
        'total_steps', 
        'mean_score', 
        'steps_per_apple',
        'json_success_rate', 
        'avg_response_time',
        'avg_secondary_response_time'
    ]
    
    # Add token statistics if available
    if 'avg_prompt_tokens' in display_df.columns:
        columns.extend(['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'])
    
    # Select columns that exist in the dataframe
    existing_columns = [col for col in columns if col in display_df.columns]
    
    # Create the display dataframe
    result_df = display_df[existing_columns].copy()
    
    # Format date column
    if 'date' in result_df.columns:
        result_df['date'] = pd.to_datetime(result_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Format numeric columns to 2 decimal places
    numeric_columns = [
        'mean_score', 
        'steps_per_apple', 
        'json_success_rate', 
        'avg_response_time',
        'avg_secondary_response_time',
        'avg_prompt_tokens',
        'avg_completion_tokens',
        'avg_total_tokens'
    ]
    
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Add token usage columns if they exist
    if 'token_stats' in stats_df.columns:
        # Extract primary token stats
        result_df['Primary Tokens'] = stats_df['token_stats'].apply(
            lambda x: x.get('primary', {}).get('total_tokens', 0) if isinstance(x, dict) else 0
        )
        
        # Extract secondary token stats
        result_df['Secondary Tokens'] = stats_df['token_stats'].apply(
            lambda x: x.get('secondary', {}).get('total_tokens', 0) if isinstance(x, dict) else 0
        )
        
        # Calculate total tokens
        result_df['Total Tokens'] = result_df['Primary Tokens'] + result_df['Secondary Tokens']
    
    # Rename columns for display
    column_renames = {
        'date': 'Date',
        'total_games': 'Games',
        'total_score': 'Score',
        'total_steps': 'Steps',
        'mean_score': 'Avg Score',
        'steps_per_apple': 'Steps/Apple',
        'json_success_rate': 'JSON Success %',
        'avg_response_time': 'Primary LLM Response (s)',
        'avg_secondary_response_time': 'Secondary LLM Response (s)',
        'avg_prompt_tokens': 'Avg Prompt Tokens',
        'avg_completion_tokens': 'Avg Completion Tokens',
        'avg_total_tokens': 'Avg Total Tokens'
    }
    
    # Rename columns that exist in the dataframe
    rename_dict = {k: v for k, v in column_renames.items() if k in result_df.columns}
    result_df = result_df.rename(columns=rename_dict)
    
    return result_df

def create_game_performance_chart(stats_df, output_dir=None):
    """Create a chart comparing game performance metrics.
    
    Args:
        stats_df: DataFrame containing game statistics
        output_dir: Directory to save the chart (optional)
        
    Returns:
        Plotly figure object
    """
    if stats_df is None or len(stats_df) == 0:
        return None
    
    # Extract folder names for display
    stats_df['session'] = stats_df['folder'].apply(lambda x: os.path.basename(x))
    
    # Extract performance metrics
    performance_df = pd.DataFrame({
        'Session': stats_df['session'],
        'Average Score': stats_df['mean_score'],
        'Steps per Apple': stats_df['steps_per_apple'],
        'JSON Success Rate (%)': stats_df['json_success_rate'],
        'Primary Response Time (s)': stats_df['avg_response_time'],
        'Secondary Response Time (s)': stats_df['avg_secondary_response_time']
    })
    
    # Add token metrics if available
    if 'token_stats' in stats_df.columns:
        # Check for primary token stats
        has_primary_tokens = any(
            isinstance(x, dict) and 'primary' in x and 'avg_total_tokens' in x['primary']
            for x in stats_df['token_stats'] if isinstance(x, dict)
        )
        
        if has_primary_tokens:
            performance_df['Primary Tokens per Request'] = stats_df['token_stats'].apply(
                lambda x: x.get('primary', {}).get('avg_total_tokens', 0) if isinstance(x, dict) else 0
            )
        
        # Check for secondary token stats
        has_secondary_tokens = any(
            isinstance(x, dict) and 'secondary' in x and 'avg_total_tokens' in x['secondary']
            for x in stats_df['token_stats'] if isinstance(x, dict)
        )
        
        if has_secondary_tokens:
            performance_df['Secondary Tokens per Request'] = stats_df['token_stats'].apply(
                lambda x: x.get('secondary', {}).get('avg_total_tokens', 0) if isinstance(x, dict) else 0
            )
    
    # Melt the dataframe for plotting
    melted_df = pd.melt(performance_df, id_vars=['Session'], var_name='Metric', value_name='Value')
    
    # Create the plot
    fig = px.bar(melted_df, x='Session', y='Value', color='Metric', barmode='group',
                title='Game Performance Comparison', template='plotly_white')
    
    # Update layout
    fig.update_layout(
        xaxis_title='Game Session',
        yaxis_title='Value',
        legend_title='Metric',
        height=600
    )
    
    # Save the chart if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_image(os.path.join(output_dir, 'performance_chart.png'))
        fig.write_html(os.path.join(output_dir, 'performance_chart.html'))
    
    return fig

def create_game_dataframe(game_data):
    """Create a DataFrame from game data dictionary.
    
    Args:
        game_data: Dictionary containing game data
        
    Returns:
        DataFrame with game information
    """
    game_list = []
    for game_num, data in game_data.items():
        game_list.append({
            'Game #': game_num,
            'Score': data.get('score', 0),
            'Steps': data.get('steps', 0),
            'End Reason': data.get('game_end_reason', 'Unknown'),
            'Has Apple Positions': data.get('has_apple_positions', False),
            'Has Moves': data.get('has_moves', False),
            'Move Count': data.get('move_count', 0)
        })
    
    return pd.DataFrame(game_list)

def get_experiment_options(stats_df):
    """Extract experiment options for filtering.
    
    Args:
        stats_df: DataFrame with experiment statistics
        
    Returns:
        Dictionary with options for providers, models, primary LLMs, and secondary LLMs
    """
    # Get unique providers and models from all experiments
    all_providers = []
    all_models = []
    for s in stats_df['providers']:
        all_providers.extend([p for p in s if p not in all_providers])
    for s in stats_df['models']:
        all_models.extend([m for m in s if m not in all_models])
    
    # Get unique primary and secondary LLMs
    all_primary_llms = sorted(stats_df['primary_llm'].unique())
    all_secondary_llms = sorted(stats_df['secondary_llm'].unique())
    
    return {
        'providers': sorted(all_providers),
        'models': sorted(all_models),
        'primary_llms': all_primary_llms,
        'secondary_llms': all_secondary_llms
    }

def filter_experiments(stats_df, provider=None, model=None, primary_llm=None, secondary_llm=None):
    """Filter experiments based on criteria.
    
    Args:
        stats_df: DataFrame with experiment statistics
        provider: Filter by provider
        model: Filter by model
        primary_llm: Filter by primary LLM
        secondary_llm: Filter by secondary LLM
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = stats_df.copy()
    
    if provider:
        filtered_df = filtered_df[filtered_df['providers'].apply(lambda x: provider in x)]
    
    if model:
        filtered_df = filtered_df[filtered_df['models'].apply(lambda x: model in x)]
    
    if primary_llm:
        filtered_df = filtered_df[filtered_df['primary_llm'] == primary_llm]
    
    if secondary_llm:
        filtered_df = filtered_df[filtered_df['secondary_llm'] == secondary_llm]
    
    return filtered_df

def extract_game_stats(log_folder):
    """Extract game statistics from a log folder.
    
    Args:
        log_folder: Path to the log folder
        
    Returns:
        Dictionary with game statistics
    """
    stats = {
        'folder': log_folder,
        'date': None,
        'total_games': 0,
        'total_score': 0,
        'total_steps': 0,
        'max_score': 0,
        'min_score': 0,
        'mean_score': 0,
        'primary_llm': 'Unknown',
        'secondary_llm': 'Unknown',
        'avg_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0,
        'providers': [],
        'models': [],
        'game_data': {}  # Store per-game data
    }
    
    # Load data from info.json
    info_path = Path(log_folder) / 'info.json'
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract experiment information
            stats['date'] = info_data.get('date')
            
            # Extract LLM information
            primary_llm_info = info_data.get('primary_llm', {})
            if primary_llm_info:
                primary_provider = primary_llm_info.get('provider')
                primary_model = primary_llm_info.get('model')
                
                if primary_provider:
                    stats['primary_llm'] = primary_provider
                    stats['providers'].append(primary_provider)
                
                if primary_model:
                    stats['primary_llm'] += f" - {primary_model}"
                    stats['models'].append(primary_model)
            
            secondary_llm_info = info_data.get('secondary_llm', {})
            if secondary_llm_info:
                secondary_provider = secondary_llm_info.get('provider')
                secondary_model = secondary_llm_info.get('model')
                
                if secondary_provider:
                    stats['secondary_llm'] = secondary_provider
                    if secondary_provider not in stats['providers']:
                        stats['providers'].append(secondary_provider)
                
                if secondary_model:
                    stats['secondary_llm'] += f" - {secondary_model}"
                    if secondary_model not in stats['models']:
                        stats['models'].append(secondary_model)
            
            # Extract game statistics
            game_stats = info_data.get('game_statistics', {})
            stats['total_games'] = game_stats.get('total_games', 0)
            stats['total_score'] = game_stats.get('total_score', 0)
            stats['total_steps'] = game_stats.get('total_steps', 0)
            stats['max_score'] = game_stats.get('max_score', 0)
            stats['min_score'] = game_stats.get('min_score', 0)
            stats['mean_score'] = game_stats.get('mean_score', 0.0)
            
            # Extract JSON success rate
            json_stats = info_data.get('json_parsing_stats', {})
            stats['json_success_rate'] = json_stats.get('success_rate', 0.0)
            
        except Exception as e:
            print(f"Error reading info.json: {e}")
    
    # Extract per-game data from JSON summary files
    total_response_time = 0
    total_secondary_response_time = 0
    total_steps_per_apple = 0
    game_count = 0
    
    for i in range(1, 7):  # Assuming max 6 games
        json_summary_file = Path(log_folder) / f"game{i}_summary.json"
        if json_summary_file.exists():
            game_data = extract_game_summary(json_summary_file)
            if game_data:
                stats['game_data'][i] = game_data
                game_count += 1
                
                # Accumulate response times and steps per apple
                if 'avg_primary_response_time' in game_data:
                    total_response_time += game_data['avg_primary_response_time']
                
                if 'avg_secondary_response_time' in game_data:
                    total_secondary_response_time += game_data['avg_secondary_response_time']
                
                if 'steps_per_apple' in game_data:
                    total_steps_per_apple += game_data['steps_per_apple']
    
    # Calculate averages across games
    if game_count > 0:
        stats['avg_response_time'] = total_response_time / game_count
        stats['avg_secondary_response_time'] = total_secondary_response_time / game_count
        stats['steps_per_apple'] = total_steps_per_apple / game_count
    
    # Update total_games if not set from info.json
    if stats['total_games'] == 0:
        stats['total_games'] = len(stats['game_data'])
    
    return stats

def extract_game_summary(summary_file):
    """Extract game summary from a summary file.
    
    Args:
        summary_file: Path to the game summary file
        
    Returns:
        Dictionary with game summary information
    """
    summary = {
        'file': str(summary_file),
        'score': 0,
        'steps': 0,
        'game_end_reason': None,
        'has_apple_positions': False,
        'has_moves': False,
        'move_count': 0,
        'avg_primary_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0
    }
    
    try:
        # Parse JSON file
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract fields from JSON
        summary['score'] = data.get('score', 0)
        summary['steps'] = data.get('steps', 0)
        summary['game_end_reason'] = data.get('game_end_reason', 'Unknown')
        
        # Check if the file has apple positions
        if 'apple_positions' in data and data['apple_positions']:
            summary['has_apple_positions'] = True
            
        # Check if the file has moves
        if 'moves' in data and data['moves']:
            summary['has_moves'] = True
            summary['move_count'] = len(data['moves'])
        
        # Extract response time metrics if available
        if 'prompt_response_stats' in data:
            prompt_stats = data.get('prompt_response_stats', {})
            summary['avg_primary_response_time'] = prompt_stats.get('avg_primary_response_time', 0)
            summary['avg_secondary_response_time'] = prompt_stats.get('avg_secondary_response_time', 0)
        
        # Extract performance metrics if available
        if 'performance_metrics' in data:
            perf_metrics = data.get('performance_metrics', {})
            summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
        
        # Extract JSON parsing success rate if available
        if 'json_parsing_stats' in data:
            json_stats = data.get('json_parsing_stats', {})
            summary['json_success_rate'] = json_stats.get('success_rate', 0)
            
    except Exception as e:
        print(f"Error extracting summary from {summary_file}: {e}")
    
    return summary

def update_experiment_info(log_dir, game_count, total_score, total_steps, 
                         json_error_stats, parser_usage_count=0, game_scores=None, 
                         empty_steps=0, error_steps=0, max_empty_moves=3, token_stats=None):
    """Update the experiment information JSON file with game statistics.
    
    Args:
        log_dir: Directory containing the experiment logs
        game_count: Total number of games played
        total_score: Total score across all games
        total_steps: Total steps taken across all games
        json_error_stats: Statistics about JSON parsing errors
        parser_usage_count: Number of times the parser was used
        game_scores: List of scores from all games
        empty_steps: Number of empty steps
        error_steps: Number of error steps
        max_empty_moves: Maximum allowed empty moves
        token_stats: Token statistics from game summary files
    """
    file_path = os.path.join(log_dir, 'info.json')
    
    # Load existing data if file exists
    info_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
        except Exception as e:
            print(f"Error reading existing info.json: {e}")
    
    # Create a new ordered dictionary with important information at the top
    ordered_info = {}
    
    # 1. Basic session information (date, session ID, etc.)
    ordered_info['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ordered_info['total_games'] = game_count
    
    # 2. Critical performance metrics
    ordered_info['game_statistics'] = {
        'total_score': total_score,
        'total_steps': total_steps,
        'mean_score': total_score / game_count if game_count > 0 else 0,
        'max_score': max(game_scores) if game_scores else 0,
        'min_score': min(game_scores) if game_scores else 0,
        'steps_per_apple': total_score/(total_steps if total_steps > 0 else 1),
    }
    
    # 3. LLM information (extract from existing data)
    if 'primary_llm' in info_data:
        ordered_info['primary_llm'] = info_data['primary_llm']
    
    if 'secondary_llm' in info_data:
        ordered_info['secondary_llm'] = info_data['secondary_llm']
    
    # 4. JSON success metrics
    if json_error_stats:
        # Calculate success rate
        total_attempts = json_error_stats.get("total_extraction_attempts", 0)
        successful_extractions = json_error_stats.get("successful_extractions", 0)
        failed_extractions = json_error_stats.get("failed_extractions", 0)
        
        success_rate = (successful_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        failure_rate = (failed_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        
        ordered_info["json_parsing_stats"] = {
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "total_extraction_attempts": total_attempts,
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
        }
    
    # 5. Efficiency metrics
    ordered_info["efficiency_metrics"] = {
        "apples_per_step": total_score/(total_steps if total_steps > 0 else 1),
        "steps_per_game": total_steps/game_count if game_count > 0 else 0,
        "valid_move_ratio": (total_steps - empty_steps - error_steps)/(total_steps if total_steps > 0 else 1)
    }
    
    # 6. Detailed game statistics (moved lower in priority)
    ordered_info['detailed_game_statistics'] = {
        'total_games': game_count,
        'parser_usage_count': parser_usage_count,
        'empty_steps': empty_steps,
        'error_steps': error_steps,
        'max_empty_moves': max_empty_moves
    }
    
    # 7. Detailed JSON stats (lower priority)
    if json_error_stats:
        ordered_info["detailed_json_parsing_stats"] = {
            "json_decode_errors": json_error_stats.get("json_decode_errors", 0),
            "format_validation_errors": json_error_stats.get("format_validation_errors", 0),
            "code_block_extraction_errors": json_error_stats.get("code_block_extraction_errors", 0),
            "text_extraction_errors": json_error_stats.get("text_extraction_errors", 0),
            "fallback_extraction_success": json_error_stats.get("fallback_extraction_success", 0)
        }
    
    # 8. Raw data (lowest priority)
    if 'game_scores' in info_data:
        ordered_info['game_scores'] = info_data['game_scores']
    elif game_scores:
        ordered_info['game_scores'] = game_scores
    
    # 9. Token statistics (lowest priority)
    if token_stats:
        ordered_info['token_stats'] = token_stats
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_info, f, indent=2)
    print(Fore.GREEN + f"📝 Updated experiment info saved to {file_path}")

def report_final_statistics(log_dir, game_count, total_score, total_steps,
                          parser_usage_count, game_scores, empty_steps, 
                          error_steps, max_empty_moves):
    """Report final statistics at the end of the game session.
    
    Args:
        log_dir: Directory for logs
        game_count: Total games played
        total_score: Total score across all games
        total_steps: Total steps across all games
        parser_usage_count: Count of parser usage
        game_scores: List of scores from all games
        empty_steps: Number of empty steps
        error_steps: Number of error steps
        max_empty_moves: Maximum allowed empty moves
    """
    from utils.json_utils import get_json_error_stats
    
    # Get token statistics from game summary files
    token_stats = extract_token_stats_from_summaries(log_dir)
    
    # Update experiment info with final statistics
    json_error_stats = get_json_error_stats()
    update_experiment_info(
        log_dir, 
        game_count, 
        total_score, 
        total_steps, 
        json_error_stats,
        parser_usage_count, 
        game_scores, 
        empty_steps, 
        error_steps,
        max_empty_moves=max_empty_moves,
        token_stats=token_stats
    )
    
    print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
    print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
    print(Fore.GREEN + f"🏁 Final Score: {total_score}")
    print(Fore.GREEN + f"👣 Total Steps: {total_steps}")
    print(Fore.GREEN + f"🔄 Secondary LLM was used {parser_usage_count} times")
    
    if game_count > 0:
        print(Fore.GREEN + f"📊 Average Score: {total_score/game_count:.2f}")
    
    if total_steps > 0:
        print(Fore.GREEN + f"📈 Apples per Step: {total_score/total_steps:.4f}")
        
    print(Fore.GREEN + f"📈 Empty Steps: {empty_steps}")
    print(Fore.GREEN + f"📈 Error Steps: {error_steps}")
    
    if json_error_stats['total_extraction_attempts'] > 0:
        print(Fore.GREEN + f"📈 JSON Extraction Attempts: {json_error_stats['total_extraction_attempts']}")
        success_rate = (json_error_stats['successful_extractions'] / json_error_stats['total_extraction_attempts']) * 100
        print(Fore.GREEN + f"📈 JSON Extraction Success Rate: {success_rate:.2f}%")

def extract_token_stats_from_summaries(log_dir):
    """Extract token statistics from game summary files.
    
    Args:
        log_dir: Directory containing game summary files
        
    Returns:
        Dictionary with aggregated token statistics
    """
    primary_token_stats = {
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "response_times": []
    }
    
    secondary_token_stats = {
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "response_times": []
    }
    
    # Loop through game summary files
    for i in range(1, 7):  # Assuming max 6 games
        summary_file = os.path.join(log_dir, f"game{i}_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract token statistics
                if "token_stats" in data:
                    token_stats = data["token_stats"]
                    
                    # Primary LLM token stats
                    if "primary" in token_stats:
                        primary = token_stats["primary"]
                        primary_token_stats["total_tokens"] += primary.get("total_tokens", 0)
                        primary_token_stats["total_prompt_tokens"] += primary.get("total_prompt_tokens", 0)
                        primary_token_stats["total_completion_tokens"] += primary.get("total_completion_tokens", 0)
                    
                    # Secondary LLM token stats
                    if "secondary" in token_stats:
                        secondary = token_stats["secondary"]
                        secondary_token_stats["total_tokens"] += secondary.get("total_tokens", 0)
                        secondary_token_stats["total_prompt_tokens"] += secondary.get("total_prompt_tokens", 0)
                        secondary_token_stats["total_completion_tokens"] += secondary.get("total_completion_tokens", 0)
                
                # Extract response time statistics
                if "prompt_response_stats" in data:
                    stats = data["prompt_response_stats"]
                    
                    # Store primary response times
                    if "avg_primary_response_time" in stats:
                        # Use min/max/avg to approximate response times
                        avg_time = stats.get("avg_primary_response_time", 0)
                        min_time = stats.get("min_primary_response_time", avg_time)
                        max_time = stats.get("max_primary_response_time", avg_time)
                        
                        # Add approximate times to the list
                        primary_token_stats["response_times"].append(avg_time)
                        if min_time != avg_time:
                            primary_token_stats["response_times"].append(min_time)
                        if max_time != avg_time:
                            primary_token_stats["response_times"].append(max_time)
                    
                    # Store secondary response times
                    if "avg_secondary_response_time" in stats:
                        # Use min/max/avg to approximate response times
                        avg_time = stats.get("avg_secondary_response_time", 0)
                        min_time = stats.get("min_secondary_response_time", avg_time)
                        max_time = stats.get("max_secondary_response_time", avg_time)
                        
                        # Add approximate times to the list
                        secondary_token_stats["response_times"].append(avg_time)
                        if min_time != avg_time:
                            secondary_token_stats["response_times"].append(min_time)
                        if max_time != avg_time:
                            secondary_token_stats["response_times"].append(max_time)
                
            except Exception as e:
                print(f"Error extracting token stats from {summary_file}: {e}")
    
    # Calculate averages if data is available
    request_count_primary = max(1, len(primary_token_stats["response_times"]))
    primary_token_stats["avg_total_tokens"] = primary_token_stats["total_tokens"] / request_count_primary
    primary_token_stats["avg_prompt_tokens"] = primary_token_stats["total_prompt_tokens"] / request_count_primary
    primary_token_stats["avg_completion_tokens"] = primary_token_stats["total_completion_tokens"] / request_count_primary
    
    request_count_secondary = max(1, len(secondary_token_stats["response_times"]))
    secondary_token_stats["avg_total_tokens"] = secondary_token_stats["total_tokens"] / request_count_secondary
    secondary_token_stats["avg_prompt_tokens"] = secondary_token_stats["total_prompt_tokens"] / request_count_secondary
    secondary_token_stats["avg_completion_tokens"] = secondary_token_stats["total_completion_tokens"] / request_count_secondary
    
    return {
        "primary": primary_token_stats,
        "secondary": secondary_token_stats
    } 