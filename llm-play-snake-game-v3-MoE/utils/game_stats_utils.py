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

def create_display_dataframe(stats_df):
    """Create a user-friendly display DataFrame for the UI.
    
    Args:
        stats_df: DataFrame with raw game statistics
        
    Returns:
        DataFrame formatted for display
    """
    # Select relevant columns
    display_df = stats_df[['folder', 'total_score', 'total_steps', 'mean_score', 'max_score', 
                          'primary_llm', 'secondary_llm', 'avg_response_time', 
                          'avg_secondary_response_time', 'steps_per_apple', 'json_success_rate']].copy()
    
    # Rename columns for better display
    display_df.columns = ['Folder', 'Total Score', 'Total Steps', 'Mean Score', 'Max Score', 
                          'Primary LLM', 'Secondary LLM', 'Avg Response Time (s)', 
                          'Avg Secondary Response Time (s)', 'Steps Per Apple', 'JSON Success Rate (%)']
    
    # Format display values
    display_df['Folder'] = display_df['Folder'].apply(lambda x: os.path.basename(x))
    display_df['Avg Response Time (s)'] = display_df['Avg Response Time (s)'].apply(lambda x: f"{x:.2f}")
    display_df['Avg Secondary Response Time (s)'] = display_df['Avg Secondary Response Time (s)'].apply(lambda x: f"{x:.2f}")
    display_df['Steps Per Apple'] = display_df['Steps Per Apple'].apply(lambda x: f"{x:.2f}")
    display_df['JSON Success Rate (%)'] = display_df['JSON Success Rate (%)'].apply(lambda x: f"{x:.2f}")
    
    return display_df

def create_game_performance_chart(stats_df, output_dir=None):
    """Create a performance chart comparing different experiments.
    
    Args:
        stats_df: DataFrame with experiment statistics
        output_dir: Optional directory to save the chart
    """
    plt.figure(figsize=(12, 6))
    
    # Plot mean score for each experiment
    plt.bar(stats_df['folder'].apply(os.path.basename), stats_df['mean_score'])
    plt.title('Mean Score by Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Mean Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'performance_chart.png'))
    else:
        plt.show()

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
                         empty_steps=0, error_steps=0, max_empty_moves=3):
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
    
    # Update basic information
    info_data['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_data['total_games'] = game_count
    
    # Update game statistics
    info_data['game_statistics'] = {
        'total_games': game_count,
        'total_score': total_score,
        'total_steps': total_steps,
        'max_score': max(game_scores) if game_scores else 0,
        'min_score': min(game_scores) if game_scores else 0,
        'mean_score': total_score / game_count if game_count > 0 else 0,
        'parser_usage_count': parser_usage_count,
        'empty_steps': empty_steps,
        'error_steps': error_steps,
        'max_empty_moves': max_empty_moves
    }
    
    # Add JSON error statistics if available
    if json_error_stats:
        # Calculate success rate
        total_attempts = json_error_stats.get("total_extraction_attempts", 0)
        successful_extractions = json_error_stats.get("successful_extractions", 0)
        failed_extractions = json_error_stats.get("failed_extractions", 0)
        
        success_rate = (successful_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        failure_rate = (failed_extractions / total_attempts) * 100 if total_attempts > 0 else 0
        
        info_data["json_parsing_stats"] = {
            "total_extraction_attempts": total_attempts,
            "successful_extractions": successful_extractions,
            "success_rate": success_rate,
            "failed_extractions": failed_extractions,
            "failure_rate": failure_rate,
            "json_decode_errors": json_error_stats.get("json_decode_errors",0),
            "format_validation_errors": json_error_stats.get("format_validation_errors",0),
            "code_block_extraction_errors": json_error_stats.get("code_block_extraction_errors",0),
            "text_extraction_errors": json_error_stats.get("text_extraction_errors",0),
            "fallback_extraction_success": json_error_stats.get("fallback_extraction_success",0)
        }
    
    # Add efficiency metrics
    info_data["efficiency_metrics"] = {
        "apples_per_step": total_score/(total_steps if total_steps > 0 else 1),
        "steps_per_game": total_steps/game_count if game_count > 0 else 0,
        "valid_move_ratio": (total_steps - empty_steps - error_steps)/(total_steps if total_steps > 0 else 1)
    }
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2)
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
        max_empty_moves=max_empty_moves
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