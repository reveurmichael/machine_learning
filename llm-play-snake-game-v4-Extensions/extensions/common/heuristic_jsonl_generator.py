"""
Heuristic JSONL Dataset Generator (v0.04)

This module provides JSONL dataset generation for heuristic snake agents with
rich natural language explanations, specifically designed for LLM fine-tuning
and language model training pipelines.

Key Features:
- JSONL format (JSON Lines) optimal for LLM training
- Rich natural language explanations from heuristic agents
- Multiple prompt format templates (simple, detailed, instruction)
- Prompt-completion pairs for supervised fine-tuning
- Metadata tracking for training pipeline integration
- GRID SIZE ADAPTIVE: Automatically detects grid size and organizes datasets by grid size

CRITICAL ARCHITECTURAL RULE: GRID SIZE BASED DATASET ORGANIZATION
--------------------

The grid_size should NEVER be hardcoded! Generated JSONL datasets must be stored in:
./logs/extensions/datasets/grid-size-N/

This ensures proper segregation and organization by grid complexity.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Import grid size detection utilities from CSV generator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heuristic_csv_generator import GridSizeDetector, DatasetDirectoryManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeuristicJSONLGenerator:
    """Enhanced JSONL Dataset Generator with Grid Size Awareness"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def generate_dataset(self, log_dir: str, output_base_dir: str = None,
                        algorithm: str = None, prompt_format: str = "detailed",
                        max_games: Optional[int] = None) -> Dict[str, Any]:
        """Generate JSONL dataset with automatic grid size detection and organization."""
        # STEP 1: Detect grid size - CRITICAL for proper organization
        grid_size = GridSizeDetector.detect_grid_size_from_log_directory(log_dir)
        logger.info(f"Detected grid size: {grid_size}")
        
        # STEP 2: Ensure proper directory structure
        if output_base_dir is None:
            dataset_dir = DatasetDirectoryManager.ensure_grid_size_directory_exists(grid_size)
        else:
            dataset_dir = os.path.join(output_base_dir, f"grid-size-{grid_size}")
            os.makedirs(dataset_dir, exist_ok=True)
        
        # STEP 3: Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # STEP 4: Determine algorithm name if not provided
        if algorithm is None:
            algorithm = self._extract_algorithm_from_log_dir(log_dir)
        
        # STEP 5: Generate dataset filename
        filename = DatasetDirectoryManager.generate_dataset_filename(algorithm, timestamp, "jsonl")
        output_file = os.path.join(dataset_dir, filename)
        
        # STEP 6: Process game logs and generate JSONL
        dataset_stats = self._process_logs_to_jsonl(log_dir, output_file, grid_size, 
                                                   prompt_format, max_games)
        
        # STEP 7: Generate metadata
        metadata = {
            "grid_size": grid_size,
            "algorithm": algorithm,
            "prompt_format": prompt_format,
            "timestamp": timestamp,
            "source_log_dir": log_dir,
            "output_file": output_file,
            "dataset_stats": dataset_stats,
            "generator_version": "v0.04_grid_aware_language",
            "architecture_compliance": "grid-size-N directory structure"
        }
        
        # Save metadata
        metadata_file = os.path.join(dataset_dir, f"metadata_{algorithm}_{timestamp}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"JSONL dataset generated: {output_file}")
        return metadata
    
    def _extract_algorithm_from_log_dir(self, log_dir: str) -> str:
        """Extract algorithm name from log directory path."""
        log_dir_name = os.path.basename(log_dir.rstrip('/'))
        if 'heuristics' in log_dir_name:
            parts = log_dir_name.split('_')
            if len(parts) >= 1 and 'heuristics' in parts[0]:
                return parts[0].replace('heuristics', '').strip('-')
        return "unknown"
    
    def _process_logs_to_jsonl(self, log_dir: str, output_file: str, grid_size: int,
                              prompt_format: str, max_games: Optional[int]) -> Dict[str, Any]:
        """Process game logs and generate JSONL with grid-size-aware language."""
        json_files = sorted(Path(log_dir).glob("game_*.json"))
        
        if max_games:
            json_files = json_files[:max_games]
        
        if not json_files:
            raise ValueError(f"No game JSON files found in {log_dir}")
        
        total_pairs = 0
        total_games = 0
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for json_file in json_files:
                try:
                    game_pairs = self._process_single_game_file(json_file, grid_size, prompt_format)
                    
                    for pair in game_pairs:
                        jsonl_file.write(json.dumps(pair, ensure_ascii=False) + '\n')
                        total_pairs += 1
                    
                    total_games += 1
                    
                    if self.verbose and total_games % 10 == 0:
                        logger.debug(f"Processed {total_games} games, {total_pairs} prompt-completion pairs")
                        
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {e}")
                    continue
        
        return {
            "total_games": total_games,
            "total_prompt_completion_pairs": total_pairs,
            "grid_size": grid_size,
            "prompt_format": prompt_format
        }
    
    def _process_single_game_file(self, json_file: Path, grid_size: int, 
                                 prompt_format: str) -> List[Dict[str, Any]]:
        """Process a single game file into JSONL pairs."""
        with open(json_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        game_id = json_file.stem.replace('game_', '')
        pairs = []
        
        # Extract game data
        apple_positions = game_data.get('detailed_history', {}).get('apple_positions', [])
        moves = game_data.get('detailed_history', {}).get('moves', [])
        explanations = game_data.get('detailed_history', {}).get('move_explanations', [])
        
        if not apple_positions or not moves:
            return pairs
        
        # Process each move with explanation
        for step, move in enumerate(moves):
            if step >= len(apple_positions):
                break
            
            # Get current state data
            apple_position = apple_positions[step]
            apple_x = int(apple_position.get("x", 0))
            apple_y = int(apple_position.get("y", 0))
            
            # Calculate head position (simplified estimation)
            head_x = step % grid_size
            head_y = step // grid_size % grid_size
            snake_length = min(3 + step // 10, grid_size * 2)
            
            # Generate prompt based on format
            prompt = self._generate_prompt(head_x, head_y, apple_x, apple_y, 
                                         snake_length, step, grid_size, prompt_format)
            
            # Generate completion with explanation
            explanation = explanations[step] if step < len(explanations) else ""
            
            if explanation:
                completion = f"{move}. {explanation}"
            else:
                completion = f"{move}. Moving towards apple at grid position ({apple_x}, {apple_y})."
            
            # Create prompt-completion pair
            pair = {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "game_id": game_id,
                    "step": step,
                    "grid_size": grid_size,
                    "algorithm": self._extract_algorithm_from_log_dir(str(json_file.parent))
                }
            }
            
            pairs.append(pair)
        
        return pairs
    
    def _generate_prompt(self, head_x: int, head_y: int, apple_x: int, apple_y: int,
                        snake_length: int, step: int, grid_size: int, prompt_format: str) -> str:
        """Generate prompt based on format and grid size."""
        
        if prompt_format == "simple":
            return f"Snake head at ({head_x}, {head_y}), apple at ({apple_x}, {apple_y}). What move?"
        elif prompt_format == "instruction":
            return f"You are a snake game AI on a {grid_size}x{grid_size} grid. Snake head: ({head_x}, {head_y}), Apple: ({apple_x}, {apple_y}). Choose the best move:"
        else:  # detailed
            return f"Game state on {grid_size}x{grid_size} grid: Snake (length {snake_length}) at ({head_x}, {head_y}), apple at ({apple_x}, {apple_y}), step {step}. Choose optimal move:"


def main():
    """Main entry point for JSONL dataset generation."""
    parser = argparse.ArgumentParser(description="Generate JSONL datasets with grid size awareness")
    parser.add_argument("--log-dir", required=True, help="Directory containing game JSON logs")
    parser.add_argument("--output-dir", help="Base output directory (default: logs/extensions/datasets)")
    parser.add_argument("--algorithm", help="Algorithm name for filename")
    parser.add_argument("--prompt-format", choices=["simple", "detailed", "instruction"], 
                       default="detailed", help="Prompt format for LLM fine-tuning")
    parser.add_argument("--max-games", type=int, help="Maximum number of games to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    generator = HeuristicJSONLGenerator(verbose=args.verbose)
    
    try:
        metadata = generator.generate_dataset(
            log_dir=args.log_dir,
            output_base_dir=args.output_dir,
            algorithm=args.algorithm,
            prompt_format=args.prompt_format,
            max_games=args.max_games
        )
        
        print("✅ JSONL dataset generated successfully!")
        print(f"📁 Grid size: {metadata['grid_size']}")
        print(f"📁 Output directory: {os.path.dirname(metadata['output_file'])}")
        print(f"📄 Dataset file: {os.path.basename(metadata['output_file'])}")
        print(f"📊 Statistics: {metadata['dataset_stats']}")
        
    except Exception as e:
        logger.error(f"JSONL dataset generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
