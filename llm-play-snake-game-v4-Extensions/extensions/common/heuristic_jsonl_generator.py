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

Design Philosophy:
This generator bridges symbolic reasoning (heuristics) and neural reasoning (LLMs)
by capturing explicit algorithmic reasoning in natural language, enabling LLMs
to learn both "what" and "why" for game decisions.

v0.04 Innovation:
Unlike traditional tabular datasets, this generator captures the reasoning process
itself, creating language-grounded training data that teaches LLMs to think like
heuristic algorithms while maintaining natural language fluency.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


class HeuristicJSONLGenerator:
    """
    JSONL Dataset Generator for Heuristic Game Logs with Rich Explanations.
    
    This class generates JSONL (JSON Lines) format datasets from heuristic game logs,
    creating prompt-completion pairs with natural language explanations suitable
    for fine-tuning Large Language Models.
    
    The generator focuses on capturing the reasoning process of heuristic algorithms
    in natural language, enabling LLMs to learn algorithmic thinking patterns
    through language-grounded examples.
    
    Key Features:
    - Prompt-completion format for LLM fine-tuning
    - Rich explanations extracted from agent reasoning
    - Multiple prompt templates for different training scenarios
    - Metadata tracking for experiment reproducibility
    - Streaming-friendly JSONL format for large datasets
    
    Design Patterns:
    - Template Method: Consistent processing workflow
    - Strategy Pattern: Different prompt generation strategies
    - Factory Pattern: Dynamic prompt template creation
    """
    
    def __init__(self, prompt_format: str = "detailed", include_metadata: bool = True, verbose: bool = False):
        """
        Initialize JSONL dataset generator.
        
        Args:
            prompt_format: Prompt template format ('simple', 'detailed', 'instruction')
            include_metadata: Include additional metadata in JSONL entries
            verbose: Enable detailed logging output
        """
        self.prompt_format = prompt_format
        self.include_metadata = include_metadata
        self.verbose = verbose
        self.generated_files: List[str] = []
        
        # Prompt templates for different training scenarios
        self.prompt_templates = {
            "simple": "Head: {head}, Apple: {apple}. Move?",
            
            "detailed": ("Game state: Snake head at {head}, apple at {apple}, "
                        "snake length {length}. What is the optimal move and why?"),
            
            "instruction": ("You are an expert Snake game AI. Analyze the current state and choose the best move.\n\n"
                          "Current state:\n"
                          "- Snake head: {head}\n"
                          "- Apple position: {apple}\n" 
                          "- Snake length: {length}\n"
                          "- Grid size: {grid_size}\n\n"
                          "Provide your move decision and detailed reasoning:")
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def generate_jsonl_dataset(self, algorithm: str, log_directory: str, 
                              output_directory: str, max_games: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate JSONL dataset from heuristic game logs.
        
        Args:
            algorithm: Algorithm name (BFS, ASTAR, etc.)
            log_directory: Directory containing game log files
            output_directory: Directory to save JSONL dataset
            max_games: Maximum number of games to process
            
        Returns:
            Dictionary with generation statistics and file paths
        """
        start_time = time.time()
        logger.info(f"Generating JSONL dataset for {algorithm} from {log_directory}")
        
        # Load game logs
        game_logs = self._load_game_logs(log_directory, max_games)
        
        # Extract JSONL entries
        jsonl_entries = self._extract_jsonl_entries(game_logs, algorithm)
        
        # Write JSONL file
        output_files = self._write_jsonl_files(jsonl_entries, algorithm, output_directory)
        
        # Generate statistics
        stats = {
            "algorithm": algorithm,
            "log_directory": log_directory,
            "output_files": output_files,
            "prompt_format": self.prompt_format,
            "total_games": len(game_logs),
            "total_entries": len(jsonl_entries),
            "generation_time_seconds": round(time.time() - start_time, 2),
            "timestamp": datetime.now().isoformat(),
            "include_metadata": self.include_metadata
        }
        
        logger.info(f"JSONL generation completed: {len(jsonl_entries)} entries in {stats['generation_time_seconds']}s")
        return stats
    
    def _load_game_logs(self, log_directory: str, max_games: Optional[int]) -> List[Dict[str, Any]]:
        """Load game log files from directory."""
        log_path = Path(log_directory)
        if not log_path.exists():
            raise ValueError(f"Log directory does not exist: {log_directory}")
        
        game_files = sorted(log_path.glob("game_*.json"))
        if max_games:
            game_files = game_files[:max_games]
        
        logger.info(f"Loading {len(game_files)} game files")
        
        game_logs = []
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                    game_logs.append(game_data)
            except Exception as e:
                logger.warning(f"Failed to load {game_file}: {e}")
        
        return game_logs
    
    def _extract_jsonl_entries(self, game_logs: List[Dict[str, Any]], algorithm: str) -> List[Dict[str, Any]]:
        """Extract JSONL entries from game logs with explanations."""
        jsonl_entries = []
        
        for game_idx, game_data in enumerate(game_logs):
            if self.verbose:
                logger.debug(f"Processing game {game_idx + 1}/{len(game_logs)}")
            
            entries = self._process_single_game(game_data, algorithm)
            jsonl_entries.extend(entries)
        
        logger.info(f"Generated {len(jsonl_entries)} JSONL entries")
        return jsonl_entries
    
    def _process_single_game(self, game_data: Dict[str, Any], algorithm: str) -> List[Dict[str, Any]]:
        """Process a single game to extract JSONL entries."""
        entries = []
        
        # Extract game history
        detailed_history = game_data.get("detailed_history", {})
        moves = detailed_history.get("moves", [])
        apple_positions = detailed_history.get("apple_positions", [])
        move_explanations = detailed_history.get("move_explanations", [])
        
        # Get game metadata
        game_id = game_data.get("game_id", f"game_{hash(str(game_data)) % 10000}")
        snake_length_start = game_data.get("snake_length", 3)
        grid_size = 10  # Default grid size
        
        # Process each move that has an explanation
        for move_idx, move in enumerate(moves):
            if (move_idx < len(apple_positions) and 
                move_idx < len(move_explanations) and 
                move_explanations[move_idx]):
                
                entry = self._create_jsonl_entry(
                    game_data, move_idx, move,
                    apple_positions[move_idx],
                    move_explanations[move_idx],
                    algorithm, game_id, snake_length_start, grid_size
                )
                if entry:
                    entries.append(entry)
        
        return entries
    
    def _create_jsonl_entry(self, game_data: Dict[str, Any], move_idx: int,
                           move: str, apple_position: Dict[str, int],
                           explanation: str, algorithm: str, game_id: str,
                           snake_length_start: int, grid_size: int) -> Optional[Dict[str, Any]]:
        """Create a single JSONL entry with prompt-completion format."""
        try:
            # Estimate current snake length (grows with apples)
            current_length = snake_length_start + (move_idx // 10)  # Rough estimate
            
            # Extract position data
            apple_x = apple_position.get("x", 0)
            apple_y = apple_position.get("y", 0)
            
            # For simplicity, estimate head position
            # In practice, this would track actual head position through moves
            head_x = 5  # Default starting position
            head_y = 5
            
            # Format positions as strings
            head_pos = f"({head_x}, {head_y})"
            apple_pos = f"({apple_x}, {apple_y})"
            
            # Generate prompt based on selected format
            prompt_template = self.prompt_templates.get(self.prompt_format, self.prompt_templates["detailed"])
            prompt = prompt_template.format(
                head=head_pos,
                apple=apple_pos,
                length=current_length,
                grid_size=grid_size
            )
            
            # Create completion with move and explanation
            completion = f"{move}. {explanation}"
            
            # Basic JSONL entry
            entry = {
                "prompt": prompt,
                "completion": completion
            }
            
            # Add metadata if requested
            if self.include_metadata:
                entry.update({
                    "game_id": game_id,
                    "step": move_idx,
                    "algorithm": algorithm,
                    "move": move,
                    "explanation": explanation,
                    "apple_position": {"x": apple_x, "y": apple_y},
                    "estimated_head": {"x": head_x, "y": head_y},
                    "snake_length": current_length,
                    "prompt_format": self.prompt_format
                })
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to create JSONL entry for move {move_idx}: {e}")
            return None
    
    def _write_jsonl_files(self, jsonl_entries: List[Dict[str, Any]], algorithm: str, 
                          output_directory: str) -> List[str]:
        """Write JSONL entries to files."""
        if not jsonl_entries:
            raise ValueError("No JSONL entries to write")
        
        # Create output directory
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_filename = f"heuristic_{algorithm.lower()}_{timestamp}.jsonl"
        metadata_filename = f"metadata_{algorithm.lower()}_{timestamp}.json"
        
        jsonl_filepath = output_path / jsonl_filename
        metadata_filepath = output_path / metadata_filename
        
        # Write JSONL file
        with open(jsonl_filepath, 'w', encoding='utf-8') as jsonl_file:
            for entry in jsonl_entries:
                json.dump(entry, jsonl_file, ensure_ascii=False)
                jsonl_file.write('\n')
        
        # Write metadata file
        metadata = {
            "dataset_info": {
                "algorithm": algorithm,
                "prompt_format": self.prompt_format,
                "total_entries": len(jsonl_entries),
                "include_metadata": self.include_metadata,
                "generation_timestamp": datetime.now().isoformat()
            },
            "format_description": {
                "file_format": "JSONL (JSON Lines)",
                "entry_structure": "prompt-completion pairs",
                "purpose": "LLM fine-tuning for Snake game AI",
                "prompt_template": self.prompt_templates[self.prompt_format]
            },
            "sample_entries": jsonl_entries[:3] if len(jsonl_entries) >= 3 else jsonl_entries,
            "usage_instructions": {
                "training": "Use prompt-completion pairs for supervised fine-tuning",
                "evaluation": "Extract prompts for inference, compare completions",
                "data_loading": "Load one JSON object per line for streaming"
            }
        }
        
        with open(metadata_filepath, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)
        
        logger.info(f"JSONL file written: {jsonl_filepath}")
        logger.info(f"Metadata file written: {metadata_filepath}")
        
        output_files = [str(jsonl_filepath), str(metadata_filepath)]
        self.generated_files.extend(output_files)
        return output_files


def generate_heuristic_jsonl(algorithm: str, log_directory: str, output_directory: str,
                            max_games: Optional[int] = None, prompt_format: str = "detailed",
                            include_metadata: bool = True, verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function for generating heuristic JSONL datasets.
    
    Args:
        algorithm: Algorithm name (BFS, ASTAR, etc.)
        log_directory: Directory containing game logs
        output_directory: Output directory for JSONL file
        max_games: Maximum number of games to process
        prompt_format: Prompt template format ('simple', 'detailed', 'instruction')
        include_metadata: Include additional metadata in JSONL entries
        verbose: Enable verbose logging
        
    Returns:
        Generation statistics dictionary
        
    Example:
        stats = generate_heuristic_jsonl(
            algorithm="BFS",
            log_directory="logs/extensions/heuristicsbfs_20231201_120000/",
            output_directory="datasets/jsonl/",
            max_games=100,
            prompt_format="detailed",
            include_metadata=True,
            verbose=True
        )
    """
    generator = HeuristicJSONLGenerator(
        prompt_format=prompt_format,
        include_metadata=include_metadata,
        verbose=verbose
    )
    return generator.generate_jsonl_dataset(algorithm, log_directory, output_directory, max_games)


# Export main classes and functions
__all__ = ['HeuristicJSONLGenerator', 'generate_heuristic_jsonl'] 