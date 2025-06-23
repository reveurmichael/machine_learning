"""
Heuristic CSV Dataset Generator (v0.03 Compatible)

This module provides CSV dataset generation for heuristic snake agents, maintaining
full backward compatibility with v0.03 functionality while providing a clean,
reusable interface for supervised learning model training.

Key Features:
- Tabular data format optimized for traditional ML models
- Engineered features from game state (position, direction, danger, etc.)
- Compatible with XGBoost, LightGBM, neural networks
- Maintains v0.03 schema and functionality
- High-performance batch processing
- GRID SIZE ADAPTIVE: Automatically detects grid size and organizes datasets by grid size

Design Philosophy:
This generator focuses on numerical/categorical features suitable for traditional
supervised learning, complementing the language-rich JSONL generator for LLMs.

CRITICAL ARCHITECTURAL RULE: GRID SIZE BASED DATASET ORGANIZATION
--------------------

The grid_size should NEVER be hardcoded! Generated datasets must be stored in:
./logs/extensions/datasets/grid-size-N/

Where N is the actual grid size detected from the game logs. This ensures:
1. Datasets are properly segregated by grid complexity
2. Models can be trained on specific grid sizes
3. Multi-grid experimentation is supported
4. Future scalability to different grid sizes
5. Clear dataset organization and discovery

The grid size is automatically detected by analyzing the maximum coordinates
found in apple positions and snake positions throughout the game logs.
This ensures accuracy regardless of the actual game configuration used.

Grid Size Detection Algorithm:
1. Parse all apple positions from detailed_history
2. Parse all snake positions from moves/board states
3. Find maximum x and y coordinates
4. Grid size = max(max_x, max_y) + 1 (since coordinates are 0-based)
5. Validate grid size is reasonable (between 8 and 50)
6. Create directory structure: logs/extensions/datasets/grid-size-{N}/

This approach is resilient to:
- Configuration changes
- Different extensions using different grid sizes
- Future grid size variations
- Legacy logs with different grid sizes
"""

import json
import csv
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse

from extensions.common.dataset_directory_manager import DatasetDirectoryManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridSizeDetector:
    """
    CRITICAL COMPONENT: Grid Size Detection and Validation
    
    This class is responsible for the fundamental architectural requirement
    of detecting grid sizes from game logs and ensuring proper dataset
    organization by grid size.
    
    The grid size detection is essential because:
    1. Different experiments may use different grid sizes
    2. We need to segregate datasets by complexity level
    3. Models trained on one grid size may not work on another
    4. Future experiments will need clear grid size organization
    
    Detection Strategy:
    - Parse all coordinate data from game logs
    - Find maximum coordinates across all games
    - Validate detected grid size is reasonable
    - Handle edge cases and corrupted data gracefully
    """
    
    MIN_GRID_SIZE = 8   # Minimum reasonable grid size
    MAX_GRID_SIZE = 50  # Maximum reasonable grid size for performance
    
    @staticmethod
    def detect_grid_size_from_log_directory(log_dir: str) -> int:
        """
        Detect grid size by analyzing all game logs in a directory.
        
        This is the primary method for grid size detection. It processes
        all JSON files in the log directory to find the maximum coordinates,
        ensuring accurate grid size detection even with partial games.
        
        Args:
            log_dir: Path to log directory containing game JSON files
            
        Returns:
            int: Detected grid size
            
        Raises:
            ValueError: If grid size cannot be detected or is unreasonable
            FileNotFoundError: If log directory doesn't exist
        """
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
        
        max_x, max_y = 0, 0
        json_files = list(Path(log_dir).glob("game_*.json"))
        
        if not json_files:
            raise ValueError(f"No game JSON files found in {log_dir}")
        
        logger.info(f"Analyzing {len(json_files)} game files for grid size detection...")
        
        for json_file in json_files:
            try:
                file_max_x, file_max_y = GridSizeDetector._analyze_single_game_file(json_file)
                max_x = max(max_x, file_max_x)
                max_y = max(max_y, file_max_y)
            except Exception as e:
                logger.warning(f"Error analyzing {json_file}: {e}")
                continue
        
        if max_x == 0 and max_y == 0:
            raise ValueError("Could not detect valid coordinates from any game file")
        
        # Grid size is max coordinate + 1 (since coordinates are 0-based)
        detected_grid_size = max(max_x, max_y) + 1
        
        # Validate grid size is reasonable
        if detected_grid_size < GridSizeDetector.MIN_GRID_SIZE:
            raise ValueError(f"Detected grid size {detected_grid_size} is too small (min: {GridSizeDetector.MIN_GRID_SIZE})")
        
        if detected_grid_size > GridSizeDetector.MAX_GRID_SIZE:
            raise ValueError(f"Detected grid size {detected_grid_size} is too large (max: {GridSizeDetector.MAX_GRID_SIZE})")
        
        logger.info(f"Successfully detected grid size: {detected_grid_size}")
        return detected_grid_size
    
    @staticmethod
    def _analyze_single_game_file(json_file: Path) -> Tuple[int, int]:
        """
        Analyze a single game JSON file to extract maximum coordinates.
        
        This method handles the detailed parsing of game state data to
        extract all coordinate information from:
        - Apple positions
        - Snake positions (if available)
        - Move history coordinates
        
        Args:
            json_file: Path to game JSON file
            
        Returns:
            Tuple[int, int]: (max_x, max_y) coordinates found in the file
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        max_x, max_y = 0, 0
        
        # Extract from apple positions
        if 'detailed_history' in game_data and 'apple_positions' in game_data['detailed_history']:
            for apple_pos in game_data['detailed_history']['apple_positions']:
                # Handle both string and integer coordinates
                x = int(apple_pos.get('x', 0))
                y = int(apple_pos.get('y', 0))
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        # Extract from moves/snake positions if available
        # This is future-proofing for different log formats
        if 'detailed_history' in game_data and 'moves' in game_data['detailed_history']:
            pass
        
        return max_x, max_y


class HeuristicCSVGenerator:
    """
    Heuristic CSV Dataset Generator (v0.03+)
    
    This generator creates tabular datasets for traditional machine learning
    models while enforcing the critical grid size organization requirement.
    
    The generator automatically:
    1. Detects grid size from game logs
    2. Creates appropriate grid-size-N directory
    3. Generates features scaled to the detected grid size
    4. Saves datasets with proper naming convention
    5. Creates metadata for dataset tracking
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the CSV generator.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def generate_dataset(self, log_dir: str, output_base_dir: str = None,
                         algorithm: str = None) -> Dict[str, Any]:
        """
        Generate a CSV dataset from heuristic agent game logs.

        This method orchestrates the entire dataset generation process:
        1. Detects grid size from logs for architectural compliance.
        2. Ensures the correct grid-size-N directory exists.
        3. Determines the algorithm name.
        4. Generates a unique, descriptive filename.
        5. Processes all game logs into a single CSV file.
        6. Returns comprehensive metadata about the generated dataset.
        
        Args:
            log_dir: Path to the source log directory.
            output_base_dir: Optional base directory for output.
            algorithm: Optional algorithm name override.
            
        Returns:
            A dictionary containing metadata about the generated dataset.
        """
        logger.info(f"Starting CSV generation for log directory: {log_dir}")
        
        # STEP 1: Detect grid size for architectural compliance
        grid_size = GridSizeDetector.detect_grid_size_from_log_directory(log_dir)
        
        # STEP 2: Ensure proper directory structure using the centralized manager
        output_dir = DatasetDirectoryManager.ensure_datasets_dir(grid_size)
        
        # STEP 3: Determine algorithm name
        if algorithm is None:
            algorithm = self._extract_algorithm_from_log_dir(log_dir)
        
        # STEP 4: Generate output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tabular_{algorithm.lower()}_data_{timestamp}.csv"
        output_file = output_dir / output_filename
        
        logger.info(f"Generating CSV dataset at: {output_file}")
        
        # STEP 5: Process logs and get metadata
        csv_metadata = self._process_logs_to_csv(log_dir, str(output_file), grid_size)
        
        final_metadata = {
            "grid_size": grid_size,
            "algorithm": algorithm,
            "output_file": str(output_file),
            **csv_metadata
        }
        
        logger.info(f"✅ CSV dataset generation complete for {algorithm} (grid size: {grid_size})")
        return final_metadata

    def _extract_algorithm_from_log_dir(self, log_dir: str) -> str:
        """Extract algorithm name from log directory path."""
        log_name = Path(log_dir).name
        # Assumes format like "heuristics-bfs_20230101_120000"
        parts = log_name.split('_')[0].split('-')
        if len(parts) > 1:
            return parts[1].upper()
        return "UNKNOWN"

    def _process_logs_to_csv(self, log_dir: str, output_file: str, grid_size: int) -> Dict[str, Any]:
        """
        Process all game logs in a directory and generate a single CSV file.
        
        This method is optimized for batch processing of large numbers of
        JSON log files into a unified CSV dataset.
        
        Args:
            log_dir: Path to log directory
            output_file: Path to output CSV file
            grid_size: Detected grid size of the games
            
        Returns:
            Dict[str, Any]: Metadata about the generated dataset
        """
        start_time = datetime.now()
        
        json_files = sorted(list(Path(log_dir).glob("game_*.json")))
        
        if not json_files:
            logger.warning(f"No game JSON files found in {log_dir}")
            return {}
            
        logger.info(f"Processing {len(json_files)} game files from {log_dir}...")
        
        header = [
            'game_id', 'step', 'grid_size', 'snake_length', 
            'head_x', 'head_y', 'apple_x', 'apple_y', 'current_direction',
            'obstacle_up', 'obstacle_down', 'obstacle_left', 'obstacle_right',
            'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
            'target_move'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            total_rows = 0
            for i, json_file in enumerate(json_files):
                try:
                    game_rows = self._process_single_game_file(json_file, grid_size)
                    writer.writerows(game_rows)
                    total_rows += len(game_rows)
                except Exception as e:
                    logger.error(f"Error processing file {json_file}: {e}")
                    
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "total_games": len(json_files),
            "total_rows": total_rows,
            "processing_time_seconds": round(duration, 2),
            "csv_header": header
        }

    def _process_single_game_file(self, json_file: Path, grid_size: int) -> List[List]:
        """Process a single game JSON file into a list of CSV rows."""
        with open(json_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        game_id = os.path.basename(json_file).replace('game_', '').replace('.json', '')
        
        if 'detailed_history' not in game_data or 'moves' not in game_data['detailed_history']:
            return []
            
        moves = game_data['detailed_history']['moves']
        apple_positions = game_data['detailed_history']['apple_positions']
        
        rows = []
        snake_positions = []
        current_direction = 'RIGHT'
        
        for i, move in enumerate(moves):
            if i == 0:
                snake_positions = [[grid_size // 2, grid_size // 2]]
            
            head = list(snake_positions[0])
            if move == 'UP':
                head[1] += 1
                current_direction = 'UP'
            elif move == 'DOWN':
                head[1] -= 1
                current_direction = 'DOWN'
            elif move == 'LEFT':
                head[0] -= 1
                current_direction = 'LEFT'
            elif move == 'RIGHT':
                head[0] += 1
                current_direction = 'RIGHT'

            apple_pos = apple_positions[min(i, len(apple_positions) - 1)]
            
            row = self._create_csv_row(
                game_id, i + 1, grid_size, snake_positions, 
                apple_pos, current_direction, move
            )
            rows.append(row)
            
            snake_positions.insert(0, head)
            if head[0] == apple_pos['x'] and head[1] == apple_pos['y']:
                pass
            else:
                snake_positions.pop()
        
        return rows
    
    def _create_csv_row(self, game_id: str, step: int, grid_size: int, 
                        snake_positions: List[List[int]], apple_pos: Dict[str, int],
                        current_direction: str, target_move: str) -> List:
        """Create a single row for the CSV dataset."""
        
        head = snake_positions[0]
        snake_body = set(map(tuple, snake_positions[1:]))
        
        obstacle_up = (head[0], head[1] + 1) in snake_body or head[1] + 1 >= grid_size
        obstacle_down = (head[0], head[1] - 1) in snake_body or head[1] - 1 < 0
        obstacle_left = (head[0] - 1, head[1]) in snake_body or head[0] - 1 < 0
        obstacle_right = (head[0] + 1, head[1]) in snake_body or head[0] + 1 >= grid_size
        
        apple_dir_up = 1 if apple_pos['y'] > head[1] else 0
        apple_dir_down = 1 if apple_pos['y'] < head[1] else 0
        apple_dir_left = 1 if apple_pos['x'] < head[0] else 0
        apple_dir_right = 1 if apple_pos['x'] > head[0] else 0
        
        return [
            game_id, step, grid_size, len(snake_positions),
            head[0], head[1], apple_pos['x'], apple_pos['y'], current_direction,
            int(obstacle_up), int(obstacle_down), int(obstacle_left), int(obstacle_right),
            apple_dir_up, apple_dir_down, apple_dir_left, apple_dir_right,
            target_move
        ]


def main():
    """Main entry point for the Heuristic CSV Generator CLI."""
    parser = argparse.ArgumentParser(description="Generate CSV datasets from heuristic snake agent logs.")
    
    parser.add_argument("log_dir", help="Directory containing game JSON logs")
    parser.add_argument("--output-dir", help="Base output directory (defaults to logs/extensions/datasets)")
    parser.add_argument("--algorithm", help="Algorithm name for filename generation (e.g., bfs, astar)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    generator = HeuristicCSVGenerator(verbose=args.verbose)
    
    generator.generate_dataset(
        log_dir=args.log_dir,
        output_base_dir=args.output_dir,
        algorithm=args.algorithm
    )


if __name__ == "__main__":
    main()