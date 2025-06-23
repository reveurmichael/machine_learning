#!/usr/bin/env python3
"""
Unified Dataset Generator CLI for Heuristic Snake Agents

CRITICAL ARCHITECTURAL ENFORCEMENT: GRID SIZE BASED ORGANIZATION
===============================================================

This CLI enforces the fundamental rule that all datasets MUST be organized by 
grid size in the structure: ./logs/extensions/datasets/grid-size-N/

This ensures proper separation by game complexity and consistent organization.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the grid-aware generators
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heuristic_csv_generator import HeuristicCSVGenerator, GridSizeDetector, DatasetDirectoryManager
from heuristic_jsonl_generator import HeuristicJSONLGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGeneratorOrchestrator:
    """Grid-Aware Dataset Generation Orchestrator"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.csv_generator = HeuristicCSVGenerator(verbose=verbose)
        self.jsonl_generator = HeuristicJSONLGenerator(verbose=verbose)
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def generate_datasets(self, format_type: str, log_dir: str = None, 
                         output_base_dir: str = None, algorithm: str = None,
                         prompt_format: str = "detailed", max_games: Optional[int] = None,
                         all_algorithms: bool = False) -> Dict[str, Any]:
        """Generate datasets with grid size awareness and proper organization."""
        logger.info("🎯 Starting grid-aware dataset generation...")
        logger.info("📏 Enforcing grid-size-N directory structure compliance")
        
        if all_algorithms:
            return self._generate_for_all_algorithms(format_type, output_base_dir, 
                                                   prompt_format, max_games)
        elif log_dir:
            return self._generate_for_single_log_dir(format_type, log_dir, output_base_dir,
                                                   algorithm, prompt_format, max_games)
        else:
            raise ValueError("Must specify either --log-dir or --all-algorithms")
    
    def _generate_for_single_log_dir(self, format_type: str, log_dir: str, 
                                   output_base_dir: str, algorithm: str,
                                   prompt_format: str, max_games: Optional[int]) -> Dict[str, Any]:
        """Generate datasets for a single log directory with grid awareness."""
        
        # STEP 1: Detect grid size for architectural compliance
        grid_size = GridSizeDetector.detect_grid_size_from_log_directory(log_dir)
        logger.info(f"📏 Detected grid size: {grid_size}")
        
        # STEP 2: Ensure grid-size-N directory exists
        if output_base_dir is None:
            dataset_dir = DatasetDirectoryManager.ensure_grid_size_directory_exists(grid_size)
        else:
            dataset_dir = os.path.join(output_base_dir, f"grid-size-{grid_size}")
            os.makedirs(dataset_dir, exist_ok=True)
        
        logger.info(f"📁 Using grid-compliant directory: {dataset_dir}")
        
        results = {
            "grid_size": grid_size,
            "dataset_directory": dataset_dir,
            "architecture_compliance": "grid-size-N structure enforced",
            "generated_files": []
        }
        
        # STEP 3: Generate datasets based on format type
        if format_type in ["csv", "both"]:
            logger.info("📊 Generating CSV dataset...")
            csv_metadata = self.csv_generator.generate_dataset(
                log_dir=log_dir,
                output_base_dir=output_base_dir,
                algorithm=algorithm
            )
            results["csv_metadata"] = csv_metadata
            results["generated_files"].append(csv_metadata["output_file"])
            logger.info(f"✅ CSV dataset generated: {os.path.basename(csv_metadata['output_file'])}")
        
        if format_type in ["jsonl", "both"]:
            logger.info("🧠 Generating JSONL dataset...")
            jsonl_metadata = self.jsonl_generator.generate_dataset(
                log_dir=log_dir,
                output_base_dir=output_base_dir,
                algorithm=algorithm,
                prompt_format=prompt_format,
                max_games=max_games
            )
            results["jsonl_metadata"] = jsonl_metadata
            results["generated_files"].append(jsonl_metadata["output_file"])
            logger.info(f"✅ JSONL dataset generated: {os.path.basename(jsonl_metadata['output_file'])}")
        
        # STEP 4: Validate architectural compliance
        self._validate_grid_size_compliance(results)
        
        return results
    
    def _generate_for_all_algorithms(self, format_type: str, output_base_dir: str,
                                   prompt_format: str, max_games: Optional[int]) -> Dict[str, Any]:
        """Generate datasets for all discovered algorithms with grid awareness."""
        
        # Discover all heuristic log directories
        log_dirs = self._discover_heuristic_log_directories()
        
        if not log_dirs:
            raise ValueError("No heuristic log directories found in logs/extensions/")
        
        logger.info(f"🔍 Discovered {len(log_dirs)} heuristic log directories")
        
        results = {
            "total_algorithms": len(log_dirs),
            "algorithm_results": {},
            "grid_size_summary": {},
            "architecture_compliance": "grid-size-N structure enforced for all algorithms"
        }
        
        for log_dir in log_dirs:
            algorithm = self._extract_algorithm_from_path(log_dir)
            logger.info(f"\n🚀 Processing algorithm: {algorithm}")
            
            try:
                algo_result = self._generate_for_single_log_dir(
                    format_type, log_dir, output_base_dir, algorithm, 
                    prompt_format, max_games
                )
                results["algorithm_results"][algorithm] = algo_result
                
                # Track grid sizes
                grid_size = algo_result["grid_size"]
                if grid_size not in results["grid_size_summary"]:
                    results["grid_size_summary"][grid_size] = []
                results["grid_size_summary"][grid_size].append(algorithm)
                
            except Exception as e:
                logger.error(f"❌ Failed to process {algorithm}: {e}")
                results["algorithm_results"][algorithm] = {"error": str(e)}
        
        # Report grid size distribution
        self._report_grid_size_distribution(results["grid_size_summary"])
        
        return results
    
    def _discover_heuristic_log_directories(self) -> List[str]:
        """Discover all heuristic log directories."""
        extensions_logs = Path("logs/extensions")
        if not extensions_logs.exists():
            return []
        
        heuristic_dirs = []
        for log_dir in extensions_logs.iterdir():
            if log_dir.is_dir() and "heuristics" in log_dir.name.lower():
                # Check if it contains game JSON files
                if list(log_dir.glob("game_*.json")):
                    heuristic_dirs.append(str(log_dir))
        
        return sorted(heuristic_dirs)
    
    def _extract_algorithm_from_path(self, log_path: str) -> str:
        """Extract algorithm name from log directory path."""
        log_name = os.path.basename(log_path)
        if "heuristics" in log_name.lower():
            # Extract algorithm from names like "heuristicsbfs_timestamp"
            parts = log_name.lower().split('_')
            if parts and 'heuristics' in parts[0]:
                return parts[0].replace('heuristics', '').upper() or "UNKNOWN"
        return "UNKNOWN"
    
    def _validate_grid_size_compliance(self, results: Dict[str, Any]) -> None:
        """Validate that generated datasets comply with grid-size-N structure."""
        grid_size = results["grid_size"]
        dataset_dir = results["dataset_directory"]
        
        # Check directory name compliance
        expected_pattern = f"grid-size-{grid_size}"
        if expected_pattern not in dataset_dir:
            raise ValueError(f"Directory structure violation: {dataset_dir} does not contain {expected_pattern}")
        
        # Verify generated files are in correct location
        for file_path in results["generated_files"]:
            file_dir = os.path.dirname(file_path)
            if expected_pattern not in file_dir:
                raise ValueError(f"File placement violation: {file_path} not in grid-size-{grid_size} directory")
        
        logger.info(f"✅ Grid size compliance validated: All files properly organized in grid-size-{grid_size}/")
    
    def _report_grid_size_distribution(self, grid_summary: Dict[int, List[str]]) -> None:
        """Report the distribution of algorithms across different grid sizes."""
        logger.info("\n📊 Grid Size Distribution Report:")
        logger.info("=" * 50)
        
        for grid_size in sorted(grid_summary.keys()):
            algorithms = grid_summary[grid_size]
            logger.info(f"📏 Grid Size {grid_size}x{grid_size}: {len(algorithms)} algorithms")
            for algo in algorithms:
                logger.info(f"    • {algo}")
        
        logger.info("=" * 50)
        logger.info(f"🎯 Total grid sizes discovered: {len(grid_summary)}")
        logger.info("📁 All datasets properly organized in grid-size-N directories")


def main():
    """Main entry point enforcing grid-size-N architecture."""
    parser = argparse.ArgumentParser(description="Generate grid-aware datasets for heuristic snake agents")
    
    parser.add_argument("format", choices=["csv", "jsonl", "both"], help="Dataset format to generate")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--log-dir", help="Specific log directory to process")
    input_group.add_argument("--all-algorithms", action="store_true", help="Process all discovered heuristic algorithms")
    
    # Output options
    parser.add_argument("--output-dir", help="Base output directory (default: logs/extensions/datasets)")
    parser.add_argument("--algorithm", help="Algorithm name override")
    
    # JSONL specific options
    parser.add_argument("--prompt-format", choices=["simple", "detailed", "instruction"], default="detailed", help="Prompt format for JSONL datasets")
    parser.add_argument("--max-games", type=int, help="Maximum games to process per algorithm")
    
    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DatasetGeneratorOrchestrator(verbose=args.verbose)
    
    try:
        # Generate datasets with grid awareness
        results = orchestrator.generate_datasets(
            format_type=args.format,
            log_dir=args.log_dir,
            output_base_dir=args.output_dir,
            algorithm=args.algorithm,
            prompt_format=args.prompt_format,
            max_games=args.max_games,
            all_algorithms=args.all_algorithms
        )
        
        # Report success
        print("\n" + "=" * 60)
        print("🎉 DATASET GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if "grid_size" in results:
            print(f"📏 Grid Size: {results['grid_size']}x{results['grid_size']}")
            print(f"📁 Dataset Directory: {results['dataset_directory']}")
            print(f"�� Generated Files: {len(results['generated_files'])}")
            for file_path in results['generated_files']:
                print(f"    • {os.path.basename(file_path)}")
        else:
            print(f"🚀 Processed Algorithms: {results['total_algorithms']}")
            print(f"📊 Grid Size Distribution: {len(results['grid_size_summary'])} different grid sizes")
        
        print(f"✅ Architecture Compliance: {results['architecture_compliance']}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Dataset generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
