"""
File Management System with OOP Architecture and Design Patterns.

This module implements a comprehensive file management system using multiple
design patterns for educational and practical purposes:

1. **Singleton Pattern**: Ensures single instance for thread-safe file operations
2. **Template Method Pattern**: BaseFileManager defines algorithm structure
3. **Strategy Pattern**: Different implementations for different task types
4. **Factory Method Pattern**: Subclasses can override file creation logic

The architecture supports both generic file operations and Task-0 specific
functionality while maintaining clean separation of concerns.

=== SINGLE SOURCE OF TRUTH FOR FILE OPERATIONS ===
This module provides the canonical interface for ALL file operations across tasks.
BaseFileManager handles universal file operations (JSON loading/saving, directory management).
FileManager extends it with LLM-specific operations (prompt/response saving).

UNIVERSAL FILE OPERATIONS (Tasks 0-5):
- JSON loading/saving with error handling
- Directory discovery and validation
- Session metadata extraction  
- Consistent file naming conventions (game_N.json, summary.json)

LLM-SPECIFIC OPERATIONS (Task-0 only):
- Prompt/response file management
- LLM-specific directory structure
- Token usage tracking in files

=== ELEGANT JSON SCHEMA HANDLING ===
This module ensures PERFECT JSON schema consistency:

1. **Schema Validation**: All JSON files follow identical structure for shared fields
2. **Type Safety**: Consistent data types across all tasks (int, str, list, dict)
3. **Error Recovery**: Graceful handling of corrupted/missing JSON files  
4. **Backwards Compatibility**: Fixed schema ensures old files remain readable

=== FILE NAMING CONVENTIONS (Single Source of Truth) ===
- `game_N.json`: Individual game data (N = 1, 2, 3, ...)
- `summary.json`: Session aggregated statistics
- `game_N_round_M_prompt.txt`: LLM prompts (Task-0 only)
- `game_N_round_M_raw_response.txt`: LLM responses (Task-0 only)

=== TASK INHERITANCE EXAMPLES ===
```python
# Task-0 (LLM): Full FileManager with LLM file operations
file_manager = FileManager()
file_manager.save_prompt(prompt, game_num, round_num)  # LLM-specific

# Task-1 (Heuristics): Uses BaseFileManager only
file_manager = BaseFileManager()  
file_manager.save_game_json(game_data, game_num)  # Universal operation

# Task-2 (RL): Could extend BaseFileManager for RL-specific files
class RLFileManager(BaseFileManager):
    def save_episode_data(self, episode_data, episode_num): ...
```

=== JSON LOADING GUARANTEE ===
All JSON loading operations guarantee:
- Consistent field names across tasks
- Identical data types for shared fields
- Graceful fallbacks for missing optional fields
- Error logging without crashes
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
from abc import ABC, ABCMeta, abstractmethod
from config.game_constants import PROMPTS_DIR_NAME, RESPONSES_DIR_NAME
from llm.log_utils import get_llm_directories
from utils.path_utils import get_default_logs_root, get_summary_json_filename
from utils.singleton_utils import SingletonABCMeta

__all__ = [
    "BaseFileManager",
    "FileManager",
]


class BaseFileManager(ABC, metaclass=SingletonABCMeta):
    """
    Abstract base class for file management operations using Template Method pattern.
    
    This class defines the skeleton of file management algorithms while allowing
    subclasses to override specific steps. It implements the Singleton pattern
    to ensure thread-safe file operations across the entire application.
    
    Design Patterns Implemented:
    1. **Singleton Pattern**: Single instance for thread-safe operations
    2. **Template Method Pattern**: Defines algorithm structure, subclasses fill details
    3. **Strategy Pattern**: Different file naming/organization strategies per task
    
    The class provides a foundation for all file operations while ensuring:
    - Thread safety through singleton implementation
    - Consistent API across different task types
    - Extensibility for future tasks through inheritance
    - Separation of concerns between generic and task-specific operations
    """
    
    def __init__(self):
        """
        Initialize the singleton file manager instance.
        
        Note: Due to singleton pattern, this will only execute once
        per class, regardless of how many times the class is instantiated.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_manager()
    
    def _setup_manager(self) -> None:
        """
        Setup method called only once during singleton initialization.
        Override in subclasses for specific setup requirements.
        """
        pass
    
    # Template Method Pattern: Define the algorithm skeleton
    def process_log_directory(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Template method for processing a log directory.
        
        This method defines the algorithm skeleton for processing log directories:
        1. Validate directory
        2. Load metadata
        3. Process files
        4. Generate summary
        
        Subclasses can override individual steps while maintaining the overall flow.
        
        Design Pattern: **Template Method Pattern**
        Purpose: Define algorithm structure while allowing customization of steps.
        """
        # Step 1: Validate (concrete method)
        if not self._validate_directory(log_dir):
            return {}
        
        # Step 2: Load metadata (hook method - can be overridden)
        metadata = self._load_directory_metadata(log_dir)
        
        # Step 3: Process files (abstract method - must be implemented)
        file_data = self._process_directory_files(log_dir)
        
        # Step 4: Generate summary (concrete method with hooks)
        return self._generate_directory_summary(metadata, file_data)
    
    def _validate_directory(self, log_dir: Union[str, Path]) -> bool:
        """Concrete method: Validate directory exists and is accessible."""
        return Path(log_dir).is_dir()
    
    def _load_directory_metadata(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        """Hook method: Load directory metadata. Override for task-specific metadata."""
        return {"directory": str(log_dir), "type": "generic"}
    
    @abstractmethod
    def _process_directory_files(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        """Abstract method: Process files in directory. Must be implemented by subclasses."""
        pass
    
    def _generate_directory_summary(self, metadata: Dict[str, Any], file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Concrete method: Generate directory summary with extension points."""
        summary = {**metadata, **file_data}
        # Hook for subclasses to add specific summary data
        summary.update(self._add_task_specific_summary(summary))
        return summary
    
    def _add_task_specific_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Hook method: Add task-specific summary data. Override in subclasses."""
        return {}
    
    # Core file operations (concrete methods)
    def get_next_game_number(self, log_dir: Union[str, Path]) -> int:
        """Determine the next game number to start from."""
        log_path = Path(log_dir)
        if not log_path.exists():
            return 1

        game_numbers = []
        for f in log_path.glob("game_*.json"):
            match = re.search(r"game_(\d+)\.json", f.name)
            if match:
                game_numbers.append(int(match.group(1)))

        return max(game_numbers) + 1 if game_numbers else 1
    
    def load_summary_data(self, folder_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load summary data from folder's summary.json file."""
        summary_path = Path(folder_path) / get_summary_json_filename()
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (IOError, json.JSONDecodeError):
            return None
    
    def load_game_data(self, folder_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
        """Load all game data from folder's game_*.json files."""
        games = {}
        for file in os.listdir(str(folder_path)):
            if file.startswith("game_") and file.endswith(".json"):
                try:
                    with open(os.path.join(str(folder_path), file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    num = int(file.replace("game_", "").replace(".json", ""))
                    games[num] = data
                except Exception:
                    continue
        return games
    
    def get_folder_display_name(self, path: Union[str, Path]) -> str:
        """Get human-readable display name for folder."""
        folder_name = Path(path).name
        
        # Parse timestamp if present
        if '_' in folder_name:
            parts = folder_name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) >= 8:
                timestamp = parts[-1]
                name_part = '_'.join(parts[:-1])
                
                if len(timestamp) >= 14:  # YYYYMMDD_HHMMSS
                    try:
                        formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
                        return f"{name_part} ({formatted_time})"
                    except (ValueError, IndexError):
                        pass
        
        return folder_name
    
    def get_game_json_filename(self, game_number: int) -> str:
        """Generate standardized game JSON filename."""
        return f"game_{game_number}.json"
    
    def join_log_path(self, log_dir: Union[str, Path], filename: str) -> str:
        """Join log directory path with filename."""
        return str(Path(log_dir).joinpath(filename).resolve())
    
    def get_total_games(self, log_dir: str) -> int:
        """Get total number of games in log directory."""
        summary_file = os.path.join(log_dir, get_summary_json_filename())
        if os.path.isfile(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total = int(data.get("game_statistics", {}).get("total_games", 0))
                if total > 0:
                    return total
            except Exception:
                pass  # fall back to disk scan below

        # Disk fallback – count game_*.json files
        game_files = glob.glob(os.path.join(log_dir, "game_*.json"))
        return max(1, len(game_files))
    
    def find_valid_log_folders(self, root_dir: str = None, max_depth: int = 4) -> List[Path]:
        """
        Finds all valid log folders.
        
        Base implementation only checks for summary.json.
        Subclasses can override to add additional validation requirements.

        Args:
            root_dir: The root directory to start the search from (defaults to logs dir).
            max_depth: The maximum depth to search for log folders.

        Returns:
            A list of Path objects for each valid log folder.
        """
        if root_dir is None:
            root_dir = get_default_logs_root()
        root = Path(root_dir)
        if not root.exists():
            return []

        folders = [
            p.parent
            for p in root.glob(f"**/{'*/' * (max_depth - 1)}{get_summary_json_filename()}")
            if p.is_file()
        ]
        return sorted(folders)
    
    def extract_game_summary(self, summary_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Extracts basic summary information from a summary.json file.
        
        Base implementation provides minimal extraction.
        Subclasses can override to extract task-specific information.

        Args:
            summary_file: The path to the summary.json file.

        Returns:
            A dictionary containing summary information.
        """
        summary_path = Path(summary_file)
        if not summary_path.exists():
            return {"error": "Summary file not found."}

        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            game_count = data.get("game_count", 0)
            total_score = data.get("total_score", 0)

            return {
                "date": data.get("date", "Unknown"),
                "game_count": game_count,
                "total_score": total_score,
                "total_steps": data.get("total_steps", 0),
                "avg_score": total_score / max(1, game_count),
            }
        except (json.JSONDecodeError, IOError) as e:
            return {"error": f"Failed to read or parse summary file: {e}"}


class FileManager(BaseFileManager):
    """
    Task-0 specific file manager implementing comprehensive LLM game file operations.
    
    This class extends BaseFileManager to provide Task-0 (LLM Snake Game) specific
    functionality while maintaining the singleton pattern for thread safety.
    
    Design Patterns Implemented:
    1. **Singleton Pattern** (inherited): Thread-safe single instance
    2. **Template Method Pattern** (inherited): Uses base algorithm structure
    3. **Strategy Pattern**: Task-0 specific file organization strategy
    4. **Factory Method Pattern**: Creates Task-0 specific file structures
    
    Task-0 Specific Features:
    - LLM prompt/response file management
    - Token statistics and timing data
    - Experiment configuration handling
    - Game state serialization with LLM metadata
    
    Thread Safety: Guaranteed through singleton metaclass implementation.
    """
    
    def _setup_manager(self) -> None:
        """Setup Task-0 specific file manager configuration."""
        super()._setup_manager()
        self._task_type = "llm_snake_game"
        self._required_directories = [PROMPTS_DIR_NAME, RESPONSES_DIR_NAME]
    
    def _process_directory_files(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Process Task-0 specific files in log directory.
        
        Implementation of abstract method from BaseFileManager.
        Processes LLM-specific files including prompts, responses, and game data.
        """
        directory_path = Path(log_dir)
        
        # Count different file types
        game_files = list(directory_path.glob("game_*.json"))
        prompts_dir, responses_dir = get_llm_directories(directory_path)
        prompt_files = list(prompts_dir.glob("*.txt")) if prompts_dir.exists() else []
        response_files = list(responses_dir.glob("*.txt")) if responses_dir.exists() else []
        
        return {
            "game_count": len(game_files),
            "prompt_count": len(prompt_files),
            "response_count": len(response_files),
            "has_prompts_dir": prompts_dir.exists(),
            "has_responses_dir": responses_dir.exists(),
        }
    
    def _add_task_specific_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Add Task-0 specific summary information."""
        return {
            "task_type": self._task_type,
            "llm_experiment": True,
            "file_structure_valid": summary.get("has_prompts_dir", False) and summary.get("has_responses_dir", False)
        }
    
    def find_valid_log_folders(self, root_dir: str = None, max_depth: int = 4) -> List[Path]:
        """
        Find valid Task-0 experiment folders with required LLM structure.
        
        A valid Task-0 experiment folder must contain:
        - At least one game_*.json file
        - prompts/ directory (for LLM prompts)
        - responses/ directory (for LLM responses)
        - summary.json file
        
        This method implements a validation strategy specific to Task-0 requirements.
        
        Design Pattern: **Strategy Pattern**
        Purpose: Different validation strategies for different task types.
        """
        if root_dir is None:
            root_dir = get_default_logs_root()
        valid_folders = []
        
        def _check_folder(folder_path: Path) -> bool:
            """Check if folder meets Task-0 validation criteria."""
            # Must have summary.json
            if not (folder_path / get_summary_json_filename()).exists():
                return False
            
            # Must have at least one game file
            if not any(folder_path.glob("game_*.json")):
                return False
            
            # Must have LLM-specific directories
            prompts_dir, responses_dir = get_llm_directories(folder_path)
            if not prompts_dir.is_dir():
                return False
            
            if not responses_dir.is_dir():
                return False
            
            return True
        
        def _scan_directory(current_path: Path, current_depth: int):
            """Recursively scan directory tree for valid experiments."""
            if current_depth > max_depth:
                return
            
            try:
                for item in current_path.iterdir():
                    if item.is_dir():
                        if _check_folder(item):
                            valid_folders.append(item)
                        else:
                            _scan_directory(item, current_depth + 1)
            except (PermissionError, OSError):
                pass
        
        root_path = Path(root_dir)
        if root_path.exists():
            _scan_directory(root_path, 0)
        
        return sorted(valid_folders)
    
    def extract_game_summary(self, summary_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract comprehensive game summary with LLM-specific metrics.
        
        This method processes Task-0 summary files to extract:
        - Basic game statistics
        - LLM provider and model information
        - Token usage statistics
        - Response time metrics
        - Performance indicators
        
        Returns a normalized dictionary suitable for dashboard display.
        """
        summary = {}
        
        try:
            summary_path = Path(summary_file)
            if not summary_path.exists():
                return summary
                
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            
            # Extract basic stats
            summary['date'] = data.get('date', 'Unknown')
            summary['game_count'] = data.get('game_count', 0)
            summary['total_score'] = data.get('total_score', 0)
            summary['total_steps'] = data.get('total_steps', 0)
            summary['avg_score'] = summary['total_score'] / max(1, summary['game_count'])
            summary['avg_steps'] = summary['total_steps'] / max(1, summary['game_count'])
            
            # Extract LLM information
            if 'primary_llm' in data:
                llm_info = data['primary_llm']
                summary['primary_provider'] = llm_info.get('provider', 'Unknown')
                summary['primary_model'] = llm_info.get('model', 'Unknown')
                
            if 'secondary_llm' in data:
                llm_info = data['secondary_llm']
                summary['secondary_provider'] = llm_info.get('provider', 'None')
                summary['secondary_model'] = llm_info.get('model', 'None')
                
            # Extract response time metrics
            if 'prompt_response_stats' in data:
                prompt_stats = data.get('prompt_response_stats', {})
                summary['avg_primary_response_time'] = prompt_stats.get('avg_primary_response_time', 0)
                summary['avg_secondary_response_time'] = prompt_stats.get('avg_secondary_response_time', 0)
                summary['min_primary_response_time'] = prompt_stats.get('min_primary_response_time', 0)
                summary['max_primary_response_time'] = prompt_stats.get('max_primary_response_time', 0)
                summary['min_secondary_response_time'] = prompt_stats.get('min_secondary_response_time', 0)
                summary['max_secondary_response_time'] = prompt_stats.get('max_secondary_response_time', 0)
            
            # Extract performance metrics
            if 'efficiency_metrics' in data:
                eff_metrics = data.get('efficiency_metrics', {})
                summary['apples_per_step'] = eff_metrics.get('apples_per_step', 0)
                summary['steps_per_game'] = eff_metrics.get('steps_per_game', 0)
                summary['valid_move_ratio'] = eff_metrics.get('valid_move_ratio', 0)
            elif 'performance_metrics' in data:
                perf_metrics = data.get('performance_metrics', {})
                summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
            
            # Extract token statistics
            if 'token_stats' in data:
                token_stats = data.get('token_stats', {})
                summary['token_stats'] = token_stats
                
        except Exception as e:
            print(f"Error extracting summary: {e}")
            
        return summary
    
    def clean_prompt_files(self, log_dir: Union[str, Path], start_game: int) -> None:
        """Clean prompt and response files for games >= start_game."""
        # Use centralized utility function
        from llm.log_utils import cleanup_game_artifacts
        cleanup_game_artifacts(log_dir, start_game)
    
    def save_to_file(
        self,
        content: str,
        directory: Union[str, Path],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save content to file with optional metadata header.
        
        This method implements a standardized file saving strategy for Task-0,
        including automatic metadata injection and proper formatting.
        
        Design Pattern: **Template Method Pattern**
        Purpose: Standardized file saving with customizable metadata formatting.
        """
        path_dir = Path(directory)
        path_dir.mkdir(parents=True, exist_ok=True)

        file_path = path_dir / filename
        
        # Format content with metadata if provided
        formatted_content = self._format_file_content(content, metadata, filename)
        
        # Write the content to the file
        file_path.write_text(formatted_content, encoding="utf-8")
        
        return str(file_path)
    
    def _format_file_content(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]], 
        filename: str
    ) -> str:
        """
        Format file content with metadata header.
        
        Private method implementing the content formatting strategy.
        """
        if not metadata:
            return content
        
        # Add timestamp if not provided
        if 'timestamp' not in metadata and 'Timestamp' not in metadata:
            metadata['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format metadata as key-value pairs
        formatted_content = ""
        for key, value in metadata.items():
            # Skip lowercase timestamp as we prefer the capitalized version
            if key == 'timestamp' and 'Timestamp' in metadata:
                continue
            formatted_content += f"{key}: {value}\n"
        
        # Add section header based on file type
        if "prompt" in filename.lower():
            if "parser" in filename.lower():
                formatted_content += "\n\n========== SECONDARY LLM PROMPT ==========\n\n"
            else:
                formatted_content += "\n\n========== PRIMARY LLM PROMPT ==========\n\n"
        elif "response" in filename.lower():
            if "parsed" in filename.lower():
                formatted_content += "\n\n========== SECONDARY LLM RESPONSE ==========\n\n"
            elif "raw" in filename.lower():
                formatted_content += "\n\n========== PRIMARY LLM RESPONSE (GAME STRATEGY) ==========\n\n"
        
        # Append the main content
        formatted_content += content
        
        return formatted_content
    
    def get_prompt_filename(
        self,
        game_number: int,
        round_number: int,
        file_type: str = "prompt",
    ) -> str:
        """
        Generate standardized prompt/response filename for Task-0.
        
        Design Pattern: **Factory Method Pattern**
        Purpose: Centralized filename generation with consistent naming strategy.
        """
        return f"game_{game_number}_round_{round_number}_{file_type}.txt" 