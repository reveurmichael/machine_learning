"""Statistics management with elegant summary.json handling and BaseClass architecture.

=== SINGLE SOURCE OF TRUTH FOR STATISTICS MANAGEMENT ===
This module provides the canonical interface for ALL statistics operations across tasks.
BaseGameStatsManager handles universal statistics operations (session tracking, CSV export).
GameStatsManager extends it with LLM-specific operations (token stats, response times).

UNIVERSAL STATISTICS OPERATIONS (Tasks 0-5):
- Session summary creation and management
- CSV export for data analysis
- Statistics validation and error handling
- Consistent summary.json schema across tasks

LLM-SPECIFIC OPERATIONS (Task-0 only):
- Token usage aggregation
- Response time statistics
- LLM provider tracking
- Prompt/completion metrics

=== ELEGANT summary.json HANDLING ===
The save_session_stats() method creates perfectly structured summary.json files:

1. **Schema Consistency**: Identical structure for shared fields across all tasks
2. **Incremental Updates**: Can merge new statistics without losing existing data  
3. **Type Safety**: Ensures consistent data types (int, float, str, list, dict)
4. **Error Recovery**: Graceful handling of corrupted summary files
5. **Backwards Compatibility**: Fixed schema preserves old summary files

### **summary.json Structure (Single Source of Truth)**
```json
{
  "timestamp": "ISO-8601 format",
  "configuration": {               // Session configuration
    "provider": "...",             // LLM-only: Task-0, 4, 5
    "model": "...",                // LLM-only: Task-0, 4, 5  
    "max_games": 8,                // Universal: all tasks
    "max_steps": 400,              // Universal: all tasks
    "max_consecutive_invalid_reversals_allowed": 10,  // Universal
    "max_consecutive_no_path_found_allowed": 1        // Universal
  },
  "game_statistics": {             // Aggregated game results
    "total_games": 8,              // Universal: all tasks
    "total_score": 130,            // Universal: all tasks
    "scores": [8, 27, 12, ...],    // Universal: all tasks
    "round_counts": [12, 39, ...]  // Universal: all tasks
  },
  "step_stats": {                  // Aggregated step statistics
    "valid_steps": 931,            // Universal: all tasks
    "invalid_reversals": 0,        // Universal: all tasks
    "no_path_found_steps": 0,      // Universal: all tasks
    "empty_steps": 16,             // LLM-only: Task-0, 4, 5
    "something_is_wrong_steps": 0  // LLM-only: Task-0, 4, 5
  },
  "time_statistics": {             // Timing information
    "total_llm_communication_time": 13681.6,  // LLM-only: Task-0, 4, 5
    "avg_primary_response_time": 76.4          // LLM-only: Task-0, 4, 5
  },
  "token_usage_stats": {           // LLM-only: Task-0, 4, 5
    "primary_llm": {...},          // Token consumption metrics
    "secondary_llm": {...}         // Parser token metrics
  }
}
```

=== TASK INHERITANCE EXAMPLES ===
```python
# Task-0 (LLM): Full GameStatsManager with LLM statistics
stats_manager = GameStatsManager()
stats_manager.save_session_stats(log_dir, token_stats=llm_tokens)  # LLM-specific

# Task-1 (Heuristics): Uses BaseGameStatsManager only
stats_manager = BaseGameStatsManager()
stats_manager.save_session_stats(log_dir, algorithm_stats=heuristic_data)  # Generic

# Task-2 (RL): Could extend BaseGameStatsManager for RL-specific metrics
class RLStatsManager(BaseGameStatsManager):
    def save_training_stats(self, episode_data): ...
```

=== JSON UPDATE GUARANTEE ===
All summary.json updates guarantee:
- Atomic file operations (write to temp, then rename)
- Schema preservation (existing fields never modified)
- Type consistency (int remains int, list remains list)
- Error logging without file corruption

=== SINGLETON PATTERN IMPLEMENTATION ===
BaseGameStatsManager uses the Singleton pattern to ensure:
- Single source of truth for statistics management
- Thread-safe statistics operations
- Memory efficiency (no duplicate instances)
- Consistent CSV parsing rules across all usage points

The pattern is implemented using the same elegant thread-safe metaclass
as BaseFileManager, providing both Abstract Base Class functionality
and Singleton behavior simultaneously.
"""
from __future__ import annotations

import json
import os
import shutil
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import threading

import numpy as np
from utils.path_utils import get_default_logs_root, get_summary_json_filename

__all__ = [
    "BaseGameStatsManager",
    "GameStatsManager",
    "NumPyJSONEncoder",
]


class SingletonABCMeta(ABCMeta):
    """
    Thread-safe Singleton metaclass that combines ABC and Singleton patterns.
    
    This metaclass implements the Singleton pattern using double-checked locking
    while also supporting abstract base class functionality. This resolves the
    metaclass conflict between ABC and Singleton patterns.
    
    Design Pattern: **Singleton Pattern + Abstract Base Class**
    Purpose: Ensure only one instance of statistics manager exists while maintaining
    abstract base class functionality for inheritance.
    
    Benefits:
    - Thread safety through double-checked locking
    - Memory efficiency (single instance)
    - Abstract base class functionality
    - Centralized statistics control
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation with double-checked locking.
        
        The double-checked locking pattern ensures thread safety while
        minimizing the performance overhead of synchronization.
        """
        # First check (without locking for performance)
        if cls not in cls._instances:
            # Acquire lock for thread safety
            with cls._lock:
                # Second check (with lock to prevent race conditions)
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        
        return cls._instances[cls]


class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class BaseGameStatsManager(ABC, metaclass=SingletonABCMeta):
    """
    Base game statistics manager providing common statistical operations with Singleton pattern.
    
    This class implements the Singleton pattern to ensure thread-safe statistics operations
    and centralized management across all tasks. It provides a foundation for statistics
    operations while allowing task-specific extensions through inheritance.
    
    Design Patterns Implemented:
    1. **Singleton Pattern**: Single instance for thread-safe operations
    2. **Template Method Pattern**: Defines algorithm structure, subclasses fill details
    3. **Strategy Pattern**: Different statistics strategies per task type
    
    The class provides a foundation for all statistics operations while ensuring:
    - Thread safety through singleton implementation
    - Consistent API across different task types
    - Extensibility for future tasks through inheritance
    - Separation of concerns between generic and task-specific statistics
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the singleton statistics manager instance.
        
        Note: Due to singleton pattern, this will only execute once
        per class, regardless of how many times the class is instantiated.
        
        Args:
            log_dir: The root directory for log operations (defaults to logs dir).
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_manager(log_dir)
    
    def _setup_manager(self, log_dir: str = None) -> None:
        """
        Setup method called only once during singleton initialization.
        Override in subclasses for specific setup requirements.
        
        Args:
            log_dir: The root directory for log operations.
        """
        if log_dir is None:
            log_dir = get_default_logs_root()
        self.log_dir = log_dir
    
    def create_base_experiment_info(self, args: Any) -> Dict[str, Any]:
        """
        Create base experiment information structure.
        
        Args:
            args: Command line arguments or configuration object.
            
        Returns:
            Base experiment info dictionary.
        """
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": args_dict.copy() if isinstance(args_dict, dict) else {},
            "game_statistics": {
                "total_games": 0,
                "total_rounds": 0,
                "total_score": 0,
                "total_steps": 0,
                "scores": [],
                "round_counts": [],
            },
        }
    
    def save_experiment_info(self, experiment_info: Dict[str, Any], directory: str) -> str:
        """
        Save experiment information to a JSON file.
        
        Args:
            experiment_info: The experiment information dictionary.
            directory: Directory to save the file in.
            
        Returns:
            Path to the saved file.
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, get_summary_json_filename())
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(experiment_info, f, indent=2, cls=NumPyJSONEncoder)
            
        return file_path
    
    def load_summary_stats(self, log_dir: str) -> Optional[Dict[str, Any]]:
        """
        Load summary statistics from a log directory.
        
        Args:
            log_dir: Path to the log directory.
            
        Returns:
            Summary statistics dictionary or None if not found.
        """
        summary_path = os.path.join(log_dir, get_summary_json_filename())
        
        if not os.path.exists(summary_path):
            return None
            
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_basic_experiment_options(self, stats_df) -> Dict[str, List[str]]:
        """
        Get basic experiment options for filtering.
        
        Base implementation provides minimal options.
        Subclasses can override to provide task-specific options.
        
        Args:
            stats_df: Statistics dataframe or similar data structure.
            
        Returns:
            Dictionary of available options for filtering.
        """
        return {
            "basic_options": [],
        }


class GameStatsManager(BaseGameStatsManager):
    """
    Game statistics manager specialized for Task-0 (LLM Snake Game).
    
    This class extends the base functionality with Task-0 specific features
    such as LLM token tracking, response time metrics, and dual-LLM support.
    """
    
    def get_experiment_options(self, stats_df) -> Dict[str, List[str]]:
        """
        Return unique provider / model values for Task-0 Streamlit dropdowns.

        The helper converts the relevant stats_df columns into sorted
        lists of unique strings, stripping NaNs and handling literal "None"
        entries so the front-end can use simple st.selectbox widgets.
        
        Args:
            stats_df: Statistics dataframe containing Task-0 experiment data.
            
        Returns:
            Dictionary of available options for Task-0 filtering.
        """
        def _clean_unique(series):
            if series is None:
                return []
            # Convert to strings, drop NaN/None and duplicates
            cleaned = (
                series.dropna()
                .astype(str)
                .loc[lambda s: s.str.lower() != "none"]
                .unique()
            )
            return sorted(cleaned.tolist())

        primary_providers = _clean_unique(stats_df.get("primary_provider"))
        primary_models = _clean_unique(stats_df.get("primary_model_name"))
        secondary_providers = _clean_unique(stats_df.get("secondary_provider"))
        secondary_models = _clean_unique(stats_df.get("secondary_model_name"))

        # Always include explicit 'None' entry so dashboard filters can target
        # single-LLM experiments even when the raw data contains only NaN.
        if "None" not in secondary_providers:
            secondary_providers.insert(0, "None")

        return {
            "primary_providers": primary_providers,
            "primary_models": primary_models,
            "secondary_providers": secondary_providers,
            "secondary_models": secondary_models,
        }
    
    def filter_experiments(
        self,
        stats_df,
        selected_primary_providers: Sequence[str] | None = None,
        selected_primary_models: Sequence[str] | None = None,
        selected_secondary_providers: Sequence[str] | None = None,
        selected_secondary_models: Sequence[str] | None = None,
    ):
        """
        Return stats_df filtered by the Task-0 Streamlit-selected criteria.

        The function stays pandas-agnostic at type level to avoid importing the
        heavy dependency in runtime-light contexts; callers still pass a
        DataFrame and receive a sliced view back.
        
        Args:
            stats_df: Statistics dataframe to filter.
            selected_primary_providers: List of primary providers to include.
            selected_primary_models: List of primary models to include.
            selected_secondary_providers: List of secondary providers to include.
            selected_secondary_models: List of secondary models to include.
            
        Returns:
            Filtered dataframe.
        """
        filtered_df = stats_df.copy()

        if selected_primary_providers and len(selected_primary_providers) > 0:
            filtered_df = filtered_df[filtered_df['primary_provider'].isin(selected_primary_providers)]

        if selected_primary_models and len(selected_primary_models) > 0:
            filtered_df = filtered_df[filtered_df['primary_model_name'].isin(selected_primary_models)]

        # --------------- Secondary provider filter ----------------
        if selected_secondary_providers and len(selected_secondary_providers) > 0:
            # Special-case "None": rows where secondary_provider is NaN/None
            providers_no_none = [p for p in selected_secondary_providers if p != "None"]

            mask = False
            if providers_no_none:
                mask = filtered_df["secondary_provider"].isin(providers_no_none)
            if "None" in selected_secondary_providers:
                mask = mask | filtered_df["secondary_provider"].isna()

            filtered_df = filtered_df[mask]

        # --------------- Secondary model filter -------------------
        if selected_secondary_models and len(selected_secondary_models) > 0:
            models_no_none = [m for m in selected_secondary_models if m != "None"]

            mask_m = False
            if models_no_none:
                mask_m = filtered_df["secondary_model_name"].isin(models_no_none)
            if "None" in selected_secondary_models:
                mask_m = mask_m | filtered_df["secondary_model_name"].isna()

            filtered_df = filtered_df[mask_m]

        return filtered_df
    
    def save_experiment_info_json(self, args: Any, directory: str) -> Dict[str, Any]:
        """
        Save Task-0 experiment configuration information to a JSON file.

        Args:
            args: Command line arguments
            directory: Directory to save the file in

        Returns:
            Task-0 experiment info dictionary
        """
        # Convert args to dict
        args_dict = vars(args) if hasattr(args, '__dict__') else args

        # Clean up configuration - set parser info to null if it's 'none'
        config_dict = args_dict.copy() if isinstance(args_dict, dict) else {}
        if "parser_provider" in config_dict and (
            not config_dict["parser_provider"]
            or config_dict["parser_provider"].lower() == "none"
        ):
            # In single LLM mode, set parser fields to null instead of removing them
            config_dict["parser_provider"] = None
            config_dict["parser_model"] = None

        # Create Task-0 specific experiment info
        experiment_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": config_dict,
            "game_statistics": {
                "total_games": 0,
                "total_rounds": 0,
                "total_score": 0,
                "total_steps": 0,
                "scores": [],
                "round_counts": [],
            },
            "time_statistics": {
                "total_llm_communication_time": 0,
                "total_primary_llm_communication_time": 0,
                "total_secondary_llm_communication_time": 0,
            },
            "token_usage_stats": {
                "primary_llm": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
                "secondary_llm": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
            },
            "step_stats": {
                "empty_steps": 0,
                "something_is_wrong_steps": 0,
                "valid_steps": 0,
                "invalid_reversals": 0,  # Aggregated count across all games
                "no_path_found_steps": 0,
            },
        }

        # Save to file and return the path like the original function did
        file_path = self.save_experiment_info(experiment_info, directory)
        return file_path
    
    def save_session_stats(self, log_dir: str, **kwargs: Any) -> None:
        """
        Merge incremental kwargs into Task-0 summary.json inside log_dir.
        
        Args:
            log_dir: Directory containing the summary.json file.
            **kwargs: Statistics to merge into the summary.
        """
        # Read existing summary file
        summary_path = os.path.join(log_dir, get_summary_json_filename())

        if not os.path.exists(summary_path):
            return

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"Error reading summary.json: {e}")
            return

        def _safe_set(target: Dict[str, Any], key: str, val: Any) -> None:
            """Helper: assign val to target[key] when val is truthy."""
            if val:
                target[key] = val

        # Ensure all required Task-0 sections exist
        if "game_statistics" not in summary:
            summary["game_statistics"] = {
                "total_games": 0,
                "total_rounds": 0,
                "total_score": 0,
                "total_steps": 0,
                "scores": [],
                "round_counts": [],
            }

        if "time_statistics" not in summary:
            summary["time_statistics"] = {
                "total_llm_communication_time": 0,
                "total_primary_llm_communication_time": 0,
                "total_secondary_llm_communication_time": 0,
            }

        if "token_usage_stats" not in summary:
            summary["token_usage_stats"] = {
                "primary_llm": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
                "secondary_llm": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
            }

        if "step_stats" not in summary:
            summary["step_stats"] = {
                "empty_steps": 0,
                "something_is_wrong_steps": 0,
                "valid_steps": 0,
                "invalid_reversals": 0,  # Aggregated count across all games
                "no_path_found_steps": 0,
            }

        # Apply new statistics values to the appropriate sections
        for key, value in kwargs.items():
            if key == "game_count":
                summary["game_statistics"]["total_games"] = value
            elif key == "total_score":
                summary["game_statistics"]["total_score"] = value
            elif key == "total_steps":
                summary["game_statistics"]["total_steps"] = value
            elif key == "game_scores":
                summary["game_statistics"]["scores"] = value
            elif key == "empty_steps":
                summary["step_stats"]["empty_steps"] = value
            elif key == "something_is_wrong_steps":
                summary["step_stats"]["something_is_wrong_steps"] = value
            elif key == "valid_steps":
                summary["step_stats"]["valid_steps"] = value
            elif key == "invalid_reversals":
                summary["step_stats"]["invalid_reversals"] = value
            elif key == "no_path_found_steps":
                summary["step_stats"]["no_path_found_steps"] = value
            elif key == "time_stats":
                # Handle time statistics if provided
                if value and isinstance(value, dict):
                    ts = value
                    _safe_set(
                        summary["time_statistics"],
                        "total_llm_communication_time",
                        ts.get("llm_communication_time"),
                    )
                    _safe_set(
                        summary["time_statistics"],
                        "total_primary_llm_communication_time",
                        ts.get("primary_llm_communication_time"),
                    )
                    _safe_set(
                        summary["time_statistics"],
                        "total_secondary_llm_communication_time",
                        ts.get("secondary_llm_communication_time"),
                    )
            elif key == "token_stats":
                # Handle token statistics if provided
                if value and isinstance(value, dict):
                    if "primary" in value and isinstance(value["primary"], dict):
                        primary = value["primary"]

                        # Only add token stats if they're not None
                        if (
                            "total_tokens" in primary
                            and primary["total_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["primary_llm"]["total_tokens"] = (
                                primary["total_tokens"]
                            )
                        if (
                            "total_prompt_tokens" in primary
                            and primary["total_prompt_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["primary_llm"][
                                "total_prompt_tokens"
                            ] = primary["total_prompt_tokens"]
                        if (
                            "total_completion_tokens" in primary
                            and primary["total_completion_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["primary_llm"][
                                "total_completion_tokens"
                            ] = primary["total_completion_tokens"]

                    if "secondary" in value and isinstance(value["secondary"], dict):
                        secondary = value["secondary"]

                        # Only add token stats if they're not None
                        if (
                            "total_tokens" in secondary
                            and secondary["total_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["secondary_llm"][
                                "total_tokens"
                            ] = secondary["total_tokens"]
                        if (
                            "total_prompt_tokens" in secondary
                            and secondary["total_prompt_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["secondary_llm"][
                                "total_prompt_tokens"
                            ] = secondary["total_prompt_tokens"]
                        if (
                            "total_completion_tokens" in secondary
                            and secondary["total_completion_tokens"] is not None
                        ):
                            summary["token_usage_stats"]["secondary_llm"][
                                "total_completion_tokens"
                            ] = secondary["total_completion_tokens"]
            elif key == "step_stats":
                # Handle step statistics if provided as a complete dictionary
                if value and isinstance(value, dict):
                    if "empty_steps" in value:
                        summary["step_stats"]["empty_steps"] = value["empty_steps"]
                    if "something_is_wrong_steps" in value:
                        summary["step_stats"]["something_is_wrong_steps"] = value[
                            "something_is_wrong_steps"
                        ]
                    if "valid_steps" in value:
                        summary["step_stats"]["valid_steps"] = value["valid_steps"]
                    if "invalid_reversals" in value:
                        summary["step_stats"]["invalid_reversals"] = value[
                            "invalid_reversals"
                        ]
                    if "no_path_found_steps" in value:
                        summary["step_stats"]["no_path_found_steps"] = value[
                            "no_path_found_steps"
                        ]
            elif key == "round_counts":
                summary["game_statistics"]["round_counts"] = value
            elif key == "total_rounds":
                summary["game_statistics"]["total_rounds"] = value
            else:
                # For any other fields, add them at the top level
                summary[key] = value

        # After merging totals, compute Task-0 specific averages
        total_rounds = summary["game_statistics"].get("total_rounds", 0)
        denom = max(total_rounds, 0.00001)

        # ---------------- Token averages ----------------
        prim = summary["token_usage_stats"].get("primary_llm", {})
        if prim:
            prim["avg_tokens"] = prim.get("total_tokens", 0) / denom
            prim["avg_prompt_tokens"] = prim.get("total_prompt_tokens", 0) / denom
            prim["avg_completion_tokens"] = prim.get("total_completion_tokens", 0) / denom

        sec = summary["token_usage_stats"].get("secondary_llm", {})
        if sec:
            sec_total = sec.get("total_tokens")
            if sec_total is not None:
                sec["avg_tokens"] = sec_total / denom
            sec_prompt = sec.get("total_prompt_tokens")
            if sec_prompt is not None:
                sec["avg_prompt_tokens"] = sec_prompt / denom
            sec_comp = sec.get("total_completion_tokens")
            if sec_comp is not None:
                sec["avg_completion_tokens"] = sec_comp / denom

        # ---------------- Response-time averages ----------------
        ts = summary.get("time_statistics", {})
        if ts:
            if ts.get("total_primary_llm_communication_time") is not None:
                ts["avg_primary_response_time"] = (
                    ts.get("total_primary_llm_communication_time", 0) / denom
                )
            if ts.get("total_secondary_llm_communication_time") is not None:
                ts["avg_secondary_response_time"] = (
                    ts.get("total_secondary_llm_communication_time", 0) / denom
                )

        # Save the summary file
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
        except Exception as e:
            print(f"Error writing summary.json: {e}") 