"""Tests for replay.replay_utils module."""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from replay.replay_utils import load_game_json, parse_game_data


class TestLoadGameJson:
    """Test suite for load_game_json function."""

    def test_load_game_json_success(self) -> None:
        """Test successful loading of game JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test game data
            game_data = {
                "game_number": 1,
                "score": 150,
                "moves": ["UP", "DOWN", "LEFT", "RIGHT"]
            }
            
            # Create game file
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
            
            # Test loading
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    file_path, loaded_data = load_game_json(temp_dir, 1)
                    
                    assert file_path == game_file
                    assert loaded_data == game_data

    def test_load_game_json_file_not_found(self) -> None:
        """Test loading when game JSON file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_file = os.path.join(temp_dir, "game_999.json")
            
            with patch('utils.file_utils.get_game_json_filename', return_value="game_999.json"):
                with patch('utils.file_utils.join_log_path', return_value=non_existent_file):
                    file_path, loaded_data = load_game_json(temp_dir, 999)
                    
                    assert file_path == non_existent_file
                    assert loaded_data is None

    def test_load_game_json_invalid_json(self) -> None:
        """Test loading when JSON file is malformed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed JSON file
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                f.write("{ invalid json content")
            
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    file_path, loaded_data = load_game_json(temp_dir, 1)
                    
                    assert file_path == game_file
                    assert loaded_data is None

    def test_load_game_json_empty_file(self) -> None:
        """Test loading when JSON file is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty file
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                f.write("")
            
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    file_path, loaded_data = load_game_json(temp_dir, 1)
                    
                    assert file_path == game_file
                    assert loaded_data is None

    def test_load_game_json_permission_error(self) -> None:
        """Test loading when file permissions prevent reading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            game_file = os.path.join(temp_dir, "game_1.json")
            
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    with patch('pathlib.Path.open', side_effect=PermissionError("Access denied")):
                        file_path, loaded_data = load_game_json(temp_dir, 1)
                        
                        assert file_path == game_file
                        assert loaded_data is None

    def test_load_game_json_encoding_error(self) -> None:
        """Test loading when file has encoding issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            game_file = os.path.join(temp_dir, "game_1.json")
            
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    with patch('pathlib.Path.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
                        file_path, loaded_data = load_game_json(temp_dir, 1)
                        
                        assert file_path == game_file
                        assert loaded_data is None

    def test_load_game_json_path_handling(self) -> None:
        """Test proper path handling in load_game_json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('utils.file_utils.get_game_json_filename') as mock_get_filename:
                with patch('utils.file_utils.join_log_path') as mock_join_path:
                    mock_get_filename.return_value = "game_5.json"
                    mock_join_path.return_value = "/test/path/game_5.json"
                    
                    with patch('pathlib.Path.exists', return_value=False):
                        # Direct call to load_game_json - will call the mocked functions
                        file_path, loaded_data = load_game_json(temp_dir, 5)
                        
                        mock_get_filename.assert_called_once_with(5)
                        mock_join_path.assert_called_once_with(temp_dir, "game_5.json")
                        assert file_path == "/test/path/game_5.json"
                        assert loaded_data is None


class TestParseGameData:
    """Test suite for parse_game_data function."""

    def create_valid_game_data(self) -> Dict[str, Any]:
        """Create valid game data for testing."""
        return {
            "detailed_history": {
                "apple_positions": [
                    {"x": 5, "y": 6},
                    {"x": 7, "y": 8},
                    [3, 4]  # Test both formats
                ],
                "moves": ["UP", "DOWN", "LEFT", "RIGHT"],
                "rounds_data": {
                    "round_1": {
                        "moves": ["PLANNED", "UP", "DOWN"]
                    },
                    "round_2": {
                        "moves": ["PLANNED", "LEFT"]
                    }
                }
            },
            "llm_info": {
                "primary_provider": "test_provider",
                "primary_model": "test_model",
                "parser_provider": "parser_provider",
                "parser_model": "parser_model"
            },
            "game_end_reason": "apple_eaten",
            "metadata": {
                "timestamp": "2024-01-01 10:00:00",
                "round_count": 2
            }
        }

    def test_parse_game_data_success(self) -> None:
        """Test successful parsing of valid game data."""
        game_data = self.create_valid_game_data()
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["apple_positions"] == [[5, 6], [7, 8], [3, 4]]
        assert result["moves"] == ["UP", "DOWN", "LEFT", "RIGHT"]
        assert result["planned_moves"] == ["UP", "DOWN"]  # From first round
        assert result["game_end_reason"] == "apple_eaten"
        assert result["primary_llm"] == "test_provider/test_model"
        assert result["secondary_llm"] == "parser_provider/parser_model"
        assert result["timestamp"] == "2024-01-01 10:00:00"
        assert result["round_count"] == 2
        assert result["raw"] == game_data

    def test_parse_game_data_missing_detailed_history(self) -> None:
        """Test parsing when detailed_history is missing."""
        game_data = {"score": 100}
        
        result = parse_game_data(game_data)
        
        assert result is None

    def test_parse_game_data_invalid_detailed_history(self) -> None:
        """Test parsing when detailed_history is not a dict."""
        game_data = {"detailed_history": "not_a_dict"}
        
        result = parse_game_data(game_data)
        
        assert result is None

    def test_parse_game_data_empty_apple_positions(self) -> None:
        """Test parsing when apple_positions is empty."""
        game_data = {
            "detailed_history": {
                "apple_positions": [],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is None

    def test_parse_game_data_invalid_apple_positions(self) -> None:
        """Test parsing with invalid apple position formats."""
        game_data = {
            "detailed_history": {
                "apple_positions": [
                    {"x": 5},  # Missing y
                    [1],       # Too few coordinates
                    "invalid"  # Wrong type
                ],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        # Should still work if at least one valid apple position exists
        assert result is None  # All positions are invalid

    def test_parse_game_data_mixed_apple_position_formats(self) -> None:
        """Test parsing with mixed apple position formats."""
        game_data = {
            "detailed_history": {
                "apple_positions": [
                    {"x": 5, "y": 6},    # Dict format
                    [7, 8],              # List format
                    (9, 10),             # Tuple format
                    {"x": 1},            # Invalid dict
                    [2]                  # Invalid list
                ],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["apple_positions"] == [[5, 6], [7, 8], [9, 10]]

    def test_parse_game_data_empty_moves(self) -> None:
        """Test parsing when moves array is empty."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": []
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is None

    def test_parse_game_data_missing_moves(self) -> None:
        """Test parsing when moves key is missing."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is None

    def test_parse_game_data_planned_moves_extraction(self) -> None:
        """Test planned moves extraction from rounds data."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"],
                "rounds_data": {
                    "round_1": {
                        "moves": ["PLANNED", "UP", "DOWN", "LEFT"]
                    },
                    "round_2": {
                        "moves": ["OTHER", "RIGHT"]
                    }
                }
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == ["UP", "DOWN", "LEFT"]

    def test_parse_game_data_planned_moves_no_rounds(self) -> None:
        """Test planned moves when no rounds data exists."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == []

    def test_parse_game_data_planned_moves_invalid_rounds(self) -> None:
        """Test planned moves with invalid rounds data."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"],
                "rounds_data": {
                    "round_1": {
                        "moves": "not_a_list"
                    }
                }
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == []

    def test_parse_game_data_planned_moves_short_list(self) -> None:
        """Test planned moves when moves list is too short."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"],
                "rounds_data": {
                    "round_1": {
                        "moves": ["ONLY_ONE"]  # Too short
                    }
                }
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == []

    def test_parse_game_data_llm_info_complete(self) -> None:
        """Test LLM info parsing with complete data."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "llm_info": {
                "primary_provider": "deepseek",
                "primary_model": "deepseek-reasoner",
                "parser_provider": "mistral",
                "parser_model": "mistral-7b"
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "deepseek/deepseek-reasoner"
        assert result["secondary_llm"] == "mistral/mistral-7b"

    def test_parse_game_data_llm_info_no_parser(self) -> None:
        """Test LLM info parsing without parser."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "llm_info": {
                "primary_provider": "deepseek",
                "primary_model": "deepseek-reasoner"
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "deepseek/deepseek-reasoner"
        assert result["secondary_llm"] == "None/None"

    def test_parse_game_data_llm_info_parser_none(self) -> None:
        """Test LLM info parsing with parser set to None."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "llm_info": {
                "primary_provider": "deepseek",
                "primary_model": "deepseek-reasoner",
                "parser_provider": None,
                "parser_model": None
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "deepseek/deepseek-reasoner"
        assert result["secondary_llm"] == "None/None"

    def test_parse_game_data_llm_info_parser_none_string(self) -> None:
        """Test LLM info parsing with parser set to 'none' string."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "llm_info": {
                "primary_provider": "deepseek",
                "primary_model": "deepseek-reasoner",
                "parser_provider": "none",
                "parser_model": "none"
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "deepseek/deepseek-reasoner"
        assert result["secondary_llm"] == "None/None"

    def test_parse_game_data_llm_info_missing(self) -> None:
        """Test LLM info parsing when llm_info is missing."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "Unknown/Unknown"
        assert result["secondary_llm"] == "None/None"

    def test_parse_game_data_llm_info_partial(self) -> None:
        """Test LLM info parsing with partial data."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "llm_info": {
                "primary_provider": "deepseek"
                # Missing primary_model
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["primary_llm"] == "deepseek/Unknown"
        assert result["secondary_llm"] == "None/None"

    def test_parse_game_data_metadata_extraction(self) -> None:
        """Test metadata extraction from game data."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            },
            "game_end_reason": "max_steps",
            "metadata": {
                "timestamp": "2024-01-01 12:00:00",
                "round_count": 5
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["game_end_reason"] == "max_steps"
        assert result["timestamp"] == "2024-01-01 12:00:00"
        assert result["round_count"] == 5

    def test_parse_game_data_metadata_missing(self) -> None:
        """Test metadata handling when metadata is missing."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["game_end_reason"] is None
        assert result["timestamp"] is None
        assert result["round_count"] == 0

    def test_parse_game_data_rounds_data_sorting(self) -> None:
        """Test rounds data sorting by round number."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"],
                "rounds_data": {
                    "round_10": {
                        "moves": ["SHOULD_NOT_BE_FIRST"]
                    },
                    "round_2": {
                        "moves": ["SHOULD_NOT_BE_FIRST"]
                    },
                    "round_1": {
                        "moves": ["FIRST", "UP", "DOWN"]
                    }
                }
            }
        }
        
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == ["UP", "DOWN"]

    def test_parse_game_data_exception_handling(self) -> None:
        """Test exception handling during planned moves extraction."""
        game_data = {
            "detailed_history": {
                "apple_positions": [{"x": 5, "y": 6}],
                "moves": ["UP"],
                "rounds_data": {
                    "invalid_round_key": {
                        "moves": ["FIRST", "UP"]
                    }
                }
            }
        }
        
        # Should handle the exception gracefully
        result = parse_game_data(game_data)
        
        assert result is not None
        assert result["planned_moves"] == []


class TestReplayUtilsIntegration:
    """Integration tests for replay utils functions."""

    def test_load_and_parse_workflow(self) -> None:
        """Test complete workflow of loading and parsing game data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test game data
            game_data = {
                "detailed_history": {
                    "apple_positions": [
                        {"x": 5, "y": 6},
                        {"x": 7, "y": 8}
                    ],
                    "moves": ["UP", "DOWN", "LEFT"],
                    "rounds_data": {
                        "round_1": {
                            "moves": ["PLANNED", "UP", "DOWN"]
                        }
                    }
                },
                "llm_info": {
                    "primary_provider": "test_provider",
                    "primary_model": "test_model"
                },
                "game_end_reason": "apple_eaten",
                "metadata": {
                    "timestamp": "2024-01-01 10:00:00"
                }
            }
            
            # Save to file
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
            
            # Test complete workflow
            with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                with patch('utils.file_utils.join_log_path', return_value=game_file):
                    # Load JSON
                    file_path, loaded_data = load_game_json(temp_dir, 1)
                    
                    assert loaded_data is not None
                    
                    # Parse data
                    parsed_data = parse_game_data(loaded_data)
                    
                    assert parsed_data is not None
                    assert parsed_data["apple_positions"] == [[5, 6], [7, 8]]
                    assert parsed_data["moves"] == ["UP", "DOWN", "LEFT"]
                    assert parsed_data["planned_moves"] == ["UP", "DOWN"]
                    assert parsed_data["primary_llm"] == "test_provider/test_model"
                    assert parsed_data["secondary_llm"] == "None/None"

    def test_error_recovery_workflow(self) -> None:
        """Test error recovery in the complete workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with non-existent file
            with patch('utils.file_utils.get_game_json_filename', return_value="missing.json"):
                with patch('utils.file_utils.join_log_path', return_value="/nonexistent/missing.json"):
                    file_path, loaded_data = load_game_json(temp_dir, 999)
                    
                    assert loaded_data is None
                    
                    # Should handle None gracefully
                    parsed_data = parse_game_data({})  # Empty dict instead of None
                    assert parsed_data is None

    def test_data_validation_workflow(self) -> None:
        """Test data validation throughout the workflow."""
        # Test various invalid data scenarios
        invalid_data_sets = [
            {},  # Empty
            {"detailed_history": None},  # Invalid detailed_history
            {"detailed_history": {"apple_positions": []}},  # Empty apples
            {"detailed_history": {"apple_positions": [{"x": 1, "y": 2}], "moves": []}},  # Empty moves
        ]
        
        for invalid_data in invalid_data_sets:
            result = parse_game_data(invalid_data)
            assert result is None

    def test_robust_data_handling(self) -> None:
        """Test robust handling of various data formats."""
        # Test with minimal valid data
        minimal_data = {
            "detailed_history": {
                "apple_positions": [{"x": 1, "y": 2}],
                "moves": ["UP"]
            }
        }
        
        result = parse_game_data(minimal_data)
        assert result is not None
        assert result["apple_positions"] == [[1, 2]]
        assert result["moves"] == ["UP"]
        assert result["planned_moves"] == []
        assert result["primary_llm"] == "Unknown/Unknown"
        assert result["secondary_llm"] == "None/None" 