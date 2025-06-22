"""
Tests for file utility functions.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from utils.file_utils import (
    extract_game_summary,
    get_next_game_number,
    save_to_file,
    load_summary_data,
    load_game_data,
    find_valid_log_folders,
    get_folder_display_name,
    ensure_directory_exists,
    get_game_log_folders,
    create_game_folder,
    save_json_safely,
    load_json_safely,
    cleanup_old_logs,
    get_file_size,
    is_file_readable,
    get_timestamp_string
)
from typing import List, Dict, Any, Optional, Tuple


class TestFileUtils:
    """Test cases for file utility functions."""

    def test_extract_game_summary_valid_file(self, temp_dir):
        """Test extracting game summary from valid file."""
        # Create a test game file
        game_data = {
            "score": 5,
            "steps": 25,
            "game_over": True,
            "llm_info": {"primary_provider": "test"}
        }
        
        filepath = os.path.join(temp_dir, "game_1.json")
        with open(filepath, 'w') as f:
            json.dump(game_data, f)
        
        summary = extract_game_summary(filepath)
        
        assert summary is not None
        assert summary["score"] == 5
        assert summary["steps"] == 25
        assert summary["game_over"] is True

    def test_extract_game_summary_invalid_file(self):
        """Test extracting game summary from non-existent file."""
        summary = extract_game_summary("non_existent_file.json")
        assert summary is None

    def test_extract_game_summary_invalid_json(self, temp_dir):
        """Test extracting game summary from file with invalid JSON."""
        filepath = os.path.join(temp_dir, "invalid.json")
        with open(filepath, 'w') as f:
            f.write("invalid json content")
        
        summary = extract_game_summary(filepath)
        assert summary is None

    def test_get_next_game_number_empty_dir(self, temp_dir):
        """Test getting next game number from empty directory."""
        next_num = get_next_game_number(temp_dir)
        assert next_num == 1

    def test_get_next_game_number_existing_files(self, temp_dir):
        """Test getting next game number with existing files."""
        # Create some game files
        for i in [1, 2, 4]:  # Skip 3 to test it handles gaps
            filepath = os.path.join(temp_dir, f"game_{i}.json")
            with open(filepath, 'w') as f:
                json.dump({"game": i}, f)
        
        next_num = get_next_game_number(temp_dir)
        assert next_num == 5

    def test_get_next_game_number_non_game_files(self, temp_dir):
        """Test getting next game number ignoring non-game files."""
        # Create some non-game files
        files = ["summary.json", "other.txt", "game_invalid.json"]
        for filename in files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("content")
        
        next_num = get_next_game_number(temp_dir)
        assert next_num == 1

    def test_save_to_file_success(self, temp_dir):
        """Test successful file saving."""
        content = "test content"
        filepath = os.path.join(temp_dir, "test.txt")
        
        success = save_to_file(content, filepath)
        
        assert success is True
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            assert f.read() == content

    def test_save_to_file_directory_creation(self, temp_dir):
        """Test file saving with directory creation."""
        content = "test content"
        subdir = os.path.join(temp_dir, "subdir")
        filepath = os.path.join(subdir, "test.txt")
        
        success = save_to_file(content, filepath)
        
        assert success is True
        assert os.path.exists(filepath)
        assert os.path.exists(subdir)

    def test_save_to_file_permission_error(self):
        """Test file saving with permission error."""
        content = "test content"
        filepath = "/root/cannot_write_here.txt"  # Assuming no write permission
        
        # This should handle the error gracefully
        success = save_to_file(content, filepath)
        assert success is False

    def test_load_summary_data_valid_file(self, temp_dir):
        """Test loading valid summary data."""
        summary_data = {
            "total_games": 5,
            "avg_score": 3.2,
            "game_summaries": []
        }
        
        filepath = os.path.join(temp_dir, "summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary_data, f)
        
        loaded_data = load_summary_data(filepath)
        
        assert loaded_data is not None
        assert loaded_data["total_games"] == 5
        assert loaded_data["avg_score"] == 3.2

    def test_load_summary_data_missing_file(self):
        """Test loading summary data from missing file."""
        loaded_data = load_summary_data("missing_file.json")
        assert loaded_data is None

    def test_load_summary_data_invalid_json(self, temp_dir):
        """Test loading summary data from invalid JSON file."""
        filepath = os.path.join(temp_dir, "invalid_summary.json")
        with open(filepath, 'w') as f:
            f.write("invalid json")
        
        loaded_data = load_summary_data(filepath)
        assert loaded_data is None

    def test_load_game_data_valid_file(self, temp_dir):
        """Test loading valid game data."""
        game_data = {
            "score": 10,
            "moves": ["UP", "RIGHT", "DOWN"],
            "game_over": True
        }
        
        filepath = os.path.join(temp_dir, "game_1.json")
        with open(filepath, 'w') as f:
            json.dump(game_data, f)
        
        loaded_data = load_game_data(filepath)
        
        assert loaded_data is not None
        assert loaded_data["score"] == 10
        assert loaded_data["moves"] == ["UP", "RIGHT", "DOWN"]

    def test_load_game_data_missing_file(self):
        """Test loading game data from missing file."""
        loaded_data = load_game_data("missing_game.json")
        assert loaded_data is None

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_find_valid_log_folders(self, mock_listdir, mock_exists):
        """Test finding valid log folders."""
        # Mock directory structure
        mock_listdir.return_value = [
            "hunyuan_20240101_120000",
            "deepseek_20240102_130000",
            "invalid_folder",
            "mistral_20240103_140000"
        ]
        mock_exists.return_value = True
        
        folders = find_valid_log_folders("logs")
        
        assert len(folders) == 3
        assert all("20240" in folder for folder in folders)

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_find_valid_log_folders_empty_dir(self, mock_listdir, mock_exists):
        """Test finding valid log folders in empty directory."""
        mock_listdir.return_value = []
        mock_exists.return_value = True
        
        folders = find_valid_log_folders("logs")
        assert folders == []

    @patch('os.path.exists')
    def test_find_valid_log_folders_missing_dir(self, mock_exists):
        """Test finding valid log folders when directory doesn't exist."""
        mock_exists.return_value = False
        
        folders = find_valid_log_folders("nonexistent_logs")
        assert folders == []

    def test_get_folder_display_name(self):
        """Test getting display name for folders."""
        test_cases = [
            ("hunyuan_20240101_120000", "hunyuan (2024-01-01 12:00:00)"),
            ("deepseek_20240615_143022", "deepseek (2024-06-15 14:30:22)"),
            ("invalid_format", "invalid_format"),
            ("provider_only", "provider_only"),
        ]
        
        for folder_name, expected in test_cases:
            result = get_folder_display_name(folder_name)
            assert result == expected

    def test_get_folder_display_name_edge_cases(self):
        """Test getting display name for edge cases."""
        edge_cases = [
            ("", ""),
            ("_", "_"),
            ("provider_", "provider_"),
            ("_20240101_120000", "_20240101_120000"),
        ]
        
        for folder_name, expected in edge_cases:
            result = get_folder_display_name(folder_name)
            assert result == expected

    def test_ensure_directory_exists_new_directory(self, temp_dir: str) -> None:
        """Test creating a new directory."""
        new_dir_path: str = os.path.join(temp_dir, "new_directory")
        
        # Directory shouldn't exist initially
        assert not os.path.exists(new_dir_path)
        
        success: bool = ensure_directory_exists(new_dir_path)
        
        assert success
        assert os.path.exists(new_dir_path)
        assert os.path.isdir(new_dir_path)

    def test_ensure_directory_exists_existing_directory(self, temp_dir: str) -> None:
        """Test with an already existing directory."""
        # temp_dir already exists
        success: bool = ensure_directory_exists(temp_dir)
        
        assert success
        assert os.path.exists(temp_dir)

    def test_ensure_directory_exists_nested_directories(self, temp_dir: str) -> None:
        """Test creating nested directories."""
        nested_path: str = os.path.join(temp_dir, "level1", "level2", "level3")
        
        success: bool = ensure_directory_exists(nested_path)
        
        assert success
        assert os.path.exists(nested_path)
        assert os.path.isdir(nested_path)

    def test_ensure_directory_exists_permission_error(self) -> None:
        """Test handling permission errors."""
        # Try to create directory in a location that requires admin rights
        restricted_path: str = "/root/restricted_directory"
        
        success: bool = ensure_directory_exists(restricted_path)
        
        # Should return False for permission errors
        assert not success

    def test_get_game_log_folders_multiple_folders(self, temp_dir: str) -> None:
        """Test getting game log folders when multiple exist."""
        # Create test log directories
        log_folders: List[str] = [
            "deepseek-chat_20240101_120000",
            "mistral-7b_20240102_130000",
            "gpt4_20240103_140000"
        ]
        
        for folder_name in log_folders:
            folder_path: str = os.path.join(temp_dir, folder_name)
            os.makedirs(folder_path)
        
        result: List[str] = get_game_log_folders(temp_dir)
        
        assert len(result) == 3
        for folder_name in log_folders:
            assert any(folder_name in path for path in result)

    def test_get_game_log_folders_empty_directory(self, temp_dir: str) -> None:
        """Test getting game log folders from empty directory."""
        result: List[str] = get_game_log_folders(temp_dir)
        
        assert result == []

    def test_get_game_log_folders_mixed_content(self, temp_dir: str) -> None:
        """Test filtering out non-log folders."""
        # Create mixed content
        os.makedirs(os.path.join(temp_dir, "deepseek-chat_20240101_120000"))
        os.makedirs(os.path.join(temp_dir, "not_a_log_folder"))
        
        with open(os.path.join(temp_dir, "regular_file.txt"), 'w') as f:
            f.write("test")
        
        result: List[str] = get_game_log_folders(temp_dir)
        
        assert len(result) == 1
        assert "deepseek-chat_20240101_120000" in result[0]

    def test_get_game_log_folders_nonexistent_directory(self) -> None:
        """Test with non-existent directory."""
        nonexistent_path: str = "/path/that/does/not/exist"
        
        result: List[str] = get_game_log_folders(nonexistent_path)
        
        assert result == []

    def test_get_next_game_number_new_session(self, temp_dir: str) -> None:
        """Test getting next game number for new session."""
        next_number: int = get_next_game_number(temp_dir)
        
        assert next_number == 1

    def test_get_next_game_number_existing_games(self, temp_dir: str) -> None:
        """Test getting next game number with existing games."""
        # Create existing game files
        existing_games: List[str] = ["game_1.json", "game_2.json", "game_5.json"]
        
        for game_file in existing_games:
            file_path: str = os.path.join(temp_dir, game_file)
            with open(file_path, 'w') as f:
                json.dump({"test": "data"}, f)
        
        next_number: int = get_next_game_number(temp_dir)
        
        # Should return the next number after the highest existing number
        assert next_number == 6

    def test_get_next_game_number_non_game_files(self, temp_dir: str) -> None:
        """Test ignoring non-game files when determining next number."""
        # Create mix of game and non-game files
        files: List[str] = [
            "game_1.json",
            "game_3.json", 
            "summary.json",
            "other_file.txt",
            "not_a_game.json"
        ]
        
        for file_name in files:
            file_path: str = os.path.join(temp_dir, file_name)
            with open(file_path, 'w') as f:
                f.write("test content")
        
        next_number: int = get_next_game_number(temp_dir)
        
        assert next_number == 4  # Next after game_3.json

    def test_create_game_folder_success(self, temp_dir: str) -> None:
        """Test successfully creating a game folder."""
        session_name: str = "test_session_20240101_120000"
        
        folder_path: Optional[str] = create_game_folder(temp_dir, session_name)
        
        assert folder_path is not None
        assert os.path.exists(folder_path)
        assert session_name in folder_path
        
        # Should also create subdirectories
        prompts_dir: str = os.path.join(folder_path, "prompts")
        responses_dir: str = os.path.join(folder_path, "responses")
        assert os.path.exists(prompts_dir)
        assert os.path.exists(responses_dir)

    def test_create_game_folder_existing_folder(self, temp_dir: str) -> None:
        """Test creating game folder when it already exists."""
        session_name: str = "existing_session_20240101_120000"
        existing_path: str = os.path.join(temp_dir, session_name)
        os.makedirs(existing_path)
        
        folder_path: Optional[str] = create_game_folder(temp_dir, session_name)
        
        # Should still return the path successfully
        assert folder_path is not None
        assert folder_path == existing_path

    def test_create_game_folder_permission_error(self) -> None:
        """Test handling permission errors when creating folder."""
        base_dir: str = "/root"  # Typically restricted
        session_name: str = "test_session"
        
        folder_path: Optional[str] = create_game_folder(base_dir, session_name)
        
        assert folder_path is None

    def test_save_json_safely_success(self, temp_dir: str) -> None:
        """Test successfully saving JSON data."""
        file_path: str = os.path.join(temp_dir, "test_data.json")
        test_data: Dict[str, Any] = {
            "game_id": 1,
            "score": 10,
            "moves": ["UP", "RIGHT", "DOWN"]
        }
        
        success: bool = save_json_safely(test_data, file_path)
        
        assert success
        assert os.path.exists(file_path)
        
        # Verify content
        with open(file_path, 'r') as f:
            loaded_data: Dict[str, Any] = json.load(f)
            assert loaded_data == test_data

    def test_save_json_safely_with_indent(self, temp_dir: str) -> None:
        """Test saving JSON with indentation."""
        file_path: str = os.path.join(temp_dir, "indented.json")
        test_data: Dict[str, Any] = {"key": "value", "number": 42}
        
        success: bool = save_json_safely(test_data, file_path, indent=2)
        
        assert success
        
        # Check that file has indentation
        with open(file_path, 'r') as f:
            content: str = f.read()
            assert "\n" in content
            assert "  " in content  # Indentation spaces

    def test_save_json_safely_invalid_path(self) -> None:
        """Test saving to invalid file path."""
        invalid_path: str = "/nonexistent/directory/file.json"
        test_data: Dict[str, Any] = {"test": "data"}
        
        success: bool = save_json_safely(test_data, invalid_path)
        
        assert not success

    def test_save_json_safely_non_serializable_data(self, temp_dir: str) -> None:
        """Test saving non-serializable data."""
        file_path: str = os.path.join(temp_dir, "invalid.json")
        
        # Create non-serializable object
        class NonSerializable:
            pass
        
        test_data: Dict[str, Any] = {"object": NonSerializable()}
        
        success: bool = save_json_safely(test_data, file_path)
        
        assert not success
        assert not os.path.exists(file_path)

    def test_load_json_safely_success(self, temp_dir: str) -> None:
        """Test successfully loading JSON data."""
        file_path: str = os.path.join(temp_dir, "test_load.json")
        test_data: Dict[str, Any] = {
            "game_id": 2,
            "score": 25,
            "completed": True
        }
        
        # Create test file
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(file_path)
        
        assert loaded_data is not None
        assert loaded_data == test_data

    def test_load_json_safely_nonexistent_file(self) -> None:
        """Test loading from non-existent file."""
        nonexistent_path: str = "/path/to/nonexistent.json"
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(nonexistent_path)
        
        assert loaded_data is None

    def test_load_json_safely_invalid_json(self, temp_dir: str) -> None:
        """Test loading invalid JSON file."""
        file_path: str = os.path.join(temp_dir, "invalid.json")
        
        # Create file with invalid JSON
        with open(file_path, 'w') as f:
            f.write("{ invalid json content")
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(file_path)
        
        assert loaded_data is None

    def test_load_json_safely_empty_file(self, temp_dir: str) -> None:
        """Test loading empty JSON file."""
        file_path: str = os.path.join(temp_dir, "empty.json")
        
        # Create empty file
        with open(file_path, 'w') as f:
            pass
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(file_path)
        
        assert loaded_data is None

    def test_cleanup_old_logs_by_count(self, temp_dir: str) -> None:
        """Test cleaning up old logs by keeping only recent ones."""
        # Create multiple log folders with different timestamps
        log_folders: List[str] = [
            "session_20240101_120000",
            "session_20240102_120000", 
            "session_20240103_120000",
            "session_20240104_120000",
            "session_20240105_120000"
        ]
        
        for folder_name in log_folders:
            folder_path: str = os.path.join(temp_dir, folder_name)
            os.makedirs(folder_path)
            # Add some test files
            with open(os.path.join(folder_path, "game_1.json"), 'w') as f:
                json.dump({"test": "data"}, f)
        
        # Keep only 3 most recent
        cleanup_old_logs(temp_dir, max_folders=3)
        
        remaining_folders: List[str] = [
            d for d in os.listdir(temp_dir) 
            if os.path.isdir(os.path.join(temp_dir, d))
        ]
        
        assert len(remaining_folders) == 3
        # Should keep the most recent ones
        assert "session_20240105_120000" in remaining_folders
        assert "session_20240104_120000" in remaining_folders
        assert "session_20240103_120000" in remaining_folders

    def test_cleanup_old_logs_no_cleanup_needed(self, temp_dir: str) -> None:
        """Test cleanup when no cleanup is needed."""
        # Create only 2 folders
        for i in range(2):
            folder_name: str = f"session_2024010{i+1}_120000"
            os.makedirs(os.path.join(temp_dir, folder_name))
        
        cleanup_old_logs(temp_dir, max_folders=5)
        
        remaining_folders: List[str] = [
            d for d in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, d))
        ]
        
        # All folders should remain
        assert len(remaining_folders) == 2

    def test_get_file_size_existing_file(self, temp_dir: str) -> None:
        """Test getting file size for existing file."""
        file_path: str = os.path.join(temp_dir, "test_size.json")
        test_content: str = '{"test": "data", "number": 12345}'
        
        with open(file_path, 'w') as f:
            f.write(test_content)
        
        file_size: int = get_file_size(file_path)
        
        assert file_size > 0
        assert file_size == len(test_content.encode('utf-8'))

    def test_get_file_size_nonexistent_file(self) -> None:
        """Test getting file size for non-existent file."""
        file_size: int = get_file_size("/path/to/nonexistent.json")
        
        assert file_size == 0

    def test_get_file_size_empty_file(self, temp_dir: str) -> None:
        """Test getting file size for empty file."""
        file_path: str = os.path.join(temp_dir, "empty.json")
        
        # Create empty file
        with open(file_path, 'w') as f:
            pass
        
        file_size: int = get_file_size(file_path)
        
        assert file_size == 0

    def test_is_file_readable_readable_file(self, temp_dir: str) -> None:
        """Test checking if readable file is readable."""
        file_path: str = os.path.join(temp_dir, "readable.txt")
        
        with open(file_path, 'w') as f:
            f.write("test content")
        
        is_readable: bool = is_file_readable(file_path)
        
        assert is_readable

    def test_is_file_readable_nonexistent_file(self) -> None:
        """Test checking if non-existent file is readable."""
        is_readable: bool = is_file_readable("/path/to/nonexistent.txt")
        
        assert not is_readable

    def test_is_file_readable_directory(self, temp_dir: str) -> None:
        """Test checking if directory is readable as file."""
        is_readable: bool = is_file_readable(temp_dir)
        
        assert not is_readable

    def test_get_timestamp_string_format(self) -> None:
        """Test timestamp string format."""
        timestamp: str = get_timestamp_string()
        
        # Should be in format YYYYMMDD_HHMMSS
        assert len(timestamp) == 15  # 8 digits + underscore + 6 digits
        assert "_" in timestamp
        
        date_part, time_part = timestamp.split("_")
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 6  # HHMMSS
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_get_timestamp_string_uniqueness(self) -> None:
        """Test that timestamp strings are unique."""
        timestamps: List[str] = []
        
        for _ in range(5):
            timestamp: str = get_timestamp_string()
            timestamps.append(timestamp)
            # Small delay to ensure different timestamps
            import time
            time.sleep(0.01)
        
        # All timestamps should be unique
        assert len(set(timestamps)) == len(timestamps)

    def test_json_operations_integration(self, temp_dir: str) -> None:
        """Test integration of save and load operations."""
        file_path: str = os.path.join(temp_dir, "integration_test.json")
        
        original_data: Dict[str, Any] = {
            "session_id": "test_session",
            "games": [
                {"id": 1, "score": 10},
                {"id": 2, "score": 25}
            ],
            "metadata": {
                "total_games": 2,
                "average_score": 17.5
            }
        }
        
        # Save data
        save_success: bool = save_json_safely(original_data, file_path, indent=2)
        assert save_success
        
        # Load data
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(file_path)
        assert loaded_data is not None
        assert loaded_data == original_data

    def test_directory_operations_integration(self, temp_dir: str) -> None:
        """Test integration of directory operations."""
        session_name: str = f"integration_test_{get_timestamp_string()}"
        
        # Create game folder
        folder_path: Optional[str] = create_game_folder(temp_dir, session_name)
        assert folder_path is not None
        
        # Verify directory structure
        assert os.path.exists(folder_path)
        assert os.path.exists(os.path.join(folder_path, "prompts"))
        assert os.path.exists(os.path.join(folder_path, "responses"))
        
        # Test getting log folders
        log_folders: List[str] = get_game_log_folders(temp_dir)
        assert len(log_folders) >= 1
        assert any(session_name in folder for folder in log_folders)

    @patch('os.makedirs')
    def test_error_handling_with_mocks(self, mock_makedirs: Mock) -> None:
        """Test error handling using mocks."""
        mock_makedirs.side_effect = OSError("Permission denied")
        
        success: bool = ensure_directory_exists("/test/path")
        
        assert not success
        mock_makedirs.assert_called_once()

    def test_edge_cases_special_characters(self, temp_dir: str) -> None:
        """Test handling files with special characters."""
        special_file: str = os.path.join(temp_dir, "test_file_with_特殊字符.json")
        test_data: Dict[str, str] = {"message": "Hello 世界", "emoji": "🐍"}
        
        save_success: bool = save_json_safely(test_data, special_file)
        assert save_success
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(special_file)
        assert loaded_data is not None
        assert loaded_data == test_data


class TestFileUtilsIntegration:
    """Integration tests for file utilities."""

    def test_save_and_load_round_trip(self, temp_dir):
        """Test saving and loading data in round trip."""
        original_data = {
            "score": 15,
            "moves": ["UP", "DOWN", "LEFT", "RIGHT"],
            "metadata": {"timestamp": "2024-01-01"}
        }
        
        # Save the data
        filepath = os.path.join(temp_dir, "test_game.json")
        with open(filepath, 'w') as f:
            json.dump(original_data, f)
        
        # Load it back
        loaded_data = load_game_data(filepath)
        
        assert loaded_data == original_data

    def test_game_numbering_sequence(self, temp_dir):
        """Test game numbering works correctly in sequence."""
        # Create games in sequence
        for i in range(1, 4):
            filepath = os.path.join(temp_dir, f"game_{i}.json")
            with open(filepath, 'w') as f:
                json.dump({"game_number": i}, f)
        
        # Next number should be 4
        next_num = get_next_game_number(temp_dir)
        assert next_num == 4
        
        # Create game 4
        filepath = os.path.join(temp_dir, f"game_{next_num}.json")
        with open(filepath, 'w') as f:
            json.dump({"game_number": next_num}, f)
        
        # Next number should now be 5
        next_num = get_next_game_number(temp_dir)
        assert next_num == 5

    def test_error_handling_robustness(self, temp_dir):
        """Test that file utilities handle errors gracefully."""
        # Test with various problematic files
        test_files = [
            ("empty.json", ""),
            ("partial.json", '{"incomplete": '),
            ("binary.json", b'\x80\x81\x82'),
        ]
        
        for filename, content in test_files:
            filepath = os.path.join(temp_dir, filename)
            mode = 'wb' if isinstance(content, bytes) else 'w'
            with open(filepath, mode) as f:
                f.write(content)
            
            # Should not raise exceptions
            result = load_game_data(filepath)
            assert result is None
            
            result = extract_game_summary(filepath)
            assert result is None 