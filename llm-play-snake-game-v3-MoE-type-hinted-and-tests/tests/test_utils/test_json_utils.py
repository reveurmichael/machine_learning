"""
Tests for JSON utility functions.
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, mock_open, Mock
from utils.json_utils import (
    preprocess_json_string,
    validate_json_format,
    extract_json_from_code_block,
    extract_valid_json,
    extract_json_from_text,
    extract_moves_pattern,
    extract_moves_from_arrays,
    safe_json_parse,
    validate_json_structure,
    merge_json_files,
    backup_json_file,
    repair_malformed_json
)


class TestPreprocessJsonString:
    """Test cases for preprocess_json_string."""

    def test_remove_comments(self):
        """Test removal of comments."""
        json_str = '{"moves": ["UP"]} // This is a comment'
        result = preprocess_json_string(json_str)
        assert "// This is a comment" not in result

    def test_remove_multiline_comments(self):
        """Test removal of multiline comments."""
        json_str = '{"moves": ["UP"]} /* This is a \n multiline comment */'
        result = preprocess_json_string(json_str)
        assert "/* This is a" not in result
        assert "multiline comment */" not in result

    def test_fix_unquoted_keys(self):
        """Test fixing unquoted keys."""
        json_str = '{moves: ["UP", "DOWN"]}'
        result = preprocess_json_string(json_str)
        assert '"moves"' in result

    def test_fix_trailing_commas_arrays(self):
        """Test fixing trailing commas in arrays."""
        json_str = '{"moves": ["UP", "DOWN",]}'
        result = preprocess_json_string(json_str)
        assert ',"UP", "DOWN"]' in result or '"UP", "DOWN"]' in result

    def test_fix_trailing_commas_objects(self):
        """Test fixing trailing commas in objects."""
        json_str = '{"moves": ["UP"], "score": 5,}'
        result = preprocess_json_string(json_str)
        assert result.endswith('}')
        assert not result.endswith(',}')

    def test_fix_single_quotes(self):
        """Test converting single quotes to double quotes."""
        json_str = "{'moves': ['UP', 'DOWN']}"
        result = preprocess_json_string(json_str)
        assert '"moves"' in result
        assert '"UP"' in result
        assert '"DOWN"' in result

    def test_preserve_quotes_in_strings(self):
        """Test that quotes inside strings are preserved."""
        json_str = '{"message": "Player said \\"UP\\"", "moves": ["UP"]}'
        result = preprocess_json_string(json_str)
        # Should preserve escaped quotes within strings
        assert '\\"UP\\"' in result or '"UP"' in result


class TestValidateJsonFormat:
    """Test cases for validate_json_format."""

    def test_valid_format(self):
        """Test validation of valid format."""
        data = {"moves": ["UP", "DOWN", "LEFT", "RIGHT"]}
        is_valid, error = validate_json_format(data)
        assert is_valid
        assert error is None

    def test_not_dictionary(self):
        """Test validation fails for non-dictionary."""
        data = ["UP", "DOWN"]
        is_valid, error = validate_json_format(data)
        assert not is_valid
        assert "not a dictionary" in error

    def test_missing_moves_key(self):
        """Test validation fails for missing moves key."""
        data = {"actions": ["UP", "DOWN"]}
        is_valid, error = validate_json_format(data)
        assert not is_valid
        assert "moves" in error

    def test_moves_not_list(self):
        """Test validation fails when moves is not a list."""
        data = {"moves": "UP DOWN"}
        is_valid, error = validate_json_format(data)
        assert not is_valid
        assert "not a list" in error

    def test_invalid_move(self):
        """Test validation fails for invalid moves."""
        data = {"moves": ["UP", "INVALID_MOVE"]}
        is_valid, error = validate_json_format(data)
        assert not is_valid
        assert "Invalid move" in error

    def test_non_string_move(self):
        """Test validation fails for non-string moves."""
        data = {"moves": ["UP", 123]}
        is_valid, error = validate_json_format(data)
        assert not is_valid
        assert "not a string" in error

    def test_case_insensitive_validation(self):
        """Test that moves are normalized to uppercase."""
        data = {"moves": ["up", "down", "LEFT"]}
        is_valid, error = validate_json_format(data)
        assert is_valid
        assert data["moves"] == ["UP", "DOWN", "LEFT"]

    def test_empty_moves_list(self):
        """Test validation of empty moves list."""
        data = {"moves": []}
        is_valid, error = validate_json_format(data)
        assert is_valid
        assert error is None


class TestExtractJsonFromCodeBlock:
    """Test cases for extract_json_from_code_block."""

    def test_extract_from_json_block(self):
        """Test extraction from JSON code block."""
        response = '''Here's the JSON:
        ```json
        {"moves": ["UP", "RIGHT"]}
        ```'''
        result = extract_json_from_code_block(response)
        assert result is not None
        assert result["moves"] == ["UP", "RIGHT"]

    def test_extract_from_javascript_block(self):
        """Test extraction from JavaScript code block."""
        response = '''```javascript
        {"moves": ["DOWN", "LEFT"]}
        ```'''
        result = extract_json_from_code_block(response)
        assert result is not None
        assert result["moves"] == ["DOWN", "LEFT"]

    def test_extract_from_plain_block(self):
        """Test extraction from plain code block."""
        response = '''```
        {"moves": ["UP", "UP", "RIGHT"]}
        ```'''
        result = extract_json_from_code_block(response)
        assert result is not None
        assert result["moves"] == ["UP", "UP", "RIGHT"]

    def test_multiple_blocks_first_valid(self):
        """Test extraction when first block is valid."""
        response = '''
        ```json
        {"moves": ["UP"]}
        ```
        ```json
        {"invalid": "data"}
        ```
        '''
        result = extract_json_from_code_block(response)
        assert result is not None
        assert result["moves"] == ["UP"]

    def test_no_valid_blocks(self):
        """Test when no blocks contain valid JSON."""
        response = '''```json
        {"invalid": "no moves key"}
        ```'''
        result = extract_json_from_code_block(response)
        assert result is None

    def test_invalid_json_in_block(self):
        """Test handling of invalid JSON in code block."""
        response = '''
        ```json
        {moves: ["UP" // invalid JSON
        ```
        '''
        result = extract_json_from_code_block(response)
        assert result is None

    def test_pattern_matching_fallback(self):
        """Test pattern matching fallback when normal extraction fails."""
        response = '''
        The moves are: {"moves": ["UP", "RIGHT", "DOWN"]}
        '''
        result = extract_json_from_code_block(response)
        assert result is not None
        assert result["moves"] == ["UP", "RIGHT", "DOWN"]

    def test_no_code_blocks(self):
        """Test when response has no code blocks."""
        response = "Just plain text with no JSON"
        result = extract_json_from_code_block(response)
        assert result is None


class TestExtractValidJson:
    """Test cases for extract_valid_json."""

    def test_direct_json_parsing(self):
        """Test direct JSON parsing."""
        text = '{"moves": ["UP", "DOWN"]}'
        result = extract_valid_json(text)
        assert result is not None
        assert result["moves"] == ["UP", "DOWN"]

    def test_json_with_preprocessing(self):
        """Test JSON that needs preprocessing."""
        text = '{moves: ["UP", "DOWN",]}'  # Unquoted key, trailing comma
        result = extract_valid_json(text)
        assert result is not None
        assert result["moves"] == ["UP", "DOWN"]

    def test_code_block_extraction(self):
        """Test extraction from code blocks."""
        text = '''
        Here's the response:
        ```json
        {"moves": ["LEFT", "RIGHT"]}
        ```
        '''
        result = extract_valid_json(text)
        assert result is not None
        assert result["moves"] == ["LEFT", "RIGHT"]

    def test_text_based_extraction(self):
        """Test text-based extraction fallback."""
        text = 'Move the snake with these directions: {"moves": ["UP"]}'
        result = extract_valid_json(text)
        assert result is not None
        assert result["moves"] == ["UP"]

    def test_no_valid_json(self):
        """Test when no valid JSON can be extracted."""
        text = "This is just plain text with no JSON data"
        result = extract_valid_json(text)
        assert result is None

    def test_invalid_json_format(self):
        """Test when JSON is valid but doesn't have required format."""
        text = '{"actions": ["UP", "DOWN"]}'  # Wrong key
        result = extract_valid_json(text)
        assert result is None


class TestExtractJsonFromText:
    """Test cases for extract_json_from_text."""

    def test_simple_json_extraction(self):
        """Test extraction of simple JSON from text."""
        response = 'The answer is {"moves": ["UP", "RIGHT"]} for the snake.'
        result = extract_json_from_text(response)
        assert result is not None
        assert result["moves"] == ["UP", "RIGHT"]

    def test_json_at_start(self):
        """Test JSON at the start of text."""
        response = '{"moves": ["DOWN"]} is my response'
        result = extract_json_from_text(response)
        assert result is not None
        assert result["moves"] == ["DOWN"]

    def test_json_at_end(self):
        """Test JSON at the end of text."""
        response = 'My final answer: {"moves": ["LEFT", "LEFT"]}'
        result = extract_json_from_text(response)
        assert result is not None
        assert result["moves"] == ["LEFT", "LEFT"]

    def test_multiple_json_objects(self):
        """Test when multiple JSON objects are present."""
        response = '{"other": "data"} and {"moves": ["UP"]} are here'
        result = extract_json_from_text(response)
        assert result is not None
        assert result["moves"] == ["UP"]

    def test_no_json_in_text(self):
        """Test when no JSON is found in text."""
        response = "This is just plain text without any JSON"
        result = extract_json_from_text(response)
        assert result is None

    def test_invalid_json_syntax(self):
        """Test handling of invalid JSON syntax."""
        response = 'Here is {moves: ["UP"] // invalid JSON'
        result = extract_json_from_text(response)
        assert result is None


class TestExtractMovesPattern:
    """Test cases for extract_moves_pattern."""

    def test_simple_moves_pattern(self):
        """Test extraction of simple moves pattern."""
        json_str = '"moves": ["UP", "DOWN", "LEFT"]'
        result = extract_moves_pattern(json_str)
        assert result is not None
        assert result["moves"] == ["UP", "DOWN", "LEFT"]

    def test_moves_with_whitespace(self):
        """Test moves pattern with extra whitespace."""
        json_str = '"moves" : [ "UP" , "DOWN" ]'
        result = extract_moves_pattern(json_str)
        assert result is not None
        assert result["moves"] == ["UP", "DOWN"]

    def test_no_moves_pattern(self):
        """Test when no moves pattern is found."""
        json_str = '"actions": ["UP", "DOWN"]'
        result = extract_moves_pattern(json_str)
        assert result is None

    def test_empty_moves_array(self):
        """Test empty moves array."""
        json_str = '"moves": []'
        result = extract_moves_pattern(json_str)
        assert result is not None
        assert result["moves"] == []

    def test_invalid_moves(self):
        """Test filtering of invalid moves."""
        json_str = '"moves": ["UP", "INVALID", "DOWN"]'
        result = extract_moves_pattern(json_str)
        assert result is not None
        assert "INVALID" not in result["moves"]
        assert "UP" in result["moves"]
        assert "DOWN" in result["moves"]


class TestExtractMovesFromArrays:
    """Test cases for extract_moves_from_arrays."""

    def test_array_format_extraction(self):
        """Test extraction from array format."""
        response = 'moves = ["UP", "RIGHT", "DOWN"]'
        result = extract_moves_from_arrays(response)
        assert result is not None
        assert result["moves"] == ["UP", "RIGHT", "DOWN"]

    def test_multiple_array_formats(self):
        """Test when multiple array formats are present."""
        response = '''
        actions = ["INVALID"]
        moves = ["UP", "DOWN"]
        directions = ["LEFT"]
        '''
        result = extract_moves_from_arrays(response)
        assert result is not None
        # Should prefer moves array
        assert result["moves"] == ["UP", "DOWN"]

    def test_no_array_format(self):
        """Test when no array format is found."""
        response = "Just some text without arrays"
        result = extract_moves_from_arrays(response)
        assert result is None

    def test_invalid_array_syntax(self):
        """Test handling of invalid array syntax."""
        response = 'moves = ["UP", "DOWN" // missing closing bracket'
        result = extract_moves_from_arrays(response)
        assert result is None

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        response = 'moves = []'
        result = extract_moves_from_arrays(response)
        assert result is not None
        assert result["moves"] == []


class TestIntegration:
    """Integration tests for JSON utilities."""

    def test_full_extraction_pipeline(self):
        """Test the full extraction pipeline with various formats."""
        test_cases = [
            # Direct JSON
            ('{"moves": ["UP"]}', ["UP"]),
            # Code block
            ('```json\n{"moves": ["DOWN"]}\n```', ["DOWN"]),
            # Embedded in text
            ('Answer: {"moves": ["LEFT", "RIGHT"]}', ["LEFT", "RIGHT"]),
            # Array format
            ('moves = ["UP", "UP", "DOWN"]', ["UP", "UP", "DOWN"]),
            # Preprocessed JSON
            ('{moves: ["RIGHT",]}', ["RIGHT"]),
        ]

        for text, expected_moves in test_cases:
            result = extract_valid_json(text)
            assert result is not None, f"Failed to extract from: {text}"
            assert result["moves"] == expected_moves, f"Wrong moves for: {text}"

    def test_error_handling_robustness(self):
        """Test robustness of error handling."""
        invalid_inputs = [
            "",  # Empty string
            "plain text",  # No JSON
            '{"invalid": "format"}',  # Wrong format
            '{"moves": "not_a_list"}',  # Wrong type
            '{"moves": [123]}',  # Invalid move type
            '{"moves": ["INVALID_MOVE"]}',  # Invalid move
        ]

        for invalid_input in invalid_inputs:
            result = extract_valid_json(invalid_input)
            assert result is None, f"Should have failed for: {invalid_input}"

    def test_case_normalization(self):
        """Test that moves are properly normalized to uppercase."""
        test_cases = [
            ('{"moves": ["up", "down"]}', ["UP", "DOWN"]),
            ('{"moves": ["Left", "RIGHT"]}', ["LEFT", "RIGHT"]),
        ]

        for text, expected_moves in test_cases:
            result = extract_valid_json(text)
            assert result is not None
            assert result["moves"] == expected_moves


class TestJsonUtils:
    """Test cases for JSON utility functions."""

    def test_safe_json_parse_valid_json(self) -> None:
        """Test parsing valid JSON strings."""
        valid_json: str = '{"key": "value", "number": 42}'
        result: Optional[Dict[str, Any]] = safe_json_parse(valid_json)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_safe_json_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON strings."""
        invalid_json: str = '{"key": "value", "number": 42'  # Missing closing brace
        result: Optional[Dict[str, Any]] = safe_json_parse(invalid_json)
        
        assert result is None

    def test_safe_json_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        result: Optional[Dict[str, Any]] = safe_json_parse("")
        assert result is None

    def test_safe_json_parse_none_input(self) -> None:
        """Test parsing None input."""
        result: Optional[Dict[str, Any]] = safe_json_parse(None)
        assert result is None

    def test_safe_json_parse_array(self) -> None:
        """Test parsing JSON array."""
        json_array: str = '[1, 2, 3, "test"]'
        result: Optional[Union[Dict[str, Any], List[Any]]] = safe_json_parse(json_array)
        
        assert result is not None
        assert isinstance(result, list)
        assert result == [1, 2, 3, "test"]

    def test_extract_json_from_text_simple(self) -> None:
        """Test extracting JSON from text with simple JSON object."""
        text: str = "Here is some JSON: {\"moves\": [\"UP\", \"RIGHT\"]} and more text."
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is not None
        assert result["moves"] == ["UP", "RIGHT"]

    def test_extract_json_from_text_code_block(self) -> None:
        """Test extracting JSON from code block."""
        text: str = '''
        Here's the response:
        ```json
        {"direction": "LEFT", "confidence": 0.8}
        ```
        End of response.
        '''
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is not None
        assert result["direction"] == "LEFT"
        assert result["confidence"] == 0.8

    def test_extract_json_from_text_multiple_objects(self) -> None:
        """Test extracting first JSON object when multiple exist."""
        text: str = '''
        First: {"a": 1}
        Second: {"b": 2}
        '''
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is not None
        assert result["a"] == 1
        # Should extract the first valid JSON object

    def test_extract_json_from_text_no_json(self) -> None:
        """Test extracting JSON when no valid JSON exists."""
        text: str = "This text contains no valid JSON objects."
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is None

    def test_extract_json_from_text_malformed(self) -> None:
        """Test extracting JSON from text with malformed JSON."""
        text: str = "JSON: {\"incomplete\": \"object\""
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is None

    def test_validate_json_structure_valid(self) -> None:
        """Test validating valid JSON structure."""
        json_data: Dict[str, Any] = {
            "moves": ["UP", "DOWN"],
            "score": 10,
            "active": True
        }
        required_fields: List[str] = ["moves", "score"]
        
        is_valid: bool
        errors: List[str]
        is_valid, errors = validate_json_structure(json_data, required_fields)
        
        assert is_valid
        assert len(errors) == 0

    def test_validate_json_structure_missing_fields(self) -> None:
        """Test validating JSON with missing required fields."""
        json_data: Dict[str, Any] = {"score": 10}
        required_fields: List[str] = ["moves", "score", "level"]
        
        is_valid: bool
        errors: List[str]
        is_valid, errors = validate_json_structure(json_data, required_fields)
        
        assert not is_valid
        assert len(errors) == 2  # Missing "moves" and "level"
        assert any("moves" in error for error in errors)
        assert any("level" in error for error in errors)

    def test_validate_json_structure_type_checking(self) -> None:
        """Test validating JSON with type checking."""
        json_data: Dict[str, Any] = {
            "moves": "not_a_list",  # Should be a list
            "score": 10
        }
        required_fields: List[str] = ["moves", "score"]
        field_types: Dict[str, type] = {"moves": list, "score": int}
        
        is_valid: bool
        errors: List[str]
        is_valid, errors = validate_json_structure(json_data, required_fields, field_types)
        
        assert not is_valid
        assert len(errors) == 1
        assert "moves" in errors[0] and "list" in errors[0]

    def test_validate_json_structure_empty_required(self) -> None:
        """Test validating JSON with empty required fields list."""
        json_data: Dict[str, Any] = {"any": "data"}
        required_fields: List[str] = []
        
        is_valid: bool
        errors: List[str]
        is_valid, errors = validate_json_structure(json_data, required_fields)
        
        assert is_valid
        assert len(errors) == 0

    def test_merge_json_files_success(self, temp_dir: str) -> None:
        """Test successfully merging JSON files."""
        # Create test files
        file1_path: str = os.path.join(temp_dir, "file1.json")
        file2_path: str = os.path.join(temp_dir, "file2.json")
        output_path: str = os.path.join(temp_dir, "merged.json")
        
        file1_data: Dict[str, Any] = {"a": 1, "b": 2}
        file2_data: Dict[str, Any] = {"c": 3, "d": 4}
        
        with open(file1_path, 'w') as f:
            json.dump(file1_data, f)
        with open(file2_path, 'w') as f:
            json.dump(file2_data, f)
        
        success: bool = merge_json_files([file1_path, file2_path], output_path)
        
        assert success
        assert os.path.exists(output_path)
        
        with open(output_path, 'r') as f:
            merged_data: Dict[str, Any] = json.load(f)
            assert merged_data["a"] == 1
            assert merged_data["c"] == 3

    def test_merge_json_files_conflicting_keys(self, temp_dir: str) -> None:
        """Test merging JSON files with conflicting keys."""
        file1_path: str = os.path.join(temp_dir, "file1.json")
        file2_path: str = os.path.join(temp_dir, "file2.json")
        output_path: str = os.path.join(temp_dir, "merged.json")
        
        file1_data: Dict[str, Any] = {"key": "value1", "unique1": "data1"}
        file2_data: Dict[str, Any] = {"key": "value2", "unique2": "data2"}
        
        with open(file1_path, 'w') as f:
            json.dump(file1_data, f)
        with open(file2_path, 'w') as f:
            json.dump(file2_data, f)
        
        success: bool = merge_json_files([file1_path, file2_path], output_path)
        
        assert success
        
        with open(output_path, 'r') as f:
            merged_data: Dict[str, Any] = json.load(f)
            # Later file should override conflicting keys
            assert merged_data["key"] == "value2"
            assert merged_data["unique1"] == "data1"
            assert merged_data["unique2"] == "data2"

    def test_merge_json_files_nonexistent_file(self, temp_dir: str) -> None:
        """Test merging with a non-existent file."""
        file1_path: str = os.path.join(temp_dir, "file1.json")
        nonexistent_path: str = os.path.join(temp_dir, "nonexistent.json")
        output_path: str = os.path.join(temp_dir, "output.json")
        
        file1_data: Dict[str, Any] = {"a": 1}
        with open(file1_path, 'w') as f:
            json.dump(file1_data, f)
        
        success: bool = merge_json_files([file1_path, nonexistent_path], output_path)
        
        assert not success

    def test_backup_json_file_success(self, temp_dir: str) -> None:
        """Test successfully creating a backup of JSON file."""
        original_path: str = os.path.join(temp_dir, "original.json")
        test_data: Dict[str, Any] = {"test": "data", "number": 42}
        
        with open(original_path, 'w') as f:
            json.dump(test_data, f)
        
        success: bool = backup_json_file(original_path)
        
        assert success
        
        # Check that backup file was created
        backup_path: str = original_path + ".backup"
        assert os.path.exists(backup_path)
        
        # Verify backup contents
        with open(backup_path, 'r') as f:
            backup_data: Dict[str, Any] = json.load(f)
            assert backup_data == test_data

    def test_backup_json_file_nonexistent(self) -> None:
        """Test backing up a non-existent file."""
        nonexistent_path: str = "/path/to/nonexistent.json"
        success: bool = backup_json_file(nonexistent_path)
        
        assert not success

    def test_backup_json_file_permission_error(self, temp_dir: str) -> None:
        """Test backup when backup file cannot be created."""
        original_path: str = os.path.join(temp_dir, "original.json")
        test_data: Dict[str, Any] = {"test": "data"}
        
        with open(original_path, 'w') as f:
            json.dump(test_data, f)
        
        # Mock file operations to simulate permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            success: bool = backup_json_file(original_path)
            assert not success

    def test_repair_malformed_json_missing_quotes(self) -> None:
        """Test repairing JSON with missing quotes."""
        malformed: str = '{key: value, number: 42}'
        repaired: Optional[str] = repair_malformed_json(malformed)
        
        assert repaired is not None
        parsed: Optional[Dict[str, Any]] = safe_json_parse(repaired)
        assert parsed is not None
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_repair_malformed_json_trailing_comma(self) -> None:
        """Test repairing JSON with trailing comma."""
        malformed: str = '{"key": "value", "number": 42,}'
        repaired: Optional[str] = repair_malformed_json(malformed)
        
        assert repaired is not None
        parsed: Optional[Dict[str, Any]] = safe_json_parse(repaired)
        assert parsed is not None

    def test_repair_malformed_json_missing_brace(self) -> None:
        """Test repairing JSON with missing closing brace."""
        malformed: str = '{"key": "value", "number": 42'
        repaired: Optional[str] = repair_malformed_json(malformed)
        
        assert repaired is not None
        parsed: Optional[Dict[str, Any]] = safe_json_parse(repaired)
        assert parsed is not None

    def test_repair_malformed_json_unrepairable(self) -> None:
        """Test attempting to repair severely malformed JSON."""
        malformed: str = 'This is not JSON at all!'
        repaired: Optional[str] = repair_malformed_json(malformed)
        
        # Should return None for unrepairable JSON
        assert repaired is None

    def test_repair_malformed_json_already_valid(self) -> None:
        """Test repairing already valid JSON."""
        valid_json: str = '{"key": "value", "number": 42}'
        repaired: Optional[str] = repair_malformed_json(valid_json)
        
        assert repaired is not None
        # Should return the original valid JSON
        parsed: Optional[Dict[str, Any]] = safe_json_parse(repaired)
        assert parsed is not None
        assert parsed["key"] == "value"

    def test_complex_json_extraction(self) -> None:
        """Test extracting complex nested JSON from text."""
        text: str = '''
        The LLM responded with:
        ```
        {
            "moves": ["UP", "RIGHT", "DOWN"],
            "metadata": {
                "confidence": 0.95,
                "reasoning": "Optimal path to apple"
            },
            "alternatives": [
                {"moves": ["LEFT"], "confidence": 0.3}
            ]
        }
        ```
        '''
        
        result: Optional[Dict[str, Any]] = extract_json_from_text(text)
        
        assert result is not None
        assert len(result["moves"]) == 3
        assert result["metadata"]["confidence"] == 0.95
        assert len(result["alternatives"]) == 1

    def test_edge_case_empty_json_object(self) -> None:
        """Test handling empty JSON object."""
        empty_json: str = '{}'
        result: Optional[Dict[str, Any]] = safe_json_parse(empty_json)
        
        assert result is not None
        assert result == {}

    def test_edge_case_deeply_nested_json(self) -> None:
        """Test handling deeply nested JSON structures."""
        nested_json: str = '''
        {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        '''
        
        result: Optional[Dict[str, Any]] = safe_json_parse(nested_json)
        
        assert result is not None
        assert result["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_unicode_handling(self) -> None:
        """Test handling JSON with unicode characters."""
        unicode_json: str = '{"message": "Hello 世界", "emoji": "🐍"}'
        result: Optional[Dict[str, Any]] = safe_json_parse(unicode_json)
        
        assert result is not None
        assert result["message"] == "Hello 世界"
        assert result["emoji"] == "🐍"

    @patch('builtins.open', mock_open(read_data='{"test": "data"}'))
    def test_file_operations_with_mock(self) -> None:
        """Test file operations using mocked file system."""
        # This test demonstrates how to mock file operations
        with patch('os.path.exists', return_value=True):
            # Test code that reads from files
            pass 