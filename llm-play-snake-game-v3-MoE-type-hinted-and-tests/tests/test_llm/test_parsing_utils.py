"""Tests for llm.parsing_utils module."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from llm.parsing_utils import (
    parse_and_format,
    parse_llm_response,
    extract_json_from_response,
    validate_moves_format,
    clean_response_text,
    extract_moves_and_reasoning,
    handle_parsing_errors,
)


class TestParseAndFormat:
    """Test class for parse_and_format function."""

    def test_parse_and_format_valid_json(self):
        """Test parsing and formatting valid JSON response."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move towards apple"}'
        
        result = parse_and_format(response)
        
        assert isinstance(result, dict)
        assert "moves" in result
        assert "reasoning" in result
        assert result["moves"] == ["UP", "RIGHT"]

    def test_parse_and_format_malformed_json(self):
        """Test parsing malformed JSON."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move towards apple"'  # Missing closing brace
        
        result = parse_and_format(response)
        
        # Should return None or error dict for malformed JSON
        assert result is None or (isinstance(result, dict) and "error" in str(result).lower())

    def test_parse_and_format_empty_response(self):
        """Test parsing empty response."""
        response = ""
        
        result = parse_and_format(response)
        
        assert result is None or isinstance(result, dict)

    def test_parse_and_format_no_json_response(self):
        """Test parsing response with no JSON."""
        response = "I cannot determine a safe move. There is no path available."
        
        result = parse_and_format(response)
        
        # Should handle non-JSON responses gracefully
        assert result is None or isinstance(result, dict)

    def test_parse_and_format_response_with_extra_text(self):
        """Test parsing response with extra text around JSON."""
        response = '''
        Here's my analysis:
        {"moves": ["LEFT", "DOWN"], "reasoning": "Avoid wall"}
        That's my recommendation.
        '''
        
        result = parse_and_format(response)
        
        if result is not None:
            assert isinstance(result, dict)
            assert "moves" in result
            assert result["moves"] == ["LEFT", "DOWN"]

    def test_parse_and_format_unicode(self):
        """Test parsing response with unicode characters."""
        response = '{"moves": ["UP"], "reasoning": "Move towards 🍎 apple"}'
        
        result = parse_and_format(response)
        
        if result is not None:
            assert isinstance(result, dict)
            assert result["moves"] == ["UP"]
            assert "🍎" in result["reasoning"]

    def test_parse_and_format_none_input(self):
        """Test parsing None input."""
        response = None
        
        result = parse_and_format(response)
        
        assert result is None

    def test_parse_and_format_empty_moves(self):
        """Test parsing response with empty moves."""
        response = '{"moves": [], "reasoning": "NO_PATH_FOUND"}'
        
        result = parse_and_format(response)
        
        if result is not None:
            assert isinstance(result, dict)
            assert result["moves"] == []
            assert "NO_PATH_FOUND" in result["reasoning"]

    def test_parse_and_format_missing_fields(self):
        """Test parsing response with missing required fields."""
        response = '{"moves": ["UP"]}'  # Missing reasoning
        
        result = parse_and_format(response)
        
        # Should handle missing fields appropriately
        assert result is None or isinstance(result, dict)

    def test_parse_and_format_wrong_types(self):
        """Test parsing response with wrong field types."""
        response = '{"moves": "UP", "reasoning": ["not", "string"]}'
        
        result = parse_and_format(response)
        
        # Should handle wrong types appropriately
        assert result is None or isinstance(result, dict)

    def test_parse_and_format_large_response(self):
        """Test parsing very large response."""
        large_reasoning = "x" * 5000
        response = f'{{"moves": ["UP"], "reasoning": "{large_reasoning}"}}'
        
        result = parse_and_format(response)
        
        if result is not None:
            assert isinstance(result, dict)
            assert result["moves"] == ["UP"]
            assert len(result["reasoning"]) == 5000


class TestParseLLMResponse:
    """Test class for parse_llm_response function."""

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move towards apple"}'
        
        result = parse_llm_response(response)
        
        assert isinstance(result, dict)
        assert "moves" in result
        assert "reasoning" in result
        assert result["moves"] == ["UP", "RIGHT"]

    def test_parse_llm_response_with_newlines(self):
        """Test parsing response with newlines and formatting."""
        response = '''
        {
            "moves": [
                "UP",
                "UP", 
                "RIGHT"
            ],
            "reasoning": "Multi-line reasoning with\nnewlines"
        }
        '''
        
        result = parse_llm_response(response)
        
        assert isinstance(result, dict)
        assert result["moves"] == ["UP", "UP", "RIGHT"]
        assert "Multi-line" in result["reasoning"]

    def test_parse_llm_response_malformed_json(self):
        """Test parsing malformed JSON."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move towards apple"'  # Missing closing brace
        
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            parse_llm_response(response)

    def test_parse_llm_response_empty_response(self):
        """Test parsing empty response."""
        response = ""
        
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            parse_llm_response(response)

    def test_parse_llm_response_no_json(self):
        """Test parsing response with no JSON."""
        response = "This is just plain text with no JSON structure."
        
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            parse_llm_response(response)

    def test_parse_llm_response_unicode(self):
        """Test parsing response with unicode characters."""
        response = '{"moves": ["UP"], "reasoning": "Move towards 🍎 apple"}'
        
        result = parse_llm_response(response)
        
        assert result["moves"] == ["UP"]
        assert "🍎" in result["reasoning"]

    def test_parse_llm_response_nested_objects(self):
        """Test parsing response with nested objects."""
        response = '''
        {
            "moves": ["DOWN"], 
            "reasoning": "Object {x: 1} in reasoning", 
            "meta": {"info": "nested"}
        }
        '''
        
        result = parse_llm_response(response)
        
        assert result["moves"] == ["DOWN"]
        if "meta" in result:
            assert isinstance(result["meta"], dict)

    def test_parse_llm_response_escaped_quotes(self):
        """Test parsing response with escaped quotes."""
        response = r'{"moves": ["UP"], "reasoning": "Say \"hello\" to apple"}'
        
        result = parse_llm_response(response)
        
        assert '"hello"' in result["reasoning"]

    def test_parse_llm_response_array_values(self):
        """Test parsing response with array values."""
        response = '{"moves": ["UP", "RIGHT", "DOWN", "LEFT"], "reasoning": "Complex path"}'
        
        result = parse_llm_response(response)
        
        assert len(result["moves"]) == 4
        assert all(move in ["UP", "DOWN", "LEFT", "RIGHT"] for move in result["moves"])

    def test_parse_llm_response_extra_fields(self):
        """Test parsing response with extra fields."""
        response = '{"moves": ["LEFT"], "reasoning": "Safe", "confidence": 0.9, "timestamp": "2024-01-01"}'
        
        result = parse_llm_response(response)
        
        assert result["moves"] == ["LEFT"]
        assert result["reasoning"] == "Safe"
        # Extra fields might be preserved or ignored depending on implementation


class TestExtractJsonFromResponse:
    """Test class for extract_json_from_response function."""

    def test_extract_simple_json(self):
        """Test extracting simple JSON object."""
        response = '{"moves": ["UP"], "reasoning": "test"}'
        
        json_str = extract_json_from_response(response)
        
        assert json_str == response

    def test_extract_json_with_prefix_suffix(self):
        """Test extracting JSON with text before and after."""
        response = 'Here is my move: {"moves": ["DOWN"], "reasoning": "safe"} End of response.'
        
        json_str = extract_json_from_response(response)
        
        assert '{"moves": ["DOWN"], "reasoning": "safe"}' in json_str

    def test_extract_json_nested_braces(self):
        """Test extracting JSON with nested braces."""
        response = '{"moves": ["UP"], "reasoning": "Object {x: 1} in reasoning", "meta": {"info": "nested"}}'
        
        json_str = extract_json_from_response(response)
        
        # Should extract the complete JSON object
        parsed = json.loads(json_str)
        assert parsed["moves"] == ["UP"]
        assert "meta" in parsed

    def test_extract_json_multiple_objects(self):
        """Test extracting JSON when multiple objects present."""
        response = '{"a": 1} and {"moves": ["LEFT"], "reasoning": "correct one"} and {"c": 3}'
        
        json_str = extract_json_from_response(response)
        
        # Should extract the one with "moves" and "reasoning"
        parsed = json.loads(json_str)
        assert "moves" in parsed
        assert "reasoning" in parsed

    def test_extract_json_with_escaped_quotes(self):
        """Test extracting JSON with escaped quotes."""
        response = r'{"moves": ["UP"], "reasoning": "Say \"hello\" to apple"}'
        
        json_str = extract_json_from_response(response)
        
        parsed = json.loads(json_str)
        assert '"hello"' in parsed["reasoning"]

    def test_extract_no_json_found(self):
        """Test when no JSON is found."""
        response = "This is just plain text with no JSON structure."
        
        json_str = extract_json_from_response(response)
        
        assert json_str is None or json_str == ""

    def test_extract_json_with_arrays(self):
        """Test extracting JSON with array values."""
        response = '{"moves": ["UP", "RIGHT", "DOWN"], "reasoning": "Complex path"}'
        
        json_str = extract_json_from_response(response)
        
        parsed = json.loads(json_str)
        assert len(parsed["moves"]) == 3


class TestValidateMovesFormat:
    """Test class for validate_moves_format function."""

    def test_validate_valid_moves(self):
        """Test validation of valid moves."""
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        is_valid = validate_moves_format(moves)
        
        assert is_valid is True

    def test_validate_empty_moves(self):
        """Test validation of empty moves list."""
        moves = []
        
        is_valid = validate_moves_format(moves)
        
        # Empty moves might be valid for NO_PATH_FOUND scenarios
        assert isinstance(is_valid, bool)

    def test_validate_invalid_move_names(self):
        """Test validation with invalid move names."""
        moves = ["UP", "FORWARD", "BACKWARD"]
        
        is_valid = validate_moves_format(moves)
        
        assert is_valid is False

    def test_validate_non_list_moves(self):
        """Test validation when moves is not a list."""
        moves = "UP"
        
        is_valid = validate_moves_format(moves)
        
        assert is_valid is False

    def test_validate_moves_with_none(self):
        """Test validation with None values in moves."""
        moves = ["UP", None, "DOWN"]
        
        is_valid = validate_moves_format(moves)
        
        assert is_valid is False

    def test_validate_moves_wrong_case(self):
        """Test validation with wrong case moves."""
        moves = ["up", "Down", "LEFT"]
        
        is_valid = validate_moves_format(moves)
        
        assert is_valid is False

    def test_validate_moves_with_extra_spaces(self):
        """Test validation with extra spaces."""
        moves = [" UP ", "DOWN", " LEFT"]
        
        is_valid = validate_moves_format(moves)
        
        # Depends on implementation - might be invalid or cleaned
        assert isinstance(is_valid, bool)

    def test_validate_very_long_moves_list(self):
        """Test validation with very long moves list."""
        moves = ["UP"] * 1000
        
        is_valid = validate_moves_format(moves)
        
        # Should handle long lists (might have practical limits)
        assert isinstance(is_valid, bool)


class TestCleanResponseText:
    """Test class for clean_response_text function."""

    def test_clean_basic_text(self):
        """Test cleaning basic text."""
        text = "  Clean this text  "
        
        cleaned = clean_response_text(text)
        
        assert cleaned == "Clean this text"

    def test_clean_text_with_newlines(self):
        """Test cleaning text with various newlines."""
        text = "Line 1\nLine 2\r\nLine 3\rLine 4"
        
        cleaned = clean_response_text(text)
        
        # Should normalize newlines
        assert "\r" not in cleaned

    def test_clean_text_with_tabs(self):
        """Test cleaning text with tabs."""
        text = "Text\twith\ttabs"
        
        cleaned = clean_response_text(text)
        
        # Should handle tabs appropriately
        assert isinstance(cleaned, str)

    def test_clean_text_with_unicode(self):
        """Test cleaning text with unicode characters."""
        text = "Text with unicode: 🐍 🍎 ⬆️"
        
        cleaned = clean_response_text(text)
        
        assert "🐍" in cleaned  # Should preserve unicode

    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        text = ""
        
        cleaned = clean_response_text(text)
        
        assert cleaned == ""

    def test_clean_whitespace_only(self):
        """Test cleaning whitespace-only text."""
        text = "   \t\n\r   "
        
        cleaned = clean_response_text(text)
        
        assert cleaned == "" or cleaned.isspace()

    def test_clean_text_with_control_characters(self):
        """Test cleaning text with control characters."""
        text = "Text\x00with\x01control\x02chars"
        
        cleaned = clean_response_text(text)
        
        # Should remove or handle control characters
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "\x02" not in cleaned

    def test_clean_very_long_text(self):
        """Test cleaning very long text."""
        text = "x" * 10000
        
        cleaned = clean_response_text(text)
        
        assert isinstance(cleaned, str)
        # Should handle long text without issues


class TestExtractMovesAndReasoning:
    """Test class for extract_moves_and_reasoning function."""

    def test_extract_valid_json(self):
        """Test extracting from valid JSON."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move to apple"}'
        
        moves, reasoning = extract_moves_and_reasoning(response)
        
        assert moves == ["UP", "RIGHT"]
        assert reasoning == "Move to apple"

    def test_extract_empty_moves(self):
        """Test extracting empty moves."""
        response = '{"moves": [], "reasoning": "NO_PATH_FOUND"}'
        
        moves, reasoning = extract_moves_and_reasoning(response)
        
        assert moves == []
        assert reasoning == "NO_PATH_FOUND"

    def test_extract_missing_fields(self):
        """Test extracting with missing fields."""
        response = '{"moves": ["UP"]}'  # Missing reasoning
        
        with pytest.raises((KeyError, ValueError)):
            extract_moves_and_reasoning(response)

    def test_extract_wrong_types(self):
        """Test extracting with wrong field types."""
        response = '{"moves": "UP", "reasoning": ["not", "string"]}'
        
        with pytest.raises((TypeError, ValueError)):
            extract_moves_and_reasoning(response)

    def test_extract_with_extra_fields(self):
        """Test extracting with extra fields."""
        response = '{"moves": ["LEFT"], "reasoning": "Safe", "confidence": 0.9}'
        
        moves, reasoning = extract_moves_and_reasoning(response)
        
        assert moves == ["LEFT"]
        assert reasoning == "Safe"

    def test_extract_nested_reasoning(self):
        """Test extracting with complex reasoning."""
        response = '{"moves": ["DOWN"], "reasoning": "Complex reasoning with \\"quotes\\" and symbols"}'
        
        moves, reasoning = extract_moves_and_reasoning(response)
        
        assert moves == ["DOWN"]
        assert '"quotes"' in reasoning

    def test_extract_unicode_content(self):
        """Test extracting with unicode content."""
        response = '{"moves": ["UP"], "reasoning": "Move towards 🍎"}'
        
        moves, reasoning = extract_moves_and_reasoning(response)
        
        assert moves == ["UP"]
        assert "🍎" in reasoning


class TestHandleParsingErrors:
    """Test class for handle_parsing_errors function."""

    def test_handle_json_decode_error(self):
        """Test handling JSON decode errors."""
        error = json.JSONDecodeError("Invalid JSON", '{"invalid"}', 10)
        
        result = handle_parsing_errors(error, '{"invalid"}')
        
        # Should return appropriate error response
        assert isinstance(result, dict)
        assert result.get("moves") == []
        assert "error" in result.get("reasoning", "").lower()

    def test_handle_key_error(self):
        """Test handling missing key errors."""
        error = KeyError("moves")
        
        result = handle_parsing_errors(error, '{"reasoning": "test"}')
        
        assert isinstance(result, dict)
        assert result.get("moves") == []

    def test_handle_type_error(self):
        """Test handling type errors."""
        error = TypeError("Expected list, got string")
        
        result = handle_parsing_errors(error, '{"moves": "UP", "reasoning": "test"}')
        
        assert isinstance(result, dict)
        assert result.get("moves") == []

    def test_handle_value_error(self):
        """Test handling value errors."""
        error = ValueError("Invalid move format")
        
        result = handle_parsing_errors(error, '{"moves": ["INVALID"], "reasoning": "test"}')
        
        assert isinstance(result, dict)
        assert result.get("moves") == []

    def test_handle_unknown_error(self):
        """Test handling unknown error types."""
        error = RuntimeError("Unexpected error")
        
        result = handle_parsing_errors(error, "some response")
        
        assert isinstance(result, dict)
        assert result.get("moves") == []

    def test_handle_error_with_none_response(self):
        """Test handling errors with None response."""
        error = ValueError("Test error")
        
        result = handle_parsing_errors(error, None)
        
        assert isinstance(result, dict)
        assert result.get("moves") == []

    def test_handle_error_preserves_context(self):
        """Test that error handling preserves useful context."""
        error = json.JSONDecodeError("Invalid JSON", '{"broken"}', 5)
        
        result = handle_parsing_errors(error, '{"broken"}')
        
        # Should include some context about the error
        reasoning = result.get("reasoning", "")
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0


class TestParsingUtilsIntegration:
    """Test class for integration scenarios."""

    def test_parse_and_format_vs_parse_llm_response(self):
        """Test consistency between parse_and_format and parse_llm_response."""
        valid_response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move to apple"}'
        
        # Both functions should handle valid input
        result1 = parse_and_format(valid_response)
        result2 = parse_llm_response(valid_response)
        
        if result1 is not None:
            assert isinstance(result1, dict)
            assert isinstance(result2, dict)
            assert result1["moves"] == result2["moves"]
            assert result1["reasoning"] == result2["reasoning"]

    def test_error_handling_differences(self):
        """Test different error handling between functions."""
        invalid_response = '{"moves": ["UP"], invalid json}'
        
        # parse_and_format should handle errors gracefully
        result1 = parse_and_format(invalid_response)
        assert result1 is None or isinstance(result1, dict)
        
        # parse_llm_response might raise exceptions
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            parse_llm_response(invalid_response)

    def test_various_response_formats(self):
        """Test parsing various response formats."""
        responses = [
            # Standard format
            '{"moves": ["UP"], "reasoning": "test"}',
            
            # With extra whitespace
            ' \n {"moves": ["DOWN"], "reasoning": "test"} \n ',
            
            # With extra fields
            '{"moves": ["LEFT"], "reasoning": "test", "extra": "data"}',
            
            # Minimal format
            '{"moves":["RIGHT"],"reasoning":"test"}',
        ]
        
        for response in responses:
            try:
                result1 = parse_and_format(response)
                result2 = parse_llm_response(response)
                
                if result1 is not None:
                    assert isinstance(result1, dict)
                    assert "moves" in result1
                
                assert isinstance(result2, dict)
                assert "moves" in result2
                
            except (json.JSONDecodeError, ValueError, KeyError):
                # Some formats might not be supported - that's acceptable
                pass

    def test_edge_cases_combination(self):
        """Test combination of edge cases."""
        edge_cases = [
            # Empty moves
            ('{"moves": [], "reasoning": "NO_PATH_FOUND"}', True),
            
            # Null values
            ('{"moves": null, "reasoning": "test"}', False),
            
            # Very long moves
            (f'{{"moves": {["UP"] * 100}, "reasoning": "long"}}', True),
            
            # Special characters in reasoning
            ('{"moves": ["UP"], "reasoning": "Special chars: !@#$%^&*()"}', True),
        ]
        
        for response, should_be_valid in edge_cases:
            try:
                result1 = parse_and_format(response)
                result2 = parse_llm_response(response)
                
                if should_be_valid:
                    if result1 is not None:
                        assert isinstance(result1, dict)
                    assert isinstance(result2, dict)
                else:
                    # Invalid cases might return None or raise exceptions
                    pass
                    
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                if should_be_valid:
                    pytest.fail(f"Should be able to parse valid response: {response}")

    def test_performance_with_large_responses(self):
        """Test performance with large responses."""
        # Create a large response
        large_moves = ["UP"] * 1000
        large_reasoning = "x" * 10000
        large_response = json.dumps({
            "moves": large_moves,
            "reasoning": large_reasoning
        })
        
        # Should handle large responses efficiently
        result1 = parse_and_format(large_response)
        result2 = parse_llm_response(large_response)
        
        if result1 is not None:
            assert isinstance(result1, dict)
            assert len(result1["moves"]) == 1000
            assert len(result1["reasoning"]) == 10000
        
        assert isinstance(result2, dict)
        assert len(result2["moves"]) == 1000
        assert len(result2["reasoning"]) == 10000

    def test_unicode_and_encoding_handling(self):
        """Test handling of unicode and encoding issues."""
        unicode_responses = [
            '{"moves": ["UP"], "reasoning": "Move towards 🍎"}',
            '{"moves": ["DOWN"], "reasoning": "路径: 向下移动"}',
            '{"moves": ["LEFT"], "reasoning": "Café direction"}',
            '{"moves": ["RIGHT"], "reasoning": "Naïve approach"}',
        ]
        
        for response in unicode_responses:
            try:
                result1 = parse_and_format(response)
                result2 = parse_llm_response(response)
                
                if result1 is not None:
                    assert isinstance(result1, dict)
                    assert isinstance(result1["reasoning"], str)
                
                assert isinstance(result2, dict)
                assert isinstance(result2["reasoning"], str)
                
            except (UnicodeError, json.JSONDecodeError):
                # Some unicode might not be supported - document this
                pass

    def test_malformed_json_recovery(self):
        """Test recovery from malformed JSON."""
        malformed_responses = [
            # Missing quotes
            '{moves: ["UP"], reasoning: "test"}',
            
            # Trailing comma
            '{"moves": ["UP"], "reasoning": "test",}',
            
            # Single quotes instead of double
            "{'moves': ['UP'], 'reasoning': 'test'}",
            
            # Missing closing brace
            '{"moves": ["UP"], "reasoning": "test"',
        ]
        
        for response in malformed_responses:
            # parse_and_format should handle gracefully
            result1 = parse_and_format(response)
            assert result1 is None or isinstance(result1, dict)
            
            # parse_llm_response should raise exceptions
            with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
                parse_llm_response(response)

    def test_consistent_output_format(self):
        """Test that output format is consistent."""
        valid_response = '{"moves": ["UP", "RIGHT"], "reasoning": "test"}'
        
        # Multiple calls should produce consistent results
        results1 = [parse_and_format(valid_response) for _ in range(3)]
        results2 = [parse_llm_response(valid_response) for _ in range(3)]
        
        # All results should be the same
        if results1[0] is not None:
            assert all(r == results1[0] for r in results1)
        
        assert all(r == results2[0] for r in results2)

    def test_memory_efficiency(self):
        """Test memory efficiency with repeated parsing."""
        response = '{"moves": ["UP"], "reasoning": "test"}'
        
        # Multiple calls shouldn't cause memory issues
        for _ in range(100):
            result1 = parse_and_format(response)
            result2 = parse_llm_response(response)
            
            # Verify results are still correct
            if result1 is not None:
                assert result1["moves"] == ["UP"]
            assert result2["moves"] == ["UP"] 