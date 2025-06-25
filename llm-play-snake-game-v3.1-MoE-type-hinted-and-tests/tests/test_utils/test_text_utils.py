"""
Tests for utils.text_utils module.

Focuses on testing text processing utilities for string manipulation,
formatting, parsing, validation, and text analysis.
"""

import pytest
import re
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

from utils.text_utils import TextUtils


class TestTextUtils:
    """Test text utility functions."""

    def test_string_formatting_and_templates(self) -> None:
        """Test string formatting and template processing."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test template formatting scenarios
        template_scenarios: List[Dict[str, Any]] = [
            {
                "template": "Player {player_name} scored {score} points in game {game_id}",
                "variables": {
                    "player_name": "TestPlayer",
                    "score": 250,
                    "game_id": "game_123"
                },
                "expected": "Player TestPlayer scored 250 points in game game_123"
            },
            {
                "template": "Game status: {status} | Duration: {duration:.2f}s | Score: {score:,}",
                "variables": {
                    "status": "completed",
                    "duration": 45.7834,
                    "score": 12345
                },
                "expected": "Game status: completed | Duration: 45.78s | Score: 12,345"
            },
            {
                "template": "Session {session_id}: {completed_games}/{total_games} games completed ({percentage:.1f}%)",
                "variables": {
                    "session_id": "sess_456",
                    "completed_games": 7,
                    "total_games": 10,
                    "percentage": 70.0
                },
                "expected": "Session sess_456: 7/10 games completed (70.0%)"
            },
            {
                "template": "LLM Provider: {provider} | Model: {model} | Response time: {response_time}ms",
                "variables": {
                    "provider": "deepseek",
                    "model": "deepseek-coder",
                    "response_time": 150
                },
                "expected": "LLM Provider: deepseek | Model: deepseek-coder | Response time: 150ms"
            }
        ]
        
        formatting_results: List[Dict[str, Any]] = []
        
        for scenario in template_scenarios:
            template = scenario["template"]
            variables = scenario["variables"]
            expected = scenario["expected"]
            
            # Format template
            formatted_result = text_utils.format_template(
                template=template,
                variables=variables
            )
            
            assert formatted_result["success"] is True, f"Template formatting should succeed for: {template}"
            
            formatted_text = formatted_result["formatted_text"]
            assert formatted_text == expected, f"Template formatting mismatch:\nExpected: {expected}\nActual: {formatted_text}"
            
            formatting_results.append({
                "template": template,
                "formatted_text": formatted_text,
                "success": True
            })
        
        assert len(formatting_results) == 4, "Should format all templates successfully"
        
        # Test template with missing variables
        missing_var_result = text_utils.format_template(
            template="Player {player_name} scored {missing_variable} points",
            variables={"player_name": "TestPlayer"}
        )
        
        assert missing_var_result["success"] is False, "Should fail with missing variables"
        assert "missing_variables" in missing_var_result, "Should report missing variables"

    def test_text_parsing_and_extraction(self) -> None:
        """Test text parsing and data extraction functions."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test parsing scenarios
        parsing_scenarios: List[Dict[str, Any]] = [
            {
                "text": "Game ID: game_123, Score: 250, Steps: 125, Duration: 45.5s",
                "pattern": r"Game ID: (\w+), Score: (\d+), Steps: (\d+), Duration: ([\d.]+)s",
                "expected_groups": ["game_123", "250", "125", "45.5"],
                "expected_dict": {
                    "game_id": "game_123",
                    "score": 250,
                    "steps": 125,
                    "duration": 45.5
                }
            },
            {
                "text": "Session sess_456 completed 7/10 games (70.0%) in 15.2 minutes",
                "pattern": r"Session (\w+) completed (\d+)/(\d+) games \(([\d.]+)%\) in ([\d.]+) minutes",
                "expected_groups": ["sess_456", "7", "10", "70.0", "15.2"],
                "expected_dict": {
                    "session_id": "sess_456",
                    "completed": 7,
                    "total": 10,
                    "percentage": 70.0,
                    "duration_minutes": 15.2
                }
            },
            {
                "text": "LLM Response: Provider=deepseek, Model=deepseek-coder, Tokens=150, Time=200ms",
                "pattern": r"LLM Response: Provider=(\w+), Model=([\w-]+), Tokens=(\d+), Time=(\d+)ms",
                "expected_groups": ["deepseek", "deepseek-coder", "150", "200"],
                "expected_dict": {
                    "provider": "deepseek",
                    "model": "deepseek-coder", 
                    "tokens": 150,
                    "response_time": 200
                }
            }
        ]
        
        parsing_results: List[Dict[str, Any]] = []
        
        for scenario in parsing_scenarios:
            text = scenario["text"]
            pattern = scenario["pattern"]
            expected_groups = scenario["expected_groups"]
            expected_dict = scenario["expected_dict"]
            
            # Test regex extraction
            extraction_result = text_utils.extract_with_regex(
                text=text,
                pattern=pattern
            )
            
            assert extraction_result["success"] is True, f"Regex extraction should succeed for: {text}"
            
            extracted_groups = extraction_result["groups"]
            assert extracted_groups == expected_groups, f"Extracted groups mismatch for: {text}"
            
            # Test structured parsing
            if expected_dict:
                structured_result = text_utils.parse_structured_text(
                    text=text,
                    field_patterns=expected_dict,
                    convert_types=True
                )
                
                assert structured_result["success"] is True, f"Structured parsing should succeed for: {text}"
                
                parsed_data = structured_result["parsed_data"]
                
                # Verify data types and values
                for key, expected_value in expected_dict.items():
                    assert key in parsed_data, f"Key '{key}' missing from parsed data"
                    
                    parsed_value = parsed_data[key]
                    assert parsed_value == expected_value, f"Value mismatch for '{key}': expected {expected_value}, got {parsed_value}"
                    assert type(parsed_value) == type(expected_value), f"Type mismatch for '{key}': expected {type(expected_value)}, got {type(parsed_value)}"
            
            parsing_results.append({
                "text": text,
                "extraction_result": extraction_result,
                "success": True
            })
        
        assert len(parsing_results) == 3, "Should parse all text scenarios successfully"

    def test_text_validation_and_sanitization(self) -> None:
        """Test text validation and sanitization functions."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test validation scenarios
        validation_scenarios: List[Dict[str, Any]] = [
            {
                "text": "ValidPlayerName123",
                "validation_type": "player_name",
                "rules": {
                    "min_length": 3,
                    "max_length": 20,
                    "allowed_chars": r"[a-zA-Z0-9_]",
                    "required": True
                },
                "expected_valid": True
            },
            {
                "text": "Valid Session ID",
                "validation_type": "session_name",
                "rules": {
                    "min_length": 5,
                    "max_length": 50,
                    "allowed_chars": r"[a-zA-Z0-9\s_-]",
                    "required": True
                },
                "expected_valid": True
            },
            {
                "text": "a",  # Too short
                "validation_type": "player_name",
                "rules": {
                    "min_length": 3,
                    "max_length": 20,
                    "allowed_chars": r"[a-zA-Z0-9_]",
                    "required": True
                },
                "expected_valid": False
            },
            {
                "text": "Invalid@Name!",  # Invalid characters
                "validation_type": "player_name",
                "rules": {
                    "min_length": 3,
                    "max_length": 20,
                    "allowed_chars": r"[a-zA-Z0-9_]",
                    "required": True
                },
                "expected_valid": False
            },
            {
                "text": "",  # Empty required field
                "validation_type": "required_field",
                "rules": {
                    "min_length": 1,
                    "required": True
                },
                "expected_valid": False
            }
        ]
        
        validation_results: List[Dict[str, Any]] = []
        
        for scenario in validation_scenarios:
            text = scenario["text"]
            validation_type = scenario["validation_type"]
            rules = scenario["rules"]
            expected_valid = scenario["expected_valid"]
            
            # Validate text
            validation_result = text_utils.validate_text(
                text=text,
                rules=rules
            )
            
            actual_valid = validation_result["valid"]
            assert actual_valid == expected_valid, f"Validation mismatch for '{text}' ({validation_type}): expected {expected_valid}, got {actual_valid}"
            
            if not actual_valid:
                assert "validation_errors" in validation_result, f"Validation errors missing for invalid text: {text}"
                errors = validation_result["validation_errors"]
                assert len(errors) > 0, f"Should have validation errors for invalid text: {text}"
            
            validation_results.append({
                "text": text,
                "validation_type": validation_type,
                "valid": actual_valid,
                "validation_result": validation_result
            })
        
        # Test text sanitization
        sanitization_scenarios: List[Dict[str, Any]] = [
            {
                "text": "  Player Name  ",  # Trim whitespace
                "sanitization_type": "trim",
                "expected": "Player Name"
            },
            {
                "text": "Player@Name#123!",  # Remove special characters
                "sanitization_type": "alphanumeric_only",
                "expected": "PlayerName123"
            },
            {
                "text": "UPPERCASE TEXT",  # Convert to lowercase
                "sanitization_type": "lowercase",
                "expected": "uppercase text"
            },
            {
                "text": "multiple    spaces   here",  # Normalize spaces
                "sanitization_type": "normalize_spaces",
                "expected": "multiple spaces here"
            },
            {
                "text": "<script>alert('xss')</script>Normal Text",  # Remove HTML/Script tags
                "sanitization_type": "remove_html",
                "expected": "Normal Text"
            }
        ]
        
        sanitization_results: List[Dict[str, Any]] = []
        
        for scenario in sanitization_scenarios:
            text = scenario["text"]
            sanitization_type = scenario["sanitization_type"]
            expected = scenario["expected"]
            
            # Sanitize text
            sanitization_result = text_utils.sanitize_text(
                text=text,
                sanitization_type=sanitization_type
            )
            
            assert sanitization_result["success"] is True, f"Sanitization should succeed for: {text}"
            
            sanitized_text = sanitization_result["sanitized_text"]
            assert sanitized_text == expected, f"Sanitization mismatch for '{text}' ({sanitization_type}): expected '{expected}', got '{sanitized_text}'"
            
            sanitization_results.append({
                "original_text": text,
                "sanitization_type": sanitization_type,
                "sanitized_text": sanitized_text
            })
        
        assert len(validation_results) == 5, "Should validate all text scenarios"
        assert len(sanitization_results) == 5, "Should sanitize all text scenarios"

    def test_text_analysis_and_metrics(self) -> None:
        """Test text analysis and metrics calculation."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test text analysis scenarios
        analysis_scenarios: List[Dict[str, Any]] = [
            {
                "text": "This is a sample text for analysis. It contains multiple sentences and words.",
                "expected_metrics": {
                    "character_count": 80,
                    "word_count": 13,
                    "sentence_count": 2,
                    "average_word_length": 6.15,  # Approximate
                    "readability_score": "intermediate"
                }
            },
            {
                "text": "Short text.",
                "expected_metrics": {
                    "character_count": 11,
                    "word_count": 2,
                    "sentence_count": 1,
                    "average_word_length": 4.5,
                    "readability_score": "simple"
                }
            },
            {
                "text": "This is a significantly more complex textual composition containing numerous sophisticated vocabulary elements and intricate grammatical structures that demonstrate advanced linguistic complexity.",
                "expected_metrics": {
                    "character_count": 189,
                    "word_count": 22,
                    "sentence_count": 1,
                    "average_word_length": 7.5,  # Approximate
                    "readability_score": "complex"
                }
            }
        ]
        
        analysis_results: List[Dict[str, Any]] = []
        
        for scenario in analysis_scenarios:
            text = scenario["text"]
            expected_metrics = scenario["expected_metrics"]
            
            # Analyze text
            analysis_result = text_utils.analyze_text(
                text=text,
                include_metrics=["character_count", "word_count", "sentence_count", "average_word_length", "readability_score"]
            )
            
            assert analysis_result["success"] is True, f"Text analysis should succeed for: {text[:50]}..."
            
            metrics = analysis_result["metrics"]
            
            # Verify metrics
            for metric_name, expected_value in expected_metrics.items():
                assert metric_name in metrics, f"Metric '{metric_name}' missing from analysis"
                
                actual_value = metrics[metric_name]
                
                if isinstance(expected_value, float):
                    # Allow small tolerance for floating point calculations
                    assert abs(actual_value - expected_value) < 1.0, f"Metric '{metric_name}' mismatch: expected ~{expected_value}, got {actual_value}"
                elif isinstance(expected_value, int):
                    assert actual_value == expected_value, f"Metric '{metric_name}' mismatch: expected {expected_value}, got {actual_value}"
                else:
                    assert actual_value == expected_value, f"Metric '{metric_name}' mismatch: expected {expected_value}, got {actual_value}"
            
            analysis_results.append({
                "text": text,
                "metrics": metrics,
                "analysis_result": analysis_result
            })
        
        assert len(analysis_results) == 3, "Should analyze all text scenarios"

    def test_text_search_and_replacement(self) -> None:
        """Test text search and replacement functions."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test search and replace scenarios
        search_replace_scenarios: List[Dict[str, Any]] = [
            {
                "text": "Player TestPlayer completed game game_123 with score 250",
                "search_pattern": r"Player (\w+) completed game (\w+) with score (\d+)",
                "replacement": "Game {game_id}: {player} scored {score} points",
                "replacement_mapping": {
                    "player": 1,  # First capture group
                    "game_id": 2,  # Second capture group  
                    "score": 3    # Third capture group
                },
                "expected": "Game game_123: TestPlayer scored 250 points"
            },
            {
                "text": "Session sess_456 - Status: active - Games: 7/10",
                "search_pattern": r"Session (\w+) - Status: (\w+) - Games: (\d+)/(\d+)",
                "replacement": "Session {session_id} is {status} with {completed} of {total} games done",
                "replacement_mapping": {
                    "session_id": 1,
                    "status": 2,
                    "completed": 3,
                    "total": 4
                },
                "expected": "Session sess_456 is active with 7 of 10 games done"
            },
            {
                "text": "LLM: deepseek | Response: 200ms | Tokens: 150",
                "search_pattern": r"LLM: (\w+) \| Response: (\d+)ms \| Tokens: (\d+)",
                "replacement": "Provider {provider} responded in {time}ms using {tokens} tokens",
                "replacement_mapping": {
                    "provider": 1,
                    "time": 2,
                    "tokens": 3
                },
                "expected": "Provider deepseek responded in 200ms using 150 tokens"
            }
        ]
        
        search_replace_results: List[Dict[str, Any]] = []
        
        for scenario in search_replace_scenarios:
            text = scenario["text"]
            search_pattern = scenario["search_pattern"]
            replacement = scenario["replacement"]
            replacement_mapping = scenario["replacement_mapping"]
            expected = scenario["expected"]
            
            # Perform search and replace
            replace_result = text_utils.search_and_replace(
                text=text,
                search_pattern=search_pattern,
                replacement_template=replacement,
                group_mapping=replacement_mapping
            )
            
            assert replace_result["success"] is True, f"Search and replace should succeed for: {text}"
            
            replaced_text = replace_result["replaced_text"]
            assert replaced_text == expected, f"Search and replace mismatch:\nOriginal: {text}\nExpected: {expected}\nActual: {replaced_text}"
            
            # Verify match information
            assert "matches_found" in replace_result, "Matches found count missing"
            assert replace_result["matches_found"] > 0, "Should find at least one match"
            
            search_replace_results.append({
                "original_text": text,
                "replaced_text": replaced_text,
                "matches_found": replace_result["matches_found"]
            })
        
        # Test multiple replacements in same text
        multi_replace_text = "Player1 scored 100, Player2 scored 200, Player3 scored 150"
        multi_replace_result = text_utils.search_and_replace(
            text=multi_replace_text,
            search_pattern=r"Player(\d+) scored (\d+)",
            replacement_template="P{player_num}:{score}pts",
            group_mapping={"player_num": 1, "score": 2},
            replace_all=True
        )
        
        assert multi_replace_result["success"] is True, "Multi-replacement should succeed"
        assert multi_replace_result["matches_found"] == 3, "Should find 3 matches"
        
        expected_multi = "P1:100pts, P2:200pts, P3:150pts"
        actual_multi = multi_replace_result["replaced_text"]
        assert actual_multi == expected_multi, f"Multi-replacement mismatch: expected '{expected_multi}', got '{actual_multi}'"
        
        assert len(search_replace_results) == 3, "Should perform all search and replace operations"

    def test_text_encoding_and_escaping(self) -> None:
        """Test text encoding and escaping functions."""
        
        text_utils: TextUtils = TextUtils()
        
        # Test encoding scenarios
        encoding_scenarios: List[Dict[str, Any]] = [
            {
                "text": "Hello World!",
                "encoding_type": "url",
                "expected": "Hello%20World%21"
            },
            {
                "text": "Player & Game < Session >",
                "encoding_type": "html",
                "expected": "Player &amp; Game &lt; Session &gt;"
            },
            {
                "text": 'Text with "quotes" and \'apostrophes\'',
                "encoding_type": "json",
                "expected": 'Text with \\"quotes\\" and \'apostrophes\''
            },
            {
                "text": "Line 1\nLine 2\tTabbed\rCarriage Return",
                "encoding_type": "escape_whitespace",
                "expected": "Line 1\\nLine 2\\tTabbed\\rCarriage Return"
            }
        ]
        
        encoding_results: List[Dict[str, Any]] = []
        
        for scenario in encoding_scenarios:
            text = scenario["text"]
            encoding_type = scenario["encoding_type"]
            expected = scenario["expected"]
            
            # Encode text
            encoding_result = text_utils.encode_text(
                text=text,
                encoding_type=encoding_type
            )
            
            assert encoding_result["success"] is True, f"Text encoding should succeed for: {text}"
            
            encoded_text = encoding_result["encoded_text"]
            assert encoded_text == expected, f"Encoding mismatch for '{text}' ({encoding_type}): expected '{expected}', got '{encoded_text}'"
            
            # Test decoding (roundtrip)
            decoding_result = text_utils.decode_text(
                text=encoded_text,
                encoding_type=encoding_type
            )
            
            assert decoding_result["success"] is True, f"Text decoding should succeed for: {encoded_text}"
            
            decoded_text = decoding_result["decoded_text"]
            assert decoded_text == text, f"Roundtrip encoding/decoding failed for '{text}'"
            
            encoding_results.append({
                "original_text": text,
                "encoding_type": encoding_type,
                "encoded_text": encoded_text,
                "decoded_text": decoded_text,
                "roundtrip_success": True
            })
        
        assert len(encoding_results) == 4, "Should encode/decode all text scenarios"
        
        # Verify all roundtrips succeeded
        failed_roundtrips = [r for r in encoding_results if not r["roundtrip_success"]]
        assert len(failed_roundtrips) == 0, "All encoding/decoding roundtrips should succeed"
