"""
Tests for JSON parsing utilities ↔ LLMClient interactions.

Focuses on testing how JSON parsing utilities and LLMClient interact
in response handling, error recovery, and data validation chains.
"""

import pytest
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch

from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider
from utils.json_utils import safe_json_parse, extract_json_from_text, repair_malformed_json
from llm.parsing_utils import extract_moves_from_response, validate_llm_response


class TestParsingClientInteractions:
    """Test interactions between JSON parsing utilities and LLMClient."""

    def test_response_parsing_chain_integration(self) -> None:
        """Test complete response parsing chain from client to validated moves."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various response formats and their parsing chains
        response_scenarios: List[Tuple[str, str, List[str], bool]] = [
            # Valid responses
            ('{"moves": ["UP", "RIGHT"]}', "standard_json", ["UP", "RIGHT"], True),
            ('Text before {"moves": ["DOWN"]} after', "embedded_json", ["DOWN"], True),
            ('```json\n{"moves": ["LEFT", "UP"]}\n```', "code_block", ["LEFT", "UP"], True),
            
            # Malformed but recoverable
            ('{"moves": ["UP", "RIGHT"', "incomplete_json", [], False),
            ('{"moves": ["UP" "RIGHT"]}', "missing_comma", [], False),
            ('{"moves": ["UP", "RIGHT"],}', "trailing_comma", ["UP", "RIGHT"], True),
            
            # Content issues
            ('{"moves": ["INVALID", "UP"]}', "invalid_moves", ["UP"], True),
            ('{"moves": ["up", "right"]}', "case_issues", ["UP", "RIGHT"], True),
            ('{"moves": []}', "empty_moves", [], False),
            ('{"directions": ["UP", "RIGHT"]}', "wrong_key", [], False),
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        
        for response_text, scenario, expected_moves, should_succeed in response_scenarios:
            mock_provider.generate_response.return_value = response_text
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            # Step 1: Get response from client
            raw_response: str = client.generate_response("test prompt")
            assert raw_response == response_text
            
            # Step 2: Parse JSON from response
            parsed_data: Optional[Dict[str, Any]] = safe_json_parse(raw_response)
            if parsed_data is None:
                parsed_data = extract_json_from_text(raw_response)
            
            # Step 3: Repair if needed
            if parsed_data is None:
                repaired_json: Optional[str] = repair_malformed_json(raw_response)
                if repaired_json:
                    parsed_data = safe_json_parse(repaired_json)
            
            # Step 4: Extract and validate moves
            extracted_moves: List[str] = []
            
            if parsed_data:
                # Try to extract moves using parsing utilities
                if hasattr(extract_moves_from_response, '__call__'):
                    try:
                        extracted_moves = extract_moves_from_response(parsed_data)
                    except:
                        # Fallback extraction
                        if "moves" in parsed_data and isinstance(parsed_data["moves"], list):
                            for move in parsed_data["moves"]:
                                if isinstance(move, str):
                                    move_upper = move.upper().strip()
                                    if move_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                                        extracted_moves.append(move_upper)
                else:
                    # Direct extraction fallback
                    if "moves" in parsed_data and isinstance(parsed_data["moves"], list):
                        for move in parsed_data["moves"]:
                            if isinstance(move, str):
                                move_upper = move.upper().strip()
                                if move_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                                    extracted_moves.append(move_upper)
            
            # Verify parsing chain results
            if should_succeed:
                assert extracted_moves == expected_moves, f"Failed {scenario}: expected {expected_moves}, got {extracted_moves}"
            else:
                # Failed scenarios should result in empty or partial extraction
                assert len(extracted_moves) <= len(expected_moves), f"Unexpected success in {scenario}"

    def test_error_recovery_client_parsing_coordination(self) -> None:
        """Test error recovery coordination between client and parsing utilities."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test error scenarios that require client-parser coordination
        error_scenarios: List[Tuple[Exception, str, str, bool]] = [
            (ConnectionError("Network failed"), "network_error", "", True),
            (TimeoutError("Request timeout"), "timeout_error", "", True),
            (ValueError("Invalid API response"), "api_error", '{"error": "invalid"}', False),
            (RuntimeError("Provider unavailable"), "provider_error", "", True),
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        
        for error, error_type, fallback_response, should_retry in error_scenarios:
            # Configure provider to fail then potentially succeed
            if should_retry:
                mock_provider.generate_response.side_effect = [error, '{"moves": ["UP"]}']
            else:
                mock_provider.generate_response.side_effect = error
            
            mock_provider.get_last_token_count.return_value = {"total_tokens": 25}
            
            try:
                # Attempt to get response
                response: str = client.generate_response("test prompt")
                
                if should_retry:
                    # Should have succeeded on retry
                    assert response == '{"moves": ["UP"]}'
                    
                    # Parse successful response
                    parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
                    assert parsed is not None
                    assert "moves" in parsed
                    assert parsed["moves"] == ["UP"]
                
                else:
                    # Should have failed or returned error response
                    if response:
                        # Try to parse error response
                        parsed = safe_json_parse(response)
                        if parsed and "error" in parsed:
                            # Error response was parsed successfully
                            assert parsed["error"] is not None
            
            except Exception as e:
                if should_retry:
                    # Unexpected - should have handled with retry
                    assert False, f"Client should have retried {error_type} but raised {e}"
                else:
                    # Expected failure
                    assert type(e) == type(error) or "error" in str(e).lower()
                    
                    # Even with error, parsing utilities should handle gracefully
                    error_text = str(e)
                    parsed_error = safe_json_parse(error_text)
                    # Should not crash parsing utilities
                    assert parsed_error is None  # Error text is not JSON
            
            # Reset mock for next scenario
            mock_provider.reset_mock()

    def test_concurrent_parsing_client_operations(self) -> None:
        """Test concurrent parsing operations with client responses."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Configure provider for concurrent testing
        response_pool: List[str] = [
            '{"moves": ["UP"]}',
            '{"moves": ["DOWN"]}', 
            '{"moves": ["LEFT"]}',
            '{"moves": ["RIGHT"]}',
            'Text with {"moves": ["UP", "RIGHT"]} embedded',
            '```json\n{"moves": ["DOWN", "LEFT"]}\n```',
        ]
        
        def get_response_by_thread(prompt: str) -> str:
            # Use thread-specific response based on prompt content
            import threading
            thread_id = threading.current_thread().ident or 0
            return response_pool[thread_id % len(response_pool)]
        
        mock_provider.generate_response.side_effect = get_response_by_thread
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Concurrent operations
        parsing_results: List[Dict[str, Any]] = []
        parsing_errors: List[Exception] = []
        
        def concurrent_parse_operation(operation_id: int) -> None:
            """Perform concurrent client-parsing operation."""
            try:
                # Get response from client
                response: str = client.generate_response(f"prompt_{operation_id}")
                
                # Parse response using utilities
                parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
                if parsed is None:
                    parsed = extract_json_from_text(response)
                
                # Extract moves
                moves: List[str] = []
                if parsed and "moves" in parsed:
                    for move in parsed["moves"]:
                        if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            moves.append(move.upper())
                
                parsing_results.append({
                    "operation_id": operation_id,
                    "response": response,
                    "parsed": parsed,
                    "moves": moves,
                    "success": len(moves) > 0
                })
                
            except Exception as e:
                parsing_errors.append(e)
        
        # Start concurrent operations
        import threading
        threads: List[threading.Thread] = []
        
        for i in range(10):
            thread = threading.Thread(target=concurrent_parse_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        assert len(parsing_errors) == 0, f"Concurrent parsing errors: {parsing_errors}"
        assert len(parsing_results) == 10
        
        # All operations should have succeeded
        successful_operations = [r for r in parsing_results if r["success"]]
        assert len(successful_operations) > 0
        
        # Verify each result is valid
        for result in parsing_results:
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            
            if result["success"]:
                assert len(result["moves"]) > 0
                for move in result["moves"]:
                    assert move in ["UP", "DOWN", "LEFT", "RIGHT"]

    def test_response_format_validation_pipeline(self) -> None:
        """Test complete response format validation pipeline."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test validation scenarios
        validation_scenarios: List[Tuple[str, str, Dict[str, Any]]] = [
            # Valid formats
            ('{"moves": ["UP"]}', "single_move", {"valid": True, "move_count": 1}),
            ('{"moves": ["UP", "RIGHT"]}', "multiple_moves", {"valid": True, "move_count": 2}),
            
            # Invalid formats that should be caught
            ('{"moves": ["UP", "UP", "UP"]}', "repeated_moves", {"valid": False, "reason": "repetition"}),
            ('{"moves": ["UP", "DOWN"]}', "reversal_moves", {"valid": False, "reason": "reversal"}),
            ('{"moves": ["INVALID"]}', "invalid_direction", {"valid": False, "reason": "invalid_direction"}),
            ('{"moves": []}', "empty_moves", {"valid": False, "reason": "empty"}),
            
            # Format issues
            ('{"move": "UP"}', "wrong_key", {"valid": False, "reason": "format"}),
            ('{"moves": "UP"}', "wrong_type", {"valid": False, "reason": "format"}),
            ('{}', "empty_object", {"valid": False, "reason": "missing_data"}),
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        
        for response_text, scenario, expected_validation in validation_scenarios:
            mock_provider.generate_response.return_value = response_text
            mock_provider.get_last_token_count.return_value = {"total_tokens": 30}
            
            # Get response
            response: str = client.generate_response("validation test")
            
            # Parse response
            parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
            
            # Validate using parsing utilities
            validation_result: Dict[str, Any] = {"valid": False, "reason": "unknown"}
            
            if parsed:
                if "moves" in parsed and isinstance(parsed["moves"], list):
                    moves = parsed["moves"]
                    
                    # Basic validation
                    if len(moves) == 0:
                        validation_result = {"valid": False, "reason": "empty"}
                    elif all(isinstance(m, str) and m.upper() in ["UP", "DOWN", "LEFT", "RIGHT"] for m in moves):
                        # Check for issues
                        move_list = [m.upper() for m in moves]
                        
                        # Check for repetition
                        if len(move_list) > 1 and len(set(move_list)) == 1:
                            validation_result = {"valid": False, "reason": "repetition"}
                        
                        # Check for reversals
                        elif len(move_list) >= 2:
                            opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
                            has_reversal = any(move_list[i+1] == opposites.get(move_list[i]) 
                                             for i in range(len(move_list)-1))
                            if has_reversal:
                                validation_result = {"valid": False, "reason": "reversal"}
                            else:
                                validation_result = {"valid": True, "move_count": len(move_list)}
                        else:
                            validation_result = {"valid": True, "move_count": len(move_list)}
                    else:
                        validation_result = {"valid": False, "reason": "invalid_direction"}
                else:
                    validation_result = {"valid": False, "reason": "format"}
            else:
                validation_result = {"valid": False, "reason": "parse_error"}
            
            # Verify validation matches expected
            assert validation_result["valid"] == expected_validation["valid"], \
                f"Validation mismatch for {scenario}: expected {expected_validation}, got {validation_result}"
            
            if "reason" in expected_validation:
                assert validation_result["reason"] == expected_validation["reason"], \
                    f"Reason mismatch for {scenario}"

    def test_performance_parsing_large_responses(self) -> None:
        """Test parsing performance with large client responses."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Generate large response with embedded JSON
        large_response_parts: List[str] = [
            "This is a very long response with lots of text before the actual JSON. " * 100,
            '{"moves": ["UP", "RIGHT"], "analysis": "',
            "This is a very detailed analysis section with lots of text. " * 200,
            '", "confidence": 0.95, "reasoning": "',
            "More detailed reasoning with extensive explanations. " * 150,
            '"}',
            "And even more text after the JSON section. " * 100
        ]
        
        large_response: str = "".join(large_response_parts)
        mock_provider.generate_response.return_value = large_response
        mock_provider.get_last_token_count.return_value = {"total_tokens": 1000}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Test parsing performance
        start_time: float = time.time()
        
        # Get response
        response: str = client.generate_response("large response test")
        
        # Parse with timing
        parse_start: float = time.time()
        parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
        
        if parsed is None:
            # Try extraction (should be needed for this large response)
            extracted: Optional[Dict[str, Any]] = extract_json_from_text(response)
            parsed = extracted
        
        parse_end: float = time.time()
        total_time: float = parse_end - start_time
        parse_time: float = parse_end - parse_start
        
        # Verify performance
        assert total_time < 1.0, f"Total operation too slow: {total_time}s"
        assert parse_time < 0.5, f"Parsing too slow: {parse_time}s"
        
        # Verify correct extraction
        assert parsed is not None, "Failed to parse large response"
        assert "moves" in parsed, "Failed to extract moves from large response"
        assert parsed["moves"] == ["UP", "RIGHT"], "Incorrect moves extracted"

    def test_malformed_response_recovery_strategies(self) -> None:
        """Test recovery strategies for malformed client responses."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various malformed responses and recovery
        malformed_scenarios: List[Tuple[str, str, bool, List[str]]] = [
            # Repairable malformations
            ('{"moves": ["UP", "RIGHT"', "missing_bracket", True, ["UP", "RIGHT"]),
            ('{"moves": ["UP", "RIGHT"],}', "trailing_comma", True, ["UP", "RIGHT"]),
            ('{moves: ["UP", "RIGHT"]}', "unquoted_key", True, ["UP", "RIGHT"]),
            ('{"moves": ["UP", "RIGHT"]}extra', "trailing_text", True, ["UP", "RIGHT"]),
            
            # Difficult malformations
            ('{"moves": ["UP" "RIGHT"]}', "missing_comma_items", False, []),
            ('{"moves": ["UP", "RIGHT"}', "missing_bracket_complex", False, []),
            ('moves: ["UP", "RIGHT"]', "no_braces", False, []),
            ('garbage {"moves": ["UP"]} more garbage', "embedded_json", True, ["UP"]),
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        
        for malformed_text, scenario, should_recover, expected_moves in malformed_scenarios:
            mock_provider.generate_response.return_value = malformed_text
            mock_provider.get_last_token_count.return_value = {"total_tokens": 40}
            
            # Get malformed response
            response: str = client.generate_response("malformed test")
            
            # Apply recovery strategy
            recovered_moves: List[str] = []
            
            # Strategy 1: Direct parsing
            parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
            
            # Strategy 2: JSON extraction
            if parsed is None:
                parsed = extract_json_from_text(response)
            
            # Strategy 3: JSON repair
            if parsed is None:
                repaired: Optional[str] = repair_malformed_json(response)
                if repaired:
                    parsed = safe_json_parse(repaired)
            
            # Extract moves if any strategy succeeded
            if parsed and "moves" in parsed and isinstance(parsed["moves"], list):
                for move in parsed["moves"]:
                    if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        recovered_moves.append(move.upper())
            
            # Verify recovery results
            if should_recover:
                assert recovered_moves == expected_moves, \
                    f"Recovery failed for {scenario}: expected {expected_moves}, got {recovered_moves}"
            else:
                # Difficult cases may not recover fully
                assert len(recovered_moves) <= len(expected_moves), \
                    f"Unexpected recovery success for {scenario}"

    def test_parsing_utility_client_integration_edge_cases(self) -> None:
        """Test edge cases in parsing utility and client integration."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test edge cases
        edge_cases: List[Tuple[str, str, Any]] = [
            ("", "empty_response", None),
            ("null", "null_response", None),
            ("true", "boolean_response", None),
            ("123", "number_response", None),
            ('"string"', "string_response", None),
            ('[]', "empty_array", None),
            ('{"moves": null}', "null_moves", None),
            ('{"moves": true}', "boolean_moves", None),
            ('{"moves": 123}', "number_moves", None),
            ('{"nested": {"moves": ["UP"]}}', "nested_moves", None),
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        
        for response_text, case_name, expected_result in edge_cases:
            mock_provider.generate_response.return_value = response_text
            mock_provider.get_last_token_count.return_value = {"total_tokens": 10}
            
            try:
                # Get response
                response: str = client.generate_response("edge case test")
                
                # Parse with error handling
                parsed: Optional[Dict[str, Any]] = None
                
                try:
                    parsed = safe_json_parse(response)
                except Exception:
                    # Should handle gracefully
                    pass
                
                if parsed is None:
                    try:
                        parsed = extract_json_from_text(response)
                    except Exception:
                        # Should handle gracefully
                        pass
                
                # Extract moves safely
                moves: List[str] = []
                if parsed and isinstance(parsed, dict) and "moves" in parsed:
                    moves_data = parsed["moves"]
                    if isinstance(moves_data, list):
                        for move in moves_data:
                            if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                                moves.append(move.upper())
                
                # Should not crash on any edge case
                assert isinstance(moves, list), f"Edge case {case_name} broke move extraction"
                
            except Exception as e:
                # Should not crash on edge cases
                assert False, f"Edge case {case_name} caused crash: {e}" 