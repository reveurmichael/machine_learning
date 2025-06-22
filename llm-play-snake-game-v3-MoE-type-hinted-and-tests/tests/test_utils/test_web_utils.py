"""
Tests for utils.web_utils module.

Focuses on testing web utility functions for HTTP request handling,
session management, response processing, and web framework integration.
"""

import pytest
import time
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import json
from flask import Flask, request

from utils.web_utils import WebUtils


class TestWebUtils:
    """Test web utility functions."""

    def test_http_request_processing(self) -> None:
        """Test HTTP request processing and validation."""
        
        web_utils: WebUtils = WebUtils()
        
        # Mock Flask request objects
        mock_requests: List[Dict[str, Any]] = [
            {
                "method": "GET",
                "path": "/api/game/status",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "SnakeGame-Client/1.0"
                },
                "query_params": {"session_id": "test_session_1"},
                "body": None,
                "expected_valid": True
            },
            {
                "method": "POST",
                "path": "/api/game/start",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test_token_123"
                },
                "query_params": {},
                "body": {
                    "grid_size": 10,
                    "max_games": 5,
                    "llm_provider": "deepseek"
                },
                "expected_valid": True
            },
            {
                "method": "PUT",
                "path": "/api/game/move",
                "headers": {
                    "Content-Type": "application/json"
                },
                "query_params": {"game_id": "game_123"},
                "body": {
                    "move": "UP",
                    "timestamp": time.time()
                },
                "expected_valid": True
            },
            {
                "method": "POST",
                "path": "/api/game/invalid",
                "headers": {
                    "Content-Type": "text/plain"  # Invalid content type
                },
                "query_params": {},
                "body": "invalid_body_format",
                "expected_valid": False
            }
        ]
        
        request_processing_results: List[Dict[str, Any]] = []
        
        for mock_req in mock_requests:
            # Create mock Flask request
            with patch('flask.request') as flask_request:
                flask_request.method = mock_req["method"]
                flask_request.path = mock_req["path"]
                flask_request.headers = mock_req["headers"]
                flask_request.args = mock_req["query_params"]
                
                if mock_req["body"]:
                    flask_request.get_json.return_value = mock_req["body"]
                    flask_request.is_json = True
                else:
                    flask_request.get_json.return_value = None
                    flask_request.is_json = False
                
                # Process request
                processing_result = web_utils.process_http_request(
                    validate_json=True,
                    require_auth=False
                )
                
                is_valid = processing_result["valid"]
                expected_valid = mock_req["expected_valid"]
                
                assert is_valid == expected_valid, f"Request validation mismatch for {mock_req['method']} {mock_req['path']}: expected {expected_valid}, got {is_valid}"
                
                # Verify request data extraction
                if is_valid:
                    assert "request_data" in processing_result, "Request data missing for valid request"
                    request_data = processing_result["request_data"]
                    
                    assert request_data["method"] == mock_req["method"], "Method mismatch"
                    assert request_data["path"] == mock_req["path"], "Path mismatch"
                    assert request_data["headers"] == mock_req["headers"], "Headers mismatch"
                    
                    if mock_req["body"]:
                        assert request_data["body"] == mock_req["body"], "Body mismatch"
                
                request_processing_results.append({
                    "method": mock_req["method"],
                    "path": mock_req["path"],
                    "valid": is_valid,
                    "processing_result": processing_result
                })
        
        assert len(request_processing_results) == 4, "Should process all test requests"
        
        valid_requests = [r for r in request_processing_results if r["valid"]]
        invalid_requests = [r for r in request_processing_results if not r["valid"]]
        
        assert len(valid_requests) == 3, "Should have 3 valid requests"
        assert len(invalid_requests) == 1, "Should have 1 invalid request"

    def test_response_formatting_and_serialization(self) -> None:
        """Test response formatting and JSON serialization."""
        
        web_utils: WebUtils = WebUtils()
        
        # Test various response scenarios
        response_test_cases: List[Dict[str, Any]] = [
            {
                "case_name": "success_response",
                "data": {
                    "game_id": "game_123",
                    "score": 250,
                    "snake_length": 8,
                    "apple_position": [5, 7],
                    "game_active": True
                },
                "status_code": 200,
                "message": "Game move processed successfully",
                "expected_structure": ["success", "data", "message", "timestamp"]
            },
            {
                "case_name": "error_response",
                "data": None,
                "status_code": 400,
                "message": "Invalid move: snake collision detected",
                "error_code": "SNAKE_COLLISION",
                "expected_structure": ["success", "error", "message", "timestamp"]
            },
            {
                "case_name": "list_response",
                "data": [
                    {"game_id": "game_1", "score": 150},
                    {"game_id": "game_2", "score": 200},
                    {"game_id": "game_3", "score": 100}
                ],
                "status_code": 200,
                "message": "Game history retrieved",
                "expected_structure": ["success", "data", "message", "timestamp", "count"]
            },
            {
                "case_name": "complex_nested_response",
                "data": {
                    "session_summary": {
                        "session_id": "session_123",
                        "total_games": 10,
                        "completed_games": 7,
                        "statistics": {
                            "average_score": 175.5,
                            "best_score": 300,
                            "total_steps": 850
                        },
                        "game_results": [
                            {
                                "game_id": f"game_{i}",
                                "final_score": 100 + i * 25,
                                "moves": ["UP", "RIGHT", "DOWN"] * (i + 1)
                            }
                            for i in range(3)
                        ]
                    }
                },
                "status_code": 200,
                "message": "Session summary generated",
                "expected_structure": ["success", "data", "message", "timestamp"]
            }
        ]
        
        response_formatting_results: List[Dict[str, Any]] = []
        
        for test_case in response_test_cases:
            case_name = test_case["case_name"]
            data = test_case["data"]
            status_code = test_case["status_code"]
            message = test_case["message"]
            expected_structure = test_case["expected_structure"]
            error_code = test_case.get("error_code")
            
            # Format response
            if status_code >= 400:
                formatted_response = web_utils.format_error_response(
                    message=message,
                    status_code=status_code,
                    error_code=error_code,
                    details=data
                )
            else:
                formatted_response = web_utils.format_success_response(
                    data=data,
                    message=message,
                    status_code=status_code
                )
            
            # Verify response structure
            for expected_field in expected_structure:
                assert expected_field in formatted_response, f"Missing field '{expected_field}' in {case_name}"
            
            # Verify response content
            if status_code >= 400:
                assert formatted_response["success"] is False, f"Error response should have success=False for {case_name}"
                assert "error" in formatted_response, f"Error response should have error field for {case_name}"
                if error_code:
                    assert formatted_response["error"]["code"] == error_code, f"Error code mismatch for {case_name}"
            else:
                assert formatted_response["success"] is True, f"Success response should have success=True for {case_name}"
                assert "data" in formatted_response, f"Success response should have data field for {case_name}"
                
                if isinstance(data, list):
                    assert "count" in formatted_response, f"List response should have count field for {case_name}"
                    assert formatted_response["count"] == len(data), f"Count mismatch for {case_name}"
            
            # Test JSON serialization
            try:
                json_response = json.dumps(formatted_response)
                deserialized_response = json.loads(json_response)
                
                # Verify serialization roundtrip
                assert deserialized_response == formatted_response, f"JSON serialization roundtrip failed for {case_name}"
                
                serialization_success = True
            except (TypeError, ValueError) as e:
                serialization_success = False
                serialization_error = str(e)
            
            assert serialization_success, f"JSON serialization failed for {case_name}"
            
            response_formatting_results.append({
                "case_name": case_name,
                "formatted_response": formatted_response,
                "json_serializable": serialization_success,
                "status_code": status_code
            })
        
        assert len(response_formatting_results) == 4, "Should format all test responses"

    def test_session_management_and_cookies(self) -> None:
        """Test web session management and cookie handling."""
        
        web_utils: WebUtils = WebUtils()
        
        # Mock session scenarios
        session_test_scenarios: List[Dict[str, Any]] = [
            {
                "scenario_name": "new_session_creation",
                "existing_session": None,
                "user_data": {
                    "user_id": "user_123",
                    "preferences": {
                        "grid_size": 10,
                        "difficulty": "medium",
                        "theme": "dark"
                    }
                },
                "expected_session_fields": ["session_id", "user_id", "created_at", "last_activity", "preferences"]
            },
            {
                "scenario_name": "existing_session_update",
                "existing_session": {
                    "session_id": "session_456",
                    "user_id": "user_456",
                    "created_at": time.time() - 3600,
                    "last_activity": time.time() - 300,
                    "preferences": {
                        "grid_size": 8,
                        "difficulty": "easy"
                    },
                    "game_state": {
                        "current_game": "game_789",
                        "score": 150
                    }
                },
                "user_data": {
                    "preferences": {
                        "grid_size": 12,
                        "difficulty": "hard",
                        "theme": "light"
                    }
                },
                "expected_session_fields": ["session_id", "user_id", "created_at", "last_activity", "preferences", "game_state"]
            },
            {
                "scenario_name": "session_expiration_check",
                "existing_session": {
                    "session_id": "session_789",
                    "user_id": "user_789",
                    "created_at": time.time() - 7200,  # 2 hours ago
                    "last_activity": time.time() - 3900,  # 65 minutes ago (expired)
                    "preferences": {"grid_size": 10}
                },
                "user_data": {},
                "session_timeout": 3600,  # 1 hour timeout
                "should_expire": True
            }
        ]
        
        session_management_results: List[Dict[str, Any]] = []
        
        for scenario in session_test_scenarios:
            scenario_name = scenario["scenario_name"]
            existing_session = scenario.get("existing_session")
            user_data = scenario["user_data"]
            session_timeout = scenario.get("session_timeout", 3600)
            should_expire = scenario.get("should_expire", False)
            expected_fields = scenario.get("expected_session_fields", [])
            
            if should_expire:
                # Test session expiration
                expiration_result = web_utils.check_session_expiration(
                    session=existing_session,
                    timeout_seconds=session_timeout
                )
                
                assert expiration_result["expired"] is True, f"Session should be expired for {scenario_name}"
                assert "expiration_reason" in expiration_result, f"Expiration reason missing for {scenario_name}"
                
                session_management_results.append({
                    "scenario_name": scenario_name,
                    "action": "expiration_check",
                    "result": expiration_result,
                    "session_expired": True
                })
            
            else:
                # Test session creation/update
                if existing_session:
                    session_result = web_utils.update_session(
                        session=existing_session,
                        user_data=user_data
                    )
                    action = "update"
                else:
                    session_result = web_utils.create_session(
                        user_data=user_data
                    )
                    action = "create"
                
                assert session_result["success"] is True, f"Session {action} should succeed for {scenario_name}"
                
                updated_session = session_result["session"]
                
                # Verify session structure
                for expected_field in expected_fields:
                    assert expected_field in updated_session, f"Missing field '{expected_field}' in session for {scenario_name}"
                
                # Verify session timestamps
                assert "last_activity" in updated_session, f"Last activity timestamp missing for {scenario_name}"
                current_time = time.time()
                last_activity = updated_session["last_activity"]
                assert abs(current_time - last_activity) < 5, f"Last activity timestamp not updated for {scenario_name}"
                
                # Verify user data integration
                if "preferences" in user_data:
                    session_prefs = updated_session.get("preferences", {})
                    for key, value in user_data["preferences"].items():
                        assert session_prefs.get(key) == value, f"Preference '{key}' not updated correctly for {scenario_name}"
                
                session_management_results.append({
                    "scenario_name": scenario_name,
                    "action": action,
                    "result": session_result,
                    "updated_session": updated_session,
                    "session_expired": False
                })
        
        assert len(session_management_results) == 3, "Should test all session scenarios"

    def test_request_validation_and_sanitization(self) -> None:
        """Test request data validation and sanitization."""
        
        web_utils: WebUtils = WebUtils()
        
        # Test validation scenarios
        validation_test_cases: List[Dict[str, Any]] = [
            {
                "case_name": "valid_game_start_request",
                "request_data": {
                    "grid_size": 10,
                    "max_games": 5,
                    "llm_provider": "deepseek",
                    "player_name": "TestPlayer"
                },
                "validation_rules": {
                    "grid_size": {"type": int, "min": 5, "max": 20},
                    "max_games": {"type": int, "min": 1, "max": 100},
                    "llm_provider": {"type": str, "allowed": ["deepseek", "mistral", "hunyuan"]},
                    "player_name": {"type": str, "max_length": 50}
                },
                "should_validate": True
            },
            {
                "case_name": "invalid_grid_size",
                "request_data": {
                    "grid_size": 25,  # Too large
                    "max_games": 5,
                    "llm_provider": "deepseek"
                },
                "validation_rules": {
                    "grid_size": {"type": int, "min": 5, "max": 20},
                    "max_games": {"type": int, "min": 1, "max": 100},
                    "llm_provider": {"type": str, "allowed": ["deepseek", "mistral", "hunyuan"]}
                },
                "should_validate": False,
                "expected_errors": ["grid_size"]
            },
            {
                "case_name": "invalid_llm_provider",
                "request_data": {
                    "grid_size": 10,
                    "max_games": 5,
                    "llm_provider": "invalid_provider"  # Not allowed
                },
                "validation_rules": {
                    "grid_size": {"type": int, "min": 5, "max": 20},
                    "max_games": {"type": int, "min": 1, "max": 100},
                    "llm_provider": {"type": str, "allowed": ["deepseek", "mistral", "hunyuan"]}
                },
                "should_validate": False,
                "expected_errors": ["llm_provider"]
            },
            {
                "case_name": "missing_required_fields",
                "request_data": {
                    "grid_size": 10
                    # Missing max_games and llm_provider
                },
                "validation_rules": {
                    "grid_size": {"type": int, "min": 5, "max": 20, "required": True},
                    "max_games": {"type": int, "min": 1, "max": 100, "required": True},
                    "llm_provider": {"type": str, "allowed": ["deepseek", "mistral", "hunyuan"], "required": True}
                },
                "should_validate": False,
                "expected_errors": ["max_games", "llm_provider"]
            }
        ]
        
        validation_results: List[Dict[str, Any]] = []
        
        for test_case in validation_test_cases:
            case_name = test_case["case_name"]
            request_data = test_case["request_data"]
            validation_rules = test_case["validation_rules"]
            should_validate = test_case["should_validate"]
            expected_errors = test_case.get("expected_errors", [])
            
            # Perform validation
            validation_result = web_utils.validate_request_data(
                data=request_data,
                rules=validation_rules
            )
            
            validation_passed = validation_result["valid"]
            assert validation_passed == should_validate, f"Validation outcome mismatch for {case_name}: expected {should_validate}, got {validation_passed}"
            
            if not should_validate:
                assert "errors" in validation_result, f"Validation errors missing for failing case {case_name}"
                validation_errors = validation_result["errors"]
                
                for expected_error in expected_errors:
                    assert any(expected_error in error["field"] for error in validation_errors), f"Expected error for '{expected_error}' not found in {case_name}"
            
            # Test data sanitization
            if validation_passed:
                sanitized_data = web_utils.sanitize_request_data(
                    data=request_data,
                    rules=validation_rules
                )
                
                assert "sanitized_data" in sanitized_data, f"Sanitized data missing for {case_name}"
                
                # Verify sanitized data maintains structure
                clean_data = sanitized_data["sanitized_data"]
                for key, value in request_data.items():
                    if key in validation_rules:
                        assert key in clean_data, f"Key '{key}' missing from sanitized data for {case_name}"
            
            validation_results.append({
                "case_name": case_name,
                "validation_passed": validation_passed,
                "validation_result": validation_result
            })
        
        assert len(validation_results) == 4, "Should test all validation cases"
        
        # Verify distribution of results
        passed_validations = [r for r in validation_results if r["validation_passed"]]
        failed_validations = [r for r in validation_results if not r["validation_passed"]]
        
        assert len(passed_validations) == 1, "Should have 1 valid case"
        assert len(failed_validations) == 3, "Should have 3 invalid cases"

    def test_error_handling_and_logging(self) -> None:
        """Test web error handling and request logging."""
        
        web_utils: WebUtils = WebUtils()
        
        # Mock error scenarios
        error_scenarios: List[Dict[str, Any]] = [
            {
                "error_type": "ValidationError",
                "error_message": "Invalid grid size: must be between 5 and 20",
                "error_context": {
                    "field": "grid_size",
                    "value": 25,
                    "rule": {"min": 5, "max": 20}
                },
                "expected_status_code": 400,
                "expected_error_code": "VALIDATION_ERROR"
            },
            {
                "error_type": "AuthenticationError",
                "error_message": "Invalid authentication token",
                "error_context": {
                    "token": "invalid_token_123",
                    "endpoint": "/api/game/start"
                },
                "expected_status_code": 401,
                "expected_error_code": "AUTH_ERROR"
            },
            {
                "error_type": "RateLimitError",
                "error_message": "Rate limit exceeded: too many requests",
                "error_context": {
                    "limit": "10 requests per minute",
                    "current_rate": 15,
                    "reset_time": time.time() + 60
                },
                "expected_status_code": 429,
                "expected_error_code": "RATE_LIMIT_ERROR"
            },
            {
                "error_type": "InternalServerError",
                "error_message": "Game engine initialization failed",
                "error_context": {
                    "component": "GameController",
                    "error_details": "Failed to initialize grid"
                },
                "expected_status_code": 500,
                "expected_error_code": "INTERNAL_ERROR"
            }
        ]
        
        error_handling_results: List[Dict[str, Any]] = []
        
        for scenario in error_scenarios:
            error_type = scenario["error_type"]
            error_message = scenario["error_message"]
            error_context = scenario["error_context"]
            expected_status = scenario["expected_status_code"]
            expected_error_code = scenario["expected_error_code"]
            
            # Create mock exception
            mock_exception = Exception(error_message)
            mock_exception.error_type = error_type
            mock_exception.context = error_context
            
            # Handle error
            error_response = web_utils.handle_web_error(
                exception=mock_exception,
                request_context={
                    "method": "POST",
                    "path": "/api/test/endpoint",
                    "user_id": "test_user",
                    "timestamp": time.time()
                }
            )
            
            # Verify error response structure
            assert "success" in error_response, f"Success field missing for {error_type}"
            assert error_response["success"] is False, f"Success should be False for {error_type}"
            
            assert "error" in error_response, f"Error field missing for {error_type}"
            error_info = error_response["error"]
            
            assert "code" in error_info, f"Error code missing for {error_type}"
            assert "message" in error_info, f"Error message missing for {error_type}"
            assert "timestamp" in error_response, f"Timestamp missing for {error_type}"
            
            # Verify error mapping
            actual_status = error_response.get("status_code", 500)
            assert actual_status == expected_status, f"Status code mismatch for {error_type}: expected {expected_status}, got {actual_status}"
            
            actual_error_code = error_info["code"]
            assert actual_error_code == expected_error_code, f"Error code mismatch for {error_type}: expected {expected_error_code}, got {actual_error_code}"
            
            # Test error logging
            log_entry = web_utils.log_web_error(
                error_response=error_response,
                request_context={
                    "method": "POST",
                    "path": "/api/test/endpoint",
                    "user_id": "test_user"
                },
                exception=mock_exception
            )
            
            assert log_entry["logged"] is True, f"Error logging failed for {error_type}"
            assert "log_id" in log_entry, f"Log ID missing for {error_type}"
            
            error_handling_results.append({
                "error_type": error_type,
                "error_response": error_response,
                "log_entry": log_entry,
                "status_code": actual_status
            })
        
        assert len(error_handling_results) == 4, "Should handle all error scenarios"
        
        # Verify error status code distribution
        status_codes = [r["status_code"] for r in error_handling_results]
        assert 400 in status_codes, "Should have client error (400)"
        assert 401 in status_codes, "Should have auth error (401)"
        assert 429 in status_codes, "Should have rate limit error (429)"
        assert 500 in status_codes, "Should have server error (500)"
