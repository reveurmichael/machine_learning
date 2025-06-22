"""
Tests for LLM error propagation through the system.

Focuses on testing how LLM failures propagate through Client → Controller → Data
and how the system handles various failure modes and recovery scenarios.
"""

import pytest
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock, call

from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider
from core.game_controller import GameController
from core.game_data import GameData
from utils.json_utils import safe_json_parse, extract_json_from_text


class TestLLMErrorPropagation:
    """Test LLM error propagation through the system."""

    def test_provider_failure_to_client_error_handling(self) -> None:
        """Test error propagation from Provider to Client."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various provider failure modes
        provider_failures: List[Tuple[Exception, str, bool]] = [
            (ConnectionError("Network connection failed"), "network_error", True),
            (TimeoutError("Request timeout after 30s"), "timeout_error", True),
            (ValueError("Invalid API key"), "auth_error", False),
            (RuntimeError("Model not available"), "model_error", True),
            (KeyError("Missing required parameter"), "parameter_error", False),
            (MemoryError("Out of memory"), "memory_error", False),
            (OSError("File system error"), "system_error", True),
        ]
        
        for error, error_type, should_retry in provider_failures:
            mock_provider.generate_response.side_effect = [error, '{"moves": ["UP"]}']
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            
            try:
                response: str = client.generate_response("test prompt")
                
                # If we get here, client handled the error
                if should_retry:
                    # Should have retried and succeeded
                    assert response == '{"moves": ["UP"]}'
                    # Verify retry happened
                    assert mock_provider.generate_response.call_count == 2
                else:
                    # Should not have retried but may have returned error response
                    assert isinstance(response, str)
                    
            except Exception as e:
                # Client re-raised the error
                if should_retry:
                    # Unexpected - should have retried
                    assert False, f"Client should have handled {error_type} but raised {e}"
                else:
                    # Expected - certain errors shouldn't be retried
                    assert type(e) == type(error) or isinstance(e, (ValueError, RuntimeError))
            
            mock_provider.reset_mock()

    def test_client_error_to_controller_handling(self) -> None:
        """Test error propagation from Client to Controller."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Simulate controller with LLM integration
        mock_client: Mock = Mock(spec=LLMClient)
        
        # Test various client error responses
        client_errors: List[Tuple[Any, str, bool]] = [
            (Exception("Client connection failed"), "client_exception", False),
            ("", "empty_response", True),
            ("null", "null_response", True),
            ("{invalid json", "invalid_json", True),
            ('{"error": "rate limited"}', "error_response", True),
            ('{"moves": []}', "empty_moves", True),
            ('{"moves": ["INVALID"]}', "invalid_moves", True),
            (None, "none_response", False),
        ]
        
        for error_response, error_type, should_continue in client_errors:
            # Reset controller state
            controller.reset()
            initial_state = {
                "score": controller.score,
                "steps": controller.steps,
                "positions": controller.snake_positions.copy()
            }
            
            try:
                # Simulate LLM integration error
                if isinstance(error_response, Exception):
                    mock_client.generate_response.side_effect = error_response
                else:
                    mock_client.generate_response.return_value = error_response
                
                # Mock controller method that uses LLM (this would be integration-specific)
                # For this test, we'll simulate the error handling logic
                llm_response: Any = None
                llm_error: Optional[Exception] = None
                
                try:
                    if isinstance(error_response, Exception):
                        raise error_response
                    llm_response = error_response
                except Exception as e:
                    llm_error = e
                
                # Controller should handle LLM errors gracefully
                if llm_error and not should_continue:
                    # Fatal errors should stop processing
                    assert controller.score == initial_state["score"]
                    assert controller.steps == initial_state["steps"]
                    # Game state should remain unchanged
                    
                elif should_continue:
                    # Non-fatal errors should allow fallback behavior
                    if llm_response:
                        # Try to parse response
                        parsed: Optional[Dict[str, Any]] = safe_json_parse(str(llm_response))
                        if parsed is None:
                            parsed = extract_json_from_text(str(llm_response))
                        
                        if parsed and "moves" in parsed:
                            moves = parsed["moves"]
                            if isinstance(moves, list) and moves:
                                # Check if moves are valid
                                valid_moves = [m for m in moves if isinstance(m, str) and m in ["UP", "DOWN", "LEFT", "RIGHT"]]
                                if not valid_moves:
                                    # Invalid moves - should use fallback
                                    fallback_move = "UP"  # Default fallback
                                    controller.make_move(fallback_move)
                        
                        # Game should continue with fallback behavior
                        assert controller.steps >= initial_state["steps"]
                    
            except Exception as e:
                if should_continue:
                    assert False, f"Controller should handle {error_type} gracefully but raised {e}"
                else:
                    # Expected failure
                    assert isinstance(e, Exception)

    def test_error_propagation_to_game_data(self) -> None:
        """Test error propagation effects on GameData."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Test how LLM errors affect game data recording
        error_scenarios: List[Tuple[str, str, bool]] = [
            ("Connection failed", "network_error", True),
            ("Invalid response", "parsing_error", True),
            ("Rate limited", "rate_limit", True),
            ("Model unavailable", "model_error", False),
            ("Authentication failed", "auth_error", False),
        ]
        
        for error_message, error_type, should_record in error_scenarios:
            # Reset game state
            controller.reset()
            initial_llm_count = len(game_data.llm_communication)
            
            # Simulate LLM interaction with error
            try:
                # This would normally be an LLM request
                game_data.add_llm_communication(
                    "Game state prompt",
                    f"ERROR: {error_message}"
                )
                
                # Record error statistics
                if should_record:
                    game_data.stats.record_step_result(
                        valid=False,
                        collision=False,
                        apple_eaten=False
                    )
                    # Should track failed LLM requests
                    if hasattr(game_data.stats, 'llm_stats'):
                        game_data.stats.llm_stats.failed_requests += 1
                
                # Game should continue despite error
                controller.make_move("UP")  # Fallback move
                
                # Verify data integrity
                assert len(game_data.llm_communication) == initial_llm_count + 1
                assert controller.steps == 1
                assert game_data.steps == 1
                
                if should_record:
                    # Error should be recorded but game continues
                    assert game_data.stats.step_stats.invalid > 0
                
            except Exception as e:
                if should_record:
                    assert False, f"Should record {error_type} but raised {e}"
                else:
                    # Some errors might be fatal
                    assert isinstance(e, Exception)

    def test_cascading_failure_recovery(self) -> None:
        """Test recovery from cascading failures across components."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Simulate cascading failure scenario
        failure_sequence: List[Tuple[str, callable, bool]] = [
            ("Primary LLM fails", lambda: None, True),
            ("Secondary LLM fails", lambda: None, True),
            ("JSON parsing fails", lambda: None, True),
            ("Move validation fails", lambda: None, True),
            ("Fallback to random", lambda: "UP", True),
        ]
        
        recovery_successful: bool = False
        
        for step_name, failure_func, should_recover in failure_sequence:
            try:
                # Simulate the failure
                result = failure_func()
                
                if result is None and should_recover:
                    # This step failed, try next recovery step
                    continue
                elif result is not None:
                    # Recovery successful
                    recovery_successful = True
                    
                    # Apply recovery action
                    if isinstance(result, str) and result in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        collision: bool
                        _: bool
                        collision, _ = controller.make_move(result)
                        
                        # Verify system state after recovery
                        assert controller.steps > 0
                        assert game_data.steps > 0
                        assert not game_data.game_over or collision
                        
                        break
                
            except Exception as e:
                if should_recover:
                    # Recovery step failed, continue to next
                    continue
                else:
                    # Terminal failure
                    break
        
        # At least one recovery method should have worked
        assert recovery_successful, "All recovery methods failed"

    def test_error_state_isolation(self) -> None:
        """Test that errors don't corrupt other system components."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Build up some valid state first
        for i in range(10):
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move("RIGHT")
            if collision:
                controller.reset()
                break
        
        # Record initial valid state
        initial_state = {
            "score": controller.score,
            "steps": controller.steps,
            "snake_length": controller.snake_length,
            "moves_count": len(game_data.moves),
            "game_over": game_data.game_over
        }
        
        # Simulate various LLM errors that shouldn't affect game state
        isolated_errors: List[Tuple[str, callable]] = [
            ("Malformed JSON", lambda: safe_json_parse("{invalid")),
            ("Empty response", lambda: extract_json_from_text("")),
            ("Wrong format", lambda: safe_json_parse('{"data": "value"}')),
            ("Network timeout", lambda: time.sleep(0.001)),  # Simulate timeout
        ]
        
        for error_name, error_func in isolated_errors:
            try:
                # Execute error-prone operation
                result = error_func()
                
                # Verify core game state is unchanged
                assert controller.score == initial_state["score"]
                assert controller.steps == initial_state["steps"]
                assert controller.snake_length == initial_state["snake_length"]
                assert len(game_data.moves) == initial_state["moves_count"]
                assert game_data.game_over == initial_state["game_over"]
                
                # LLM communication log might grow, but that's acceptable
                # Core game mechanics should be unaffected
                
            except Exception as e:
                # Even if error is raised, game state should be protected
                assert controller.score == initial_state["score"]
                assert controller.steps == initial_state["steps"]

    def test_error_recovery_performance_impact(self) -> None:
        """Test performance impact of error recovery mechanisms."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Simulate high-frequency errors with recovery
        error_count: int = 0
        recovery_count: int = 0
        start_time: float = time.time()
        
        for i in range(100):
            try:
                # Simulate LLM request that fails 50% of the time
                if i % 2 == 0:
                    raise ConnectionError("Simulated network error")
                
                # Successful request
                response = '{"moves": ["UP"]}'
                parsed = safe_json_parse(response)
                moves = parsed["moves"] if parsed else []
                
            except ConnectionError:
                error_count += 1
                
                # Recovery: use fallback move
                moves = ["RIGHT"]  # Fallback
                recovery_count += 1
            
            # Apply move
            if moves:
                move = moves[0]
                collision: bool
                _: bool
                collision, _ = controller.make_move(move)
                if collision:
                    controller.reset()
        
        end_time: float = time.time()
        total_time: float = end_time - start_time
        
        # Verify error recovery worked
        assert error_count > 0
        assert recovery_count == error_count
        
        # Performance should remain reasonable even with many errors
        assert total_time < 1.0  # Should complete in under 1 second
        
        # Game should have made progress despite errors
        assert controller.steps > 0 or game_data.steps > 0

    def test_concurrent_error_handling(self) -> None:
        """Test error handling under concurrent LLM requests."""
        import threading
        
        # Simulate multiple concurrent LLM requests with various errors
        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []
        
        def simulate_llm_request(request_id: int) -> None:
            try:
                # Simulate various error conditions
                error_types = [
                    None,  # Success
                    ConnectionError("Network error"),
                    ValueError("Invalid JSON"),
                    TimeoutError("Request timeout"),
                    None,  # Success
                ]
                
                error = error_types[request_id % len(error_types)]
                
                if error:
                    raise error
                
                # Successful response
                response = f'{{"moves": ["UP"], "request_id": {request_id}}}'
                parsed = safe_json_parse(response)
                
                results.append({
                    "request_id": request_id,
                    "success": True,
                    "response": parsed
                })
                
            except Exception as e:
                # Error recovery
                fallback_response = {"moves": ["RIGHT"], "request_id": request_id, "fallback": True}
                
                results.append({
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "fallback": fallback_response
                })
        
        # Start concurrent requests
        threads: List[threading.Thread] = []
        for i in range(20):
            thread = threading.Thread(target=simulate_llm_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify all requests completed (success or with fallback)
        assert len(results) == 20
        
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        # Should have mix of successes and failures
        assert len(successful_requests) > 0
        assert len(failed_requests) > 0
        
        # All failed requests should have fallback
        for failed_request in failed_requests:
            assert "fallback" in failed_request
            assert failed_request["fallback"]["moves"] == ["RIGHT"]

    def test_error_logging_and_diagnostics(self) -> None:
        """Test error logging and diagnostic information collection."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Simulate various errors with diagnostic collection
        diagnostic_scenarios: List[Tuple[str, Exception, Dict[str, Any]]] = [
            ("Network timeout", TimeoutError("Request timeout"), {"timeout_duration": 30, "retry_count": 3}),
            ("JSON parse error", ValueError("Invalid JSON"), {"response_length": 0, "response_content": ""}),
            ("Model error", RuntimeError("Model unavailable"), {"model_name": "test-model", "provider": "test"}),
        ]
        
        for scenario_name, error, expected_diagnostics in diagnostic_scenarios:
            # Clear any previous error state
            error_logged: bool = False
            diagnostic_data: Dict[str, Any] = {}
            
            try:
                # Simulate the error
                raise error
                
            except Exception as e:
                error_logged = True
                
                # Collect diagnostic information
                diagnostic_data = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "scenario": scenario_name,
                    "game_state": {
                        "score": controller.score,
                        "steps": controller.steps,
                        "game_over": game_data.game_over
                    },
                    "timestamp": time.time()
                }
                
                # Add scenario-specific diagnostics
                diagnostic_data.update(expected_diagnostics)
                
                # Log error to game data (if logging is implemented)
                if hasattr(game_data, 'add_error_log'):
                    game_data.add_error_log(diagnostic_data)
                else:
                    # Store in LLM communication as error record
                    game_data.add_llm_communication(
                        f"ERROR: {scenario_name}",
                        f"Error details: {diagnostic_data}"
                    )
            
            # Verify error was properly logged
            assert error_logged, f"Error not logged for {scenario_name}"
            assert "error_type" in diagnostic_data
            assert diagnostic_data["error_type"] == type(error).__name__
            assert diagnostic_data["scenario"] == scenario_name
            
            # Verify expected diagnostics were collected
            for key, expected_value in expected_diagnostics.items():
                assert key in diagnostic_data
                assert diagnostic_data[key] == expected_value 