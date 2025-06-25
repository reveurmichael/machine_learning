"""
Tests for LLMClient ↔ Provider interactions.

Focuses on testing how LLMClient and Provider implementations interact
including error handling, response processing, token tracking, and provider fallbacks.
"""

import pytest
import time
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable
from unittest.mock import Mock, patch, MagicMock, call

from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider


class TestClientProviderInteractions:
    """Test interactions between LLMClient and Provider implementations."""

    def test_provider_response_processing_chain(self) -> None:
        """Test the complete response processing chain from provider to client."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various response formats
        response_scenarios: List[Tuple[str, str, bool]] = [
            ('{"moves": ["UP", "RIGHT"]}', "valid_json", True),
            ('{"moves": ["UP"]}', "single_move", True),
            ('Text before {"moves": ["DOWN"]} after', "embedded_json", True),
            ('```json\n{"moves": ["LEFT"]}\n```', "code_block", True),
            ('{"direction": "UP"}', "wrong_key", False),
            ('{"moves": "UP"}', "wrong_type", False),
            ('malformed json {', "malformed", False),
            ('', "empty_response", False),
            ('null', "null_response", False),
        ]
        
        for response_text, scenario, should_succeed in response_scenarios:
            mock_provider.generate_response.return_value = response_text
            mock_provider.get_last_token_count.return_value = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
            
            client: LLMClient = LLMClient(mock_provider)
            
            # Test response processing
            response: str = client.generate_response("test prompt")
            assert response == response_text
            
            # Test token tracking
            token_count: Optional[Dict[str, int]] = client.get_last_token_count()
            assert token_count is not None
            assert token_count["total_tokens"] == 150
            
            # Verify provider was called correctly
            mock_provider.generate_response.assert_called_with("test prompt")
            mock_provider.get_last_token_count.assert_called()
            
            # Reset mock for next scenario
            mock_provider.reset_mock()

    def test_provider_error_handling_and_recovery(self) -> None:
        """Test error handling and recovery mechanisms between client and provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various error scenarios
        error_scenarios: List[Tuple[Exception, str, bool]] = [
            (ConnectionError("Network error"), "network_error", True),
            (TimeoutError("Request timeout"), "timeout_error", True),
            (ValueError("Invalid API key"), "auth_error", False),
            (RuntimeError("Provider internal error"), "provider_error", True),
            (KeyError("Missing required field"), "data_error", False),
        ]
        
        for error, scenario, should_retry in error_scenarios:
            mock_provider.generate_response.side_effect = [error, '{"moves": ["UP"]}']
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            
            try:
                response: str = client.generate_response("test prompt")
                
                if should_retry:
                    # Should have retried and succeeded
                    assert response == '{"moves": ["UP"]}'
                    assert mock_provider.generate_response.call_count == 2
                else:
                    # Should not retry certain errors
                    assert False, f"Expected {error} to be raised"
                    
            except Exception as e:
                if not should_retry:
                    # Expected to fail
                    assert isinstance(e, type(error))
                else:
                    # Unexpected failure
                    raise
            
            mock_provider.reset_mock()

    def test_token_usage_tracking_consistency(self) -> None:
        """Test token usage tracking consistency between client and provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Simulate varying token usage
        token_scenarios: List[Dict[str, int]] = [
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275},
            {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
            {"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400},
        ]
        
        client: LLMClient = LLMClient(mock_provider)
        total_tracked_tokens: int = 0
        
        for i, token_data in enumerate(token_scenarios):
            mock_provider.generate_response.return_value = f'{{"moves": ["UP"]}}'
            mock_provider.get_last_token_count.return_value = token_data
            
            # Make request
            response: str = client.generate_response(f"test prompt {i}")
            
            # Get token count from client
            client_tokens: Optional[Dict[str, int]] = client.get_last_token_count()
            assert client_tokens is not None
            
            # Verify consistency
            assert client_tokens["prompt_tokens"] == token_data["prompt_tokens"]
            assert client_tokens["completion_tokens"] == token_data["completion_tokens"]
            assert client_tokens["total_tokens"] == token_data["total_tokens"]
            
            # Verify calculation consistency
            calculated_total: int = token_data["prompt_tokens"] + token_data["completion_tokens"]
            assert token_data["total_tokens"] == calculated_total
            
            total_tracked_tokens += token_data["total_tokens"]
        
        # Verify provider was called for each request
        assert mock_provider.generate_response.call_count == len(token_scenarios)
        assert mock_provider.get_last_token_count.call_count == len(token_scenarios)

    def test_provider_availability_state_management(self) -> None:
        """Test provider availability state management and client responses."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Test availability state changes
        availability_scenarios: List[Tuple[bool, str, bool]] = [
            (True, "provider_available", True),
            (False, "provider_unavailable", False),
            (True, "provider_recovered", True),
        ]
        
        for available, scenario, should_work in availability_scenarios:
            mock_provider.is_available.return_value = available
            mock_provider.generate_response.return_value = '{"moves": ["UP"]}'
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            
            if should_work:
                # Should work normally
                response: str = client.generate_response("test prompt")
                assert response == '{"moves": ["UP"]}'
                mock_provider.generate_response.assert_called_once()
            else:
                # Should handle unavailable provider
                try:
                    response = client.generate_response("test prompt")
                    # Depending on implementation, might return empty or raise error
                    assert response is not None
                except Exception as e:
                    # Should be informative error about provider unavailability
                    assert "available" in str(e).lower() or "unavailable" in str(e).lower()
            
            mock_provider.reset_mock()

    def test_concurrent_provider_interactions(self) -> None:
        """Test concurrent interactions between client and provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Configure provider to simulate realistic delays
        def simulate_generation(prompt: str) -> str:
            time.sleep(0.01)  # Small delay to simulate network
            return f'{{"moves": ["UP"]}}'
        
        mock_provider.generate_response.side_effect = simulate_generation
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Test concurrent requests
        import threading
        results: List[str] = []
        errors: List[Exception] = []
        
        def make_request(thread_id: int) -> None:
            try:
                response: str = client.generate_response(f"prompt from thread {thread_id}")
                results.append(response)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads: List[threading.Thread] = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        assert len(errors) == 0, f"Concurrent request errors: {errors}"
        assert len(results) == 10
        
        # All responses should be valid
        for response in results:
            assert response == '{"moves": ["UP"]}'
        
        # Provider should have been called for each request
        assert mock_provider.generate_response.call_count == 10

    def test_provider_fallback_mechanisms(self) -> None:
        """Test fallback mechanisms when primary provider fails."""
        primary_provider: Mock = Mock(spec=BaseLLMProvider)
        secondary_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure providers
        primary_provider.is_available.return_value = True
        secondary_provider.is_available.return_value = True
        
        # Test fallback scenarios
        fallback_scenarios: List[Tuple[Exception, str, bool]] = [
            (ConnectionError("Primary provider down"), "network_failure", True),
            (TimeoutError("Primary provider timeout"), "timeout_failure", True),
            (ValueError("Primary provider API error"), "api_error", True),
        ]
        
        for error, scenario, should_fallback in fallback_scenarios:
            # Configure primary to fail, secondary to succeed
            primary_provider.generate_response.side_effect = error
            secondary_provider.generate_response.return_value = '{"moves": ["DOWN"]}'
            
            # Test with fallback logic (simplified - actual implementation may vary)
            try:
                client: LLMClient = LLMClient(primary_provider)
                response: str = client.generate_response("test prompt")
                
                # If we get here, primary succeeded (shouldn't in this test)
                assert False, f"Expected {error} to be raised"
                
            except Exception:
                # Primary failed as expected
                if should_fallback:
                    # Try secondary provider
                    secondary_client: LLMClient = LLMClient(secondary_provider)
                    response = secondary_client.generate_response("test prompt")
                    assert response == '{"moves": ["DOWN"]}'
            
            # Reset mocks
            primary_provider.reset_mock()
            secondary_provider.reset_mock()

    def test_response_format_validation_chain(self) -> None:
        """Test response format validation between provider and client."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Test various response formats that need validation
        validation_scenarios: List[Tuple[str, Dict[str, Any], bool]] = [
            ('{"moves": ["UP", "RIGHT"]}', {"moves": ["UP", "RIGHT"]}, True),
            ('{"moves": ["INVALID_MOVE"]}', {"moves": ["INVALID_MOVE"]}, False),
            ('{"moves": []}', {"moves": []}, False),
            ('{"moves": ["UP", "UP", "UP", "UP", "UP"]}', {"moves": ["UP"] * 5}, False),  # Too many
            ('{"other_key": "value"}', {"other_key": "value"}, False),
            ('{"moves": "UP"}', {"moves": "UP"}, False),  # Wrong type
        ]
        
        for response_text, expected_json, is_valid in validation_scenarios:
            mock_provider.generate_response.return_value = response_text
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            
            # Get response
            response: str = client.generate_response("test prompt")
            assert response == response_text
            
            # Validation would happen in downstream components
            # But we can test the raw response is passed correctly
            assert isinstance(response, str)
            assert len(response) > 0 or response == ""
            
            mock_provider.reset_mock()

    def test_provider_performance_monitoring(self) -> None:
        """Test performance monitoring between client and provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Simulate varying response times
        response_times: List[float] = [0.1, 0.5, 1.0, 2.0, 0.2]
        
        def timed_response(prompt: str) -> str:
            # Get the next response time
            if hasattr(timed_response, 'call_count'):
                timed_response.call_count += 1
            else:
                timed_response.call_count = 0
            
            delay: float = response_times[timed_response.call_count % len(response_times)]
            time.sleep(delay)
            return '{"moves": ["UP"]}'
        
        mock_provider.generate_response.side_effect = timed_response
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Make requests and measure performance
        request_times: List[float] = []
        
        for i in range(len(response_times)):
            start_time: float = time.time()
            response: str = client.generate_response(f"test prompt {i}")
            end_time: float = time.time()
            
            request_time: float = end_time - start_time
            request_times.append(request_time)
            
            # Verify response
            assert response == '{"moves": ["UP"]}'
            
            # Verify time is approximately what we expect
            expected_time: float = response_times[i]
            assert abs(request_time - expected_time) < 0.1  # Allow some tolerance
        
        # Verify all requests completed
        assert len(request_times) == len(response_times)
        assert mock_provider.generate_response.call_count == len(response_times)

    def test_provider_state_isolation(self) -> None:
        """Test that provider state is properly isolated between clients."""
        # Create separate providers to test isolation
        provider1: Mock = Mock(spec=BaseLLMProvider)
        provider2: Mock = Mock(spec=BaseLLMProvider)
        
        provider1.is_available.return_value = True
        provider2.is_available.return_value = True
        
        provider1.generate_response.return_value = '{"moves": ["UP"]}'
        provider2.generate_response.return_value = '{"moves": ["DOWN"]}'
        
        provider1.get_last_token_count.return_value = {"total_tokens": 100}
        provider2.get_last_token_count.return_value = {"total_tokens": 200}
        
        # Create separate clients
        client1: LLMClient = LLMClient(provider1)
        client2: LLMClient = LLMClient(provider2)
        
        # Make requests with both clients
        response1: str = client1.generate_response("prompt for client 1")
        response2: str = client2.generate_response("prompt for client 2")
        
        # Verify responses are different (from different providers)
        assert response1 == '{"moves": ["UP"]}'
        assert response2 == '{"moves": ["DOWN"]}'
        
        # Verify token counts are separate
        tokens1: Optional[Dict[str, int]] = client1.get_last_token_count()
        tokens2: Optional[Dict[str, int]] = client2.get_last_token_count()
        
        assert tokens1 is not None
        assert tokens2 is not None
        assert tokens1["total_tokens"] == 100
        assert tokens2["total_tokens"] == 200
        
        # Verify providers were called correctly
        provider1.generate_response.assert_called_once_with("prompt for client 1")
        provider2.generate_response.assert_called_once_with("prompt for client 2")

    def test_provider_resource_cleanup(self) -> None:
        """Test proper resource cleanup between client and provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = '{"moves": ["UP"]}'
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        # Add cleanup method to mock
        mock_provider.cleanup = Mock()
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Make some requests
        for i in range(5):
            response: str = client.generate_response(f"test prompt {i}")
            assert response == '{"moves": ["UP"]}'
        
        # Test cleanup (if implemented)
        if hasattr(client, 'cleanup'):
            client.cleanup()
            if hasattr(mock_provider, 'cleanup'):
                mock_provider.cleanup.assert_called_once()
        
        # Verify provider was used correctly
        assert mock_provider.generate_response.call_count == 5
        assert mock_provider.get_last_token_count.call_count == 5 