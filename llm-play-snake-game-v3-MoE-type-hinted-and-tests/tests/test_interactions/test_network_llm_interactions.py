"""
Tests for Network utilities ↔ LLM provider interactions.

Focuses on testing how Network utilities and LLM providers handle
connection management, retry logic, and network failure scenarios.
"""

import pytest
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable
from unittest.mock import Mock, patch, MagicMock
import requests

from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider
from utils.network_utils import check_network_connectivity, retry_with_backoff


class TestNetworkLLMInteractions:
    """Test interactions between Network utilities and LLM providers."""

    def test_connection_establishment_provider_coordination(self) -> None:
        """Test connection establishment coordination between network utils and providers."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Test connection scenarios
        connection_scenarios: List[Tuple[str, bool, Optional[Exception], int]] = [
            ("immediate_success", True, None, 1),
            ("retry_success", False, ConnectionError("Network unreachable"), 3),
            ("timeout_failure", False, TimeoutError("Connection timeout"), 5),
            ("dns_failure", False, requests.exceptions.DNSError("DNS resolution failed"), 2),
        ]
        
        for scenario_name, should_succeed, network_error, expected_attempts in connection_scenarios:
            # Reset mock for each scenario
            mock_provider.reset_mock()
            
            # Configure network connectivity check
            with patch('utils.network_utils.check_network_connectivity') as mock_connectivity:
                if should_succeed:
                    mock_connectivity.return_value = True
                    mock_provider.is_available.return_value = True
                    mock_provider.generate_response.return_value = '{"moves": ["UP"]}'
                else:
                    mock_connectivity.side_effect = [False] * (expected_attempts - 1) + [True]
                    mock_provider.is_available.side_effect = [False] * (expected_attempts - 1) + [True]
                    mock_provider.generate_response.side_effect = [network_error] * (expected_attempts - 1) + ['{"moves": ["UP"]}']
                
                client: LLMClient = LLMClient(mock_provider)
                
                # Test connection-aware request
                start_time: float = time.time()
                
                try:
                    # Simulate retry logic integration
                    response: Optional[str] = None
                    attempts: int = 0
                    
                    while attempts < expected_attempts:
                        attempts += 1
                        
                        try:
                            # Check network first
                            if not check_network_connectivity():
                                raise ConnectionError("Network not available")
                            
                            # Try provider
                            if not mock_provider.is_available():
                                raise ConnectionError("Provider not available")
                            
                            response = client.generate_response("test prompt")
                            break
                            
                        except (ConnectionError, TimeoutError, requests.exceptions.RequestException) as e:
                            if attempts >= expected_attempts:
                                raise
                            
                            # Exponential backoff
                            wait_time: float = 0.1 * (2 ** (attempts - 1))
                            time.sleep(min(wait_time, 1.0))
                    
                    if should_succeed:
                        assert response == '{"moves": ["UP"]}', f"Failed to get response in {scenario_name}"
                        assert attempts <= expected_attempts, f"Too many attempts in {scenario_name}"
                    
                except Exception as e:
                    if should_succeed:
                        assert False, f"Should have succeeded in {scenario_name}: {e}"
                    else:
                        # Verify correct error type
                        assert type(e) in [ConnectionError, TimeoutError, requests.exceptions.RequestException], \
                            f"Unexpected error type in {scenario_name}: {type(e)}"
                
                end_time: float = time.time()
                connection_time: float = end_time - start_time
                
                # Verify reasonable timing
                if should_succeed:
                    assert connection_time < 1.0, f"Connection too slow in {scenario_name}: {connection_time}s"
                else:
                    # Failed scenarios should timeout appropriately
                    expected_time = 0.1 * (2 ** expected_attempts - 1)  # Backoff sum
                    assert connection_time >= expected_time * 0.8, f"Didn't wait long enough in {scenario_name}"

    def test_concurrent_network_provider_requests(self) -> None:
        """Test concurrent network requests across multiple providers."""
        # Create multiple mock providers
        providers: List[Mock] = []
        clients: List[LLMClient] = []
        
        for i in range(3):
            provider = Mock(spec=BaseLLMProvider)
            provider.is_available.return_value = True
            provider.generate_response.return_value = f'{{"moves": ["{"UP" if i == 0 else "DOWN" if i == 1 else "LEFT"}"]}}'
            provider.get_last_token_count.return_value = {"total_tokens": 30 + i * 10}
            
            providers.append(provider)
            clients.append(LLMClient(provider))
        
        request_results: List[Dict[str, Any]] = []
        request_errors: List[Exception] = []
        
        def concurrent_network_request(client_id: int, request_count: int) -> None:
            """Perform concurrent network requests."""
            try:
                client = clients[client_id]
                
                for i in range(request_count):
                    # Simulate network delay
                    time.sleep(0.01)
                    
                    # Check network (would normally be done by network utils)
                    with patch('utils.network_utils.check_network_connectivity', return_value=True):
                        response: str = client.generate_response(f"client_{client_id}_request_{i}")
                        
                        # Verify response format
                        assert response.startswith('{"moves":'), f"Invalid response format from client {client_id}"
                        
                        request_results.append({
                            "client_id": client_id,
                            "request_id": i,
                            "response": response,
                            "timestamp": time.time()
                        })
            
            except Exception as e:
                request_errors.append(e)
        
        # Start concurrent requests
        threads: List[threading.Thread] = []
        requests_per_client: int = 10
        
        for client_id in range(3):
            thread = threading.Thread(
                target=concurrent_network_request,
                args=(client_id, requests_per_client)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(request_errors) == 0, f"Concurrent request errors: {request_errors}"
        assert len(request_results) == 3 * requests_per_client
        
        # Verify each client's requests
        for client_id in range(3):
            client_results = [r for r in request_results if r["client_id"] == client_id]
            assert len(client_results) == requests_per_client
            
            # Verify response consistency for each client
            expected_move = "UP" if client_id == 0 else "DOWN" if client_id == 1 else "LEFT"
            for result in client_results:
                assert expected_move in result["response"]

    def test_network_failure_recovery_patterns(self) -> None:
        """Test network failure recovery patterns with LLM providers."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Simulate network failure patterns
        failure_patterns: List[Tuple[str, List[Exception], str]] = [
            ("intermittent_failures", [
                ConnectionError("Network timeout"),
                ConnectionError("Network timeout"),
                None,  # Success
                ConnectionError("Network timeout"),
                None,  # Success
            ], "partial_recovery"),
            
            ("progressive_degradation", [
                None,  # Success
                TimeoutError("Slow response"),
                TimeoutError("Slower response"),
                ConnectionError("Connection lost"),
                ConnectionError("Connection lost"),
            ], "degradation"),
            
            ("complete_recovery", [
                ConnectionError("Network down"),
                ConnectionError("Network down"),
                ConnectionError("Network down"),
                None,  # Recovery
                None,  # Stable
            ], "full_recovery"),
        ]
        
        for pattern_name, failures, recovery_type in failure_patterns:
            mock_provider.reset_mock()
            
            # Configure failure sequence
            responses: List[Any] = []
            availability: List[bool] = []
            
            for failure in failures:
                if failure is None:
                    responses.append('{"moves": ["UP"]}')
                    availability.append(True)
                else:
                    responses.append(failure)
                    availability.append(False)
            
            mock_provider.generate_response.side_effect = responses
            mock_provider.is_available.side_effect = availability
            mock_provider.get_last_token_count.return_value = {"total_tokens": 40}
            
            client: LLMClient = LLMClient(mock_provider)
            
            # Test recovery behavior
            recovery_results: List[Dict[str, Any]] = []
            
            for attempt in range(len(failures)):
                attempt_start: float = time.time()
                
                try:
                    # Network-aware request with recovery
                    with patch('utils.network_utils.check_network_connectivity') as mock_connectivity:
                        mock_connectivity.return_value = availability[attempt]
                        
                        if not mock_connectivity():
                            raise ConnectionError("Network check failed")
                        
                        response: str = client.generate_response(f"recovery test {attempt}")
                        
                        recovery_results.append({
                            "attempt": attempt,
                            "success": True,
                            "response": response,
                            "time": time.time() - attempt_start
                        })
                
                except Exception as e:
                    recovery_results.append({
                        "attempt": attempt,
                        "success": False,
                        "error": str(e),
                        "time": time.time() - attempt_start
                    })
            
            # Verify recovery patterns
            successful_attempts = [r for r in recovery_results if r["success"]]
            failed_attempts = [r for r in recovery_results if not r["success"]]
            
            if recovery_type == "partial_recovery":
                assert len(successful_attempts) > 0, f"No successful attempts in {pattern_name}"
                assert len(failed_attempts) > 0, f"No failed attempts in {pattern_name}"
            
            elif recovery_type == "degradation":
                # Should start successful then degrade
                assert recovery_results[0]["success"], f"First attempt should succeed in {pattern_name}"
                assert not recovery_results[-1]["success"], f"Last attempt should fail in {pattern_name}"
            
            elif recovery_type == "full_recovery":
                # Should have initial failures then recovery
                final_attempts = recovery_results[-2:]
                assert all(r["success"] for r in final_attempts), f"Failed to recover in {pattern_name}"
            
            # Verify error handling is appropriate
            for result in failed_attempts:
                assert "error" in result
                assert any(keyword in result["error"].lower() for keyword in ["network", "connection", "timeout"])

    def test_rate_limiting_network_coordination(self) -> None:
        """Test rate limiting coordination between network utils and providers."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Simulate rate limiting responses
        rate_limit_responses: List[Any] = []
        for i in range(20):
            if i < 5:
                # Normal responses
                rate_limit_responses.append('{"moves": ["UP"]}')
            elif i < 15:
                # Rate limited
                rate_limit_responses.append(requests.exceptions.HTTPError("429 Too Many Requests"))
            else:
                # Recovery
                rate_limit_responses.append('{"moves": ["DOWN"]}')
        
        mock_provider.generate_response.side_effect = rate_limit_responses
        mock_provider.get_last_token_count.return_value = {"total_tokens": 35}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Test rate limiting behavior
        rate_limit_results: List[Dict[str, Any]] = []
        
        for request_id in range(20):
            request_start: float = time.time()
            
            try:
                # Network-aware request with rate limiting
                with patch('utils.network_utils.retry_with_backoff') as mock_retry:
                    # Configure retry behavior for rate limiting
                    def retry_func(func: Callable, max_attempts: int = 3, backoff_factor: float = 2.0) -> Any:
                        for attempt in range(max_attempts):
                            try:
                                return func()
                            except requests.exceptions.HTTPError as e:
                                if "429" in str(e) and attempt < max_attempts - 1:
                                    # Rate limit backoff
                                    wait_time = backoff_factor ** attempt * 0.1
                                    time.sleep(wait_time)
                                    continue
                                raise
                        return None
                    
                    mock_retry.side_effect = lambda func, **kwargs: retry_func(func, **kwargs)
                    
                    # Make request with retry logic
                    response: str = retry_with_backoff(
                        lambda: client.generate_response(f"rate limit test {request_id}"),
                        max_attempts=3,
                        backoff_factor=2.0
                    )
                    
                    rate_limit_results.append({
                        "request_id": request_id,
                        "success": True,
                        "response": response,
                        "time": time.time() - request_start
                    })
            
            except Exception as e:
                rate_limit_results.append({
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "time": time.time() - request_start
                })
        
        # Verify rate limiting behavior
        successful_requests = [r for r in rate_limit_results if r["success"]]
        failed_requests = [r for r in rate_limit_results if not r["success"]]
        
        # Should handle rate limiting gracefully
        initial_success = rate_limit_results[:5]
        rate_limited_period = rate_limit_results[5:15]
        recovery_period = rate_limit_results[15:]
        
        # Initial requests should succeed
        assert all(r["success"] for r in initial_success), "Initial requests should succeed"
        
        # Rate limited period should have failures or delays
        rate_limited_failures = [r for r in rate_limited_period if not r["success"]]
        rate_limited_delays = [r for r in rate_limited_period if r["success"] and r["time"] > 0.1]
        
        assert len(rate_limited_failures) > 0 or len(rate_limited_delays) > 0, \
            "Rate limiting should cause failures or delays"
        
        # Recovery period should succeed
        recovery_success = [r for r in recovery_period if r["success"]]
        assert len(recovery_success) > 0, "Should recover from rate limiting"

    def test_network_latency_provider_adaptation(self) -> None:
        """Test provider adaptation to network latency conditions."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_last_token_count.return_value = {"total_tokens": 45}
        
        # Simulate varying network latencies
        latency_scenarios: List[Tuple[str, float, str]] = [
            ("low_latency", 0.01, '{"moves": ["UP"]}'),
            ("medium_latency", 0.1, '{"moves": ["DOWN"]}'),
            ("high_latency", 0.5, '{"moves": ["LEFT"]}'),
            ("very_high_latency", 1.0, '{"moves": ["RIGHT"]}'),
        ]
        
        for scenario_name, latency, expected_response in latency_scenarios:
            # Configure provider with latency simulation
            def delayed_response(prompt: str) -> str:
                time.sleep(latency)
                return expected_response
            
            mock_provider.generate_response.side_effect = delayed_response
            
            client: LLMClient = LLMClient(mock_provider)
            
            # Test adaptation to latency
            adaptation_start: float = time.time()
            
            # Network-aware request with timeout adaptation
            with patch('utils.network_utils.check_network_connectivity', return_value=True):
                try:
                    # Adaptive timeout based on expected latency
                    timeout: float = max(latency * 2, 0.1)  # At least 100ms
                    
                    response: str = client.generate_response("latency test", timeout=timeout)
                    
                    adaptation_time: float = time.time() - adaptation_start
                    
                    # Verify response
                    assert response == expected_response, f"Wrong response in {scenario_name}"
                    
                    # Verify timing
                    assert adaptation_time >= latency, f"Response too fast in {scenario_name}"
                    assert adaptation_time < latency + 0.2, f"Response too slow in {scenario_name}"
                    
                    # High latency should trigger adaptive behavior
                    if latency > 0.3:
                        # Could implement provider switching, caching, etc.
                        print(f"High latency detected in {scenario_name}: {latency}s")
                
                except TimeoutError:
                    if latency > 1.0:
                        # Very high latency might timeout, which is acceptable
                        print(f"Timeout in {scenario_name} with latency {latency}s")
                    else:
                        assert False, f"Unexpected timeout in {scenario_name}"

    def test_network_provider_failover_mechanism(self) -> None:
        """Test failover mechanism between network issues and provider availability."""
        # Create primary and backup providers
        primary_provider: Mock = Mock(spec=BaseLLMProvider)
        backup_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure failover scenarios
        failover_scenarios: List[Tuple[str, bool, bool, str]] = [
            ("both_available", True, True, "primary"),
            ("primary_network_fail", False, True, "backup"),
            ("primary_provider_fail", True, False, "backup"),
            ("both_fail", False, False, "error"),
        ]
        
        for scenario_name, primary_network, primary_available, expected_source in failover_scenarios:
            primary_provider.reset_mock()
            backup_provider.reset_mock()
            
            # Configure primary provider
            primary_provider.is_available.return_value = primary_available
            if primary_available:
                primary_provider.generate_response.return_value = '{"moves": ["PRIMARY"]}'
            else:
                primary_provider.generate_response.side_effect = ConnectionError("Primary unavailable")
            
            # Configure backup provider
            backup_provider.is_available.return_value = True
            backup_provider.generate_response.return_value = '{"moves": ["BACKUP"]}'
            
            primary_client: LLMClient = LLMClient(primary_provider)
            backup_client: LLMClient = LLMClient(backup_provider)
            
            # Test failover logic
            with patch('utils.network_utils.check_network_connectivity') as mock_connectivity:
                mock_connectivity.return_value = primary_network
                
                try:
                    response: Optional[str] = None
                    
                    # Try primary first
                    if primary_network and primary_provider.is_available():
                        try:
                            response = primary_client.generate_response("failover test")
                            actual_source = "primary"
                        except Exception:
                            # Fall back to backup
                            response = backup_client.generate_response("failover test")
                            actual_source = "backup"
                    else:
                        # Use backup directly
                        response = backup_client.generate_response("failover test")
                        actual_source = "backup"
                    
                    if expected_source == "error":
                        assert False, f"Should have failed in {scenario_name}"
                    
                    # Verify correct source was used
                    if expected_source == "primary":
                        assert "PRIMARY" in response, f"Should use primary in {scenario_name}"
                        assert actual_source == "primary"
                    elif expected_source == "backup":
                        assert "BACKUP" in response, f"Should use backup in {scenario_name}"
                        assert actual_source == "backup"
                
                except Exception as e:
                    if expected_source != "error":
                        assert False, f"Unexpected failure in {scenario_name}: {e}"
                    
                    # Verify appropriate error handling
                    assert "network" in str(e).lower() or "connection" in str(e).lower()

    def test_network_monitoring_provider_health_correlation(self) -> None:
        """Test correlation between network monitoring and provider health."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Simulate correlated network and provider health
        health_scenarios: List[Tuple[float, bool, bool, str]] = [
            (0.0, True, True, "healthy"),      # Good network, healthy provider
            (0.1, True, True, "good"),         # Slight delay, still good
            (0.3, True, False, "degraded"),    # Moderate delay, provider issues
            (0.8, False, False, "poor"),       # High delay, provider failing
            (2.0, False, False, "critical"),   # Very high delay, provider down
        ]
        
        health_results: List[Dict[str, Any]] = []
        
        for network_delay, network_healthy, provider_healthy, expected_status in health_scenarios:
            mock_provider.reset_mock()
            
            # Configure provider health
            mock_provider.is_available.return_value = provider_healthy
            
            if provider_healthy:
                def delayed_healthy_response(prompt: str) -> str:
                    time.sleep(network_delay)
                    return '{"moves": ["UP"]}'
                mock_provider.generate_response.side_effect = delayed_healthy_response
            else:
                def delayed_unhealthy_response(prompt: str) -> str:
                    time.sleep(network_delay)
                    raise ConnectionError("Provider unhealthy")
                mock_provider.generate_response.side_effect = delayed_unhealthy_response
            
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            
            # Monitor health correlation
            health_start: float = time.time()
            
            with patch('utils.network_utils.check_network_connectivity', return_value=network_healthy):
                try:
                    # Health check request
                    response: str = client.generate_response("health check")
                    
                    health_time: float = time.time() - health_start
                    
                    health_results.append({
                        "scenario": expected_status,
                        "network_delay": network_delay,
                        "network_healthy": network_healthy,
                        "provider_healthy": provider_healthy,
                        "response_time": health_time,
                        "success": True,
                        "response": response
                    })
                
                except Exception as e:
                    health_time = time.time() - health_start
                    
                    health_results.append({
                        "scenario": expected_status,
                        "network_delay": network_delay,
                        "network_healthy": network_healthy,
                        "provider_healthy": provider_healthy,
                        "response_time": health_time,
                        "success": False,
                        "error": str(e)
                    })
        
        # Verify health correlation
        for result in health_results:
            scenario = result["scenario"]
            
            if scenario == "healthy":
                assert result["success"], f"Healthy scenario should succeed"
                assert result["response_time"] < 0.1, f"Healthy scenario should be fast"
            
            elif scenario == "good":
                assert result["success"], f"Good scenario should succeed"
                assert result["response_time"] < 0.3, f"Good scenario should be reasonably fast"
            
            elif scenario in ["degraded", "poor", "critical"]:
                # May succeed or fail, but should reflect the degraded state
                if result["success"]:
                    assert result["response_time"] >= result["network_delay"], \
                        f"Response time should reflect network delay in {scenario}"
                else:
                    assert "connection" in result["error"].lower() or "network" in result["error"].lower(), \
                        f"Error should indicate network/connection issue in {scenario}" 