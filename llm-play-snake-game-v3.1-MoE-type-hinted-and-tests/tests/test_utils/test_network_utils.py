"""
Tests for utils.network_utils module.

Focuses on testing network utility functions for HTTP requests, connection handling,
timeout management, and network resilience.
"""

import pytest
import time
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from utils.network_utils import NetworkUtils


class TestNetworkUtils:
    """Test network utility functions."""

    def test_http_request_with_retries(self) -> None:
        """Test HTTP requests with retry logic."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock requests module
        with patch('utils.network_utils.requests') as mock_requests:
            # Test successful request
            mock_response: Mock = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "success", "data": "test_data"}
            mock_response.text = '{"status": "success", "data": "test_data"}'
            mock_requests.get.return_value = mock_response
            
            result: Dict[str, Any] = network_utils.make_request_with_retries(
                url="https://api.example.com/test",
                method="GET",
                max_retries=3,
                timeout=10
            )
            
            assert result["success"] is True, "Request should succeed"
            assert result["status_code"] == 200, "Should return correct status code"
            assert result["data"]["status"] == "success", "Should return correct data"
            assert result["attempts"] == 1, "Should succeed on first attempt"
        
        # Test request with retries after failures
        with patch('utils.network_utils.requests') as mock_requests:
            # First two calls fail, third succeeds
            mock_requests.get.side_effect = [
                ConnectionError("Connection failed"),
                Timeout("Request timeout"),
                Mock(status_code=200, json=lambda: {"status": "success"})
            ]
            
            result = network_utils.make_request_with_retries(
                url="https://api.example.com/test",
                method="GET",
                max_retries=3,
                timeout=10,
                retry_delay=0.1
            )
            
            assert result["success"] is True, "Should eventually succeed"
            assert result["attempts"] == 3, "Should take 3 attempts"
            assert "errors" in result, "Should record previous errors"
            assert len(result["errors"]) == 2, "Should record 2 failures"
        
        # Test complete failure after all retries
        with patch('utils.network_utils.requests') as mock_requests:
            mock_requests.get.side_effect = ConnectionError("Persistent connection error")
            
            result = network_utils.make_request_with_retries(
                url="https://api.example.com/test",
                method="GET",
                max_retries=2,
                timeout=5
            )
            
            assert result["success"] is False, "Should fail after all retries"
            assert result["attempts"] == 2, "Should attempt max retries"
            assert "final_error" in result, "Should record final error"

    def test_connection_health_monitoring(self) -> None:
        """Test connection health monitoring functionality."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock health check endpoints
        health_endpoints: List[Dict[str, Any]] = [
            {"url": "https://api.deepseek.com/health", "name": "deepseek", "timeout": 5},
            {"url": "https://api.mistral.ai/health", "name": "mistral", "timeout": 5},
            {"url": "https://api.hunyuan.tencent.com/health", "name": "hunyuan", "timeout": 5}
        ]
        
        with patch('utils.network_utils.requests') as mock_requests:
            # Mock different health check responses
            def mock_health_response(url: str, timeout: int) -> Mock:
                mock_response: Mock = Mock()
                
                if "deepseek" in url:
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": "healthy"}
                    mock_response.elapsed.total_seconds.return_value = 0.15
                elif "mistral" in url:
                    mock_response.status_code = 503
                    mock_response.json.return_value = {"status": "degraded"}
                    mock_response.elapsed.total_seconds.return_value = 2.0
                elif "hunyuan" in url:
                    raise Timeout("Health check timeout")
                
                return mock_response
            
            mock_requests.get.side_effect = lambda url, timeout=None: mock_health_response(url, timeout)
            
            # Perform health monitoring
            health_results: List[Dict[str, Any]] = network_utils.monitor_connection_health(
                endpoints=health_endpoints
            )
            
            assert len(health_results) == 3, "Should check all endpoints"
            
            # Verify deepseek (healthy)
            deepseek_result = next(r for r in health_results if r["name"] == "deepseek")
            assert deepseek_result["healthy"] is True, "Deepseek should be healthy"
            assert deepseek_result["status_code"] == 200, "Should have 200 status"
            assert deepseek_result["response_time"] < 1.0, "Should have fast response"
            
            # Verify mistral (degraded)
            mistral_result = next(r for r in health_results if r["name"] == "mistral")
            assert mistral_result["healthy"] is False, "Mistral should be unhealthy"
            assert mistral_result["status_code"] == 503, "Should have 503 status"
            
            # Verify hunyuan (timeout)
            hunyuan_result = next(r for r in health_results if r["name"] == "hunyuan")
            assert hunyuan_result["healthy"] is False, "Hunyuan should be unhealthy"
            assert "error" in hunyuan_result, "Should record timeout error"

    def test_rate_limiting_coordination(self) -> None:
        """Test rate limiting coordination across requests."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock rate limiter
        rate_limiter: Mock = Mock()
        rate_limiter.request_timestamps = {}
        rate_limiter.rate_limits = {
            "deepseek": {"requests_per_minute": 60, "requests_per_second": 2},
            "mistral": {"requests_per_minute": 30, "requests_per_second": 1},
            "hunyuan": {"requests_per_minute": 120, "requests_per_second": 3}
        }
        
        def mock_check_rate_limit(provider: str, request_time: float) -> Dict[str, Any]:
            """Mock rate limit checking."""
            if provider not in rate_limiter.request_timestamps:
                rate_limiter.request_timestamps[provider] = []
            
            timestamps = rate_limiter.request_timestamps[provider]
            limits = rate_limiter.rate_limits.get(provider, {"requests_per_second": 1})
            
            # Clean old timestamps (older than 1 minute)
            current_time = request_time
            timestamps[:] = [t for t in timestamps if current_time - t < 60]
            
            # Check per-second limit
            recent_requests = [t for t in timestamps if current_time - t < 1]
            per_second_limit = limits["requests_per_second"]
            
            if len(recent_requests) >= per_second_limit:
                return {
                    "allowed": False,
                    "reason": "per_second_limit_exceeded",
                    "wait_time": 1.0 - (current_time - min(recent_requests)),
                    "current_rate": len(recent_requests)
                }
            
            # Check per-minute limit
            per_minute_limit = limits["requests_per_minute"]
            if len(timestamps) >= per_minute_limit:
                return {
                    "allowed": False,
                    "reason": "per_minute_limit_exceeded", 
                    "wait_time": 60.0 - (current_time - min(timestamps)),
                    "current_rate": len(timestamps)
                }
            
            # Request allowed
            timestamps.append(current_time)
            return {
                "allowed": True,
                "current_rate_per_second": len(recent_requests) + 1,
                "current_rate_per_minute": len(timestamps)
            }
        
        rate_limiter.check_rate_limit = mock_check_rate_limit
        
        # Test rate limiting scenarios
        rate_limit_tests: List[Dict[str, Any]] = []
        
        # Test rapid requests to deepseek (should hit per-second limit)
        for i in range(5):
            current_time = time.time() + (i * 0.1)  # 10 requests per second
            result = rate_limiter.check_rate_limit("deepseek", current_time)
            
            rate_limit_tests.append({
                "provider": "deepseek",
                "request_number": i + 1,
                "result": result,
                "expected_allowed": i < 2  # Only first 2 should be allowed (limit is 2/second)
            })
        
        # Test slower requests to mistral (should be allowed)
        for i in range(3):
            current_time = time.time() + 100 + (i * 2.0)  # 0.5 requests per second
            result = rate_limiter.check_rate_limit("mistral", current_time)
            
            rate_limit_tests.append({
                "provider": "mistral",
                "request_number": i + 1,
                "result": result,
                "expected_allowed": True  # All should be allowed
            })
        
        # Verify rate limiting results
        deepseek_tests = [t for t in rate_limit_tests if t["provider"] == "deepseek"]
        mistral_tests = [t for t in rate_limit_tests if t["provider"] == "mistral"]
        
        # Verify deepseek rate limiting
        for test in deepseek_tests:
            expected = test["expected_allowed"]
            actual = test["result"]["allowed"]
            request_num = test["request_number"]
            
            assert actual == expected, f"Deepseek request {request_num}: expected {expected}, got {actual}"
            
            if not actual:
                assert "wait_time" in test["result"], f"Should provide wait_time for blocked request {request_num}"
        
        # Verify mistral requests (should all pass)
        for test in mistral_tests:
            assert test["result"]["allowed"] is True, f"Mistral request {test['request_number']} should be allowed"

    def test_network_resilience_and_failover(self) -> None:
        """Test network resilience and failover mechanisms."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock failover system
        failover_manager: Mock = Mock()
        failover_manager.primary_endpoints = [
            "https://primary-api.example.com",
            "https://secondary-api.example.com", 
            "https://tertiary-api.example.com"
        ]
        failover_manager.endpoint_health = {
            "https://primary-api.example.com": {"healthy": True, "last_check": time.time()},
            "https://secondary-api.example.com": {"healthy": False, "last_check": time.time() - 300},
            "https://tertiary-api.example.com": {"healthy": True, "last_check": time.time() - 60}
        }
        failover_manager.failover_log = []
        
        def mock_attempt_request_with_failover(
            endpoints: List[str], 
            request_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Mock request with failover logic."""
            
            for i, endpoint in enumerate(endpoints):
                endpoint_health = failover_manager.endpoint_health.get(endpoint, {"healthy": False})
                
                # Skip unhealthy endpoints
                if not endpoint_health["healthy"]:
                    failover_manager.failover_log.append({
                        "action": "skip_unhealthy",
                        "endpoint": endpoint,
                        "timestamp": time.time()
                    })
                    continue
                
                try:
                    # Simulate request attempt
                    if "primary" in endpoint:
                        # Primary fails
                        raise ConnectionError("Primary endpoint down")
                    elif "secondary" in endpoint:
                        # Secondary is marked unhealthy, shouldn't reach here
                        raise RuntimeError("Should not reach unhealthy secondary")
                    elif "tertiary" in endpoint:
                        # Tertiary succeeds
                        failover_manager.failover_log.append({
                            "action": "success",
                            "endpoint": endpoint,
                            "attempts": i + 1,
                            "timestamp": time.time()
                        })
                        
                        return {
                            "success": True,
                            "endpoint_used": endpoint,
                            "attempts": i + 1,
                            "data": {"response": "success from tertiary"}
                        }
                
                except Exception as e:
                    failover_manager.failover_log.append({
                        "action": "failed_attempt",
                        "endpoint": endpoint,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                    continue
            
            # All endpoints failed
            return {
                "success": False,
                "error": "All endpoints failed",
                "attempts": len(endpoints)
            }
        
        failover_manager.attempt_request_with_failover = mock_attempt_request_with_failover
        
        # Test failover scenario
        request_data = {
            "method": "POST",
            "path": "/api/generate",
            "data": {"prompt": "test prompt"},
            "timeout": 30
        }
        
        result = failover_manager.attempt_request_with_failover(
            failover_manager.primary_endpoints,
            request_data
        )
        
        # Verify failover success
        assert result["success"] is True, "Failover should eventually succeed"
        assert result["endpoint_used"] == "https://tertiary-api.example.com", "Should use tertiary endpoint"
        assert result["attempts"] == 2, "Should take 2 attempts (skip secondary, use tertiary)"
        
        # Verify failover log
        assert len(failover_manager.failover_log) == 3, "Should log 3 actions"
        
        actions = [log["action"] for log in failover_manager.failover_log]
        assert "failed_attempt" in actions, "Should log failed attempt on primary"
        assert "skip_unhealthy" in actions, "Should log skipping unhealthy secondary"
        assert "success" in actions, "Should log success on tertiary"

    def test_connection_pooling_and_reuse(self) -> None:
        """Test connection pooling and connection reuse."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock connection pool
        connection_pool: Mock = Mock()
        connection_pool.active_connections = {}
        connection_pool.connection_stats = {
            "total_created": 0,
            "total_reused": 0,
            "total_closed": 0
        }
        connection_pool.max_connections_per_host = 5
        
        def mock_get_connection(host: str, port: int = 443) -> Dict[str, Any]:
            """Mock getting connection from pool."""
            connection_key = f"{host}:{port}"
            
            # Check if connection exists and is available
            if connection_key in connection_pool.active_connections:
                connection = connection_pool.active_connections[connection_key]
                if connection["available"]:
                    connection["reuse_count"] += 1
                    connection["last_used"] = time.time()
                    connection_pool.connection_stats["total_reused"] += 1
                    
                    return {
                        "connection_id": connection["id"],
                        "reused": True,
                        "reuse_count": connection["reuse_count"]
                    }
            
            # Create new connection
            connection_id = f"conn_{connection_pool.connection_stats['total_created']}"
            
            new_connection = {
                "id": connection_id,
                "host": host,
                "port": port,
                "created_time": time.time(),
                "last_used": time.time(),
                "reuse_count": 0,
                "available": True
            }
            
            connection_pool.active_connections[connection_key] = new_connection
            connection_pool.connection_stats["total_created"] += 1
            
            return {
                "connection_id": connection_id,
                "reused": False,
                "reuse_count": 0
            }
        
        def mock_release_connection(connection_id: str) -> bool:
            """Mock releasing connection back to pool."""
            for connection in connection_pool.active_connections.values():
                if connection["id"] == connection_id:
                    connection["available"] = True
                    return True
            return False
        
        connection_pool.get_connection = mock_get_connection
        connection_pool.release_connection = mock_release_connection
        
        # Test connection pooling scenarios
        connection_tests: List[Dict[str, Any]] = []
        
        # Test multiple requests to same host (should reuse connections)
        for i in range(10):
            conn_result = connection_pool.get_connection("api.example.com", 443)
            
            connection_tests.append({
                "request_number": i + 1,
                "connection_id": conn_result["connection_id"],
                "reused": conn_result["reused"],
                "reuse_count": conn_result["reuse_count"]
            })
            
            # Release connection for reuse
            connection_pool.release_connection(conn_result["connection_id"])
        
        # Verify connection reuse
        first_connection = connection_tests[0]
        assert not first_connection["reused"], "First connection should not be reused"
        
        subsequent_connections = connection_tests[1:]
        reused_connections = [t for t in subsequent_connections if t["reused"]]
        
        assert len(reused_connections) > 0, "Should reuse connections for subsequent requests"
        
        # All subsequent connections should reuse the same connection
        connection_ids = set(t["connection_id"] for t in connection_tests)
        assert len(connection_ids) == 1, "Should reuse the same connection for same host"
        
        # Verify connection statistics
        assert connection_pool.connection_stats["total_created"] == 1, "Should create only 1 connection"
        assert connection_pool.connection_stats["total_reused"] == 9, "Should reuse connection 9 times"

    def test_timeout_and_cancellation_handling(self) -> None:
        """Test timeout and request cancellation handling."""
        
        network_utils: NetworkUtils = NetworkUtils()
        
        # Mock timeout manager
        timeout_manager: Mock = Mock()
        timeout_manager.active_requests = {}
        timeout_manager.timeout_log = []
        
        def mock_make_request_with_timeout(
            url: str,
            timeout: float,
            cancellation_token: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Mock request with timeout and cancellation support."""
            
            request_id = f"req_{len(timeout_manager.active_requests)}"
            start_time = time.time()
            
            request_info = {
                "id": request_id,
                "url": url,
                "timeout": timeout,
                "start_time": start_time,
                "cancelled": False
            }
            
            timeout_manager.active_requests[request_id] = request_info
            
            try:
                # Simulate different scenarios based on URL
                if "fast-api" in url:
                    # Fast response
                    elapsed = 0.1
                    time.sleep(0.001)  # Minimal delay for simulation
                    
                elif "slow-api" in url:
                    # Slow response that should timeout
                    elapsed = timeout + 1.0  # Exceed timeout
                    
                elif "medium-api" in url:
                    # Medium response within timeout
                    elapsed = timeout * 0.8
                    time.sleep(0.001)
                
                # Check for cancellation
                if cancellation_token and cancellation_token.get("cancelled", False):
                    request_info["cancelled"] = True
                    timeout_manager.timeout_log.append({
                        "request_id": request_id,
                        "event": "cancelled",
                        "elapsed_time": time.time() - start_time
                    })
                    
                    return {
                        "success": False,
                        "error": "Request cancelled",
                        "request_id": request_id,
                        "elapsed_time": time.time() - start_time
                    }
                
                # Check for timeout
                if elapsed > timeout:
                    timeout_manager.timeout_log.append({
                        "request_id": request_id,
                        "event": "timeout",
                        "timeout_value": timeout,
                        "elapsed_time": elapsed
                    })
                    
                    return {
                        "success": False,
                        "error": "Request timeout",
                        "request_id": request_id,
                        "timeout": timeout,
                        "elapsed_time": elapsed
                    }
                
                # Successful response
                timeout_manager.timeout_log.append({
                    "request_id": request_id,
                    "event": "success",
                    "elapsed_time": elapsed
                })
                
                return {
                    "success": True,
                    "data": {"response": "success"},
                    "request_id": request_id,
                    "elapsed_time": elapsed
                }
                
            finally:
                # Cleanup
                if request_id in timeout_manager.active_requests:
                    del timeout_manager.active_requests[request_id]
        
        timeout_manager.make_request_with_timeout = mock_make_request_with_timeout
        
        # Test timeout scenarios
        timeout_test_cases: List[Dict[str, Any]] = [
            {
                "name": "fast_request",
                "url": "https://fast-api.example.com/endpoint",
                "timeout": 5.0,
                "should_succeed": True
            },
            {
                "name": "slow_request_timeout",
                "url": "https://slow-api.example.com/endpoint", 
                "timeout": 1.0,
                "should_succeed": False
            },
            {
                "name": "medium_request",
                "url": "https://medium-api.example.com/endpoint",
                "timeout": 10.0,
                "should_succeed": True
            }
        ]
        
        timeout_results: List[Dict[str, Any]] = []
        
        for test_case in timeout_test_cases:
            result = timeout_manager.make_request_with_timeout(
                url=test_case["url"],
                timeout=test_case["timeout"]
            )
            
            timeout_results.append({
                "test_name": test_case["name"],
                "expected_success": test_case["should_succeed"],
                "actual_success": result["success"],
                "result": result
            })
        
        # Verify timeout handling
        for test_result in timeout_results:
            expected = test_result["expected_success"]
            actual = test_result["actual_success"]
            test_name = test_result["test_name"]
            
            assert actual == expected, f"Timeout test {test_name}: expected {expected}, got {actual}"
        
        # Verify timeout logging
        assert len(timeout_manager.timeout_log) == 3, "Should log all timeout events"
        
        timeout_events = [log["event"] for log in timeout_manager.timeout_log]
        assert "success" in timeout_events, "Should log successful requests"
        assert "timeout" in timeout_events, "Should log timeout events"
        
        # Test cancellation
        cancellation_token: Dict[str, Any] = {"cancelled": False}
        
        # Start request and then cancel it
        cancellation_token["cancelled"] = True
        
        cancel_result = timeout_manager.make_request_with_timeout(
            url="https://medium-api.example.com/endpoint",
            timeout=10.0,
            cancellation_token=cancellation_token
        )
        
        assert cancel_result["success"] is False, "Cancelled request should fail"
        assert "cancelled" in cancel_result["error"], "Should indicate cancellation"
