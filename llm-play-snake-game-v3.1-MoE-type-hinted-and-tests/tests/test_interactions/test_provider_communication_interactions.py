"""
Tests for Provider ↔ Communication utilities interactions.

Focuses on testing how multiple LLM providers interact with communication
utilities for fallback handling, rate limiting, and response coordination.
"""

import pytest
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock
import threading
from concurrent.futures import ThreadPoolExecutor

from llm.providers.base_provider import BaseLLMProvider
from llm.communication_utils import CommunicationUtils


class TestProviderCommunicationInteractions:
    """Test interactions between LLM providers and communication utilities."""

    def test_multi_provider_fallback_coordination(self) -> None:
        """Test fallback coordination between multiple providers."""
        
        # Mock multiple providers with different capabilities
        primary_provider: Mock = Mock(spec=BaseLLMProvider)
        secondary_provider: Mock = Mock(spec=BaseLLMProvider)
        tertiary_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure provider availability and failure patterns
        primary_provider.is_available.return_value = True
        secondary_provider.is_available.return_value = True
        tertiary_provider.is_available.return_value = True
        
        # Primary provider fails occasionally
        primary_call_count: int = 0
        def primary_response(prompt: str) -> str:
            nonlocal primary_call_count
            primary_call_count += 1
            
            if primary_call_count in [3, 7, 12]:  # Fail on specific calls
                raise ConnectionError(f"Primary provider failed on call {primary_call_count}")
            return f"Primary response {primary_call_count}: {prompt[:50]}"
        
        primary_provider.generate_response.side_effect = primary_response
        primary_provider.get_last_token_count.return_value = {"total_tokens": 150}
        
        # Secondary provider slower but reliable
        def secondary_response(prompt: str) -> str:
            time.sleep(0.1)  # Simulate slower response
            return f"Secondary response: {prompt[:30]}"
        
        secondary_provider.generate_response.side_effect = secondary_response
        secondary_provider.get_last_token_count.return_value = {"total_tokens": 120}
        
        # Tertiary provider fastest but lower quality
        def tertiary_response(prompt: str) -> str:
            return f"Tertiary: {prompt[:20]}"
        
        tertiary_provider.generate_response.side_effect = tertiary_response
        tertiary_provider.get_last_token_count.return_value = {"total_tokens": 80}
        
        # Mock communication utils with fallback logic
        comm_utils: Mock = Mock(spec=CommunicationUtils)
        
        fallback_results: List[Dict[str, Any]] = []
        
        def coordinate_fallback(prompt: str, providers: List[BaseLLMProvider]) -> Dict[str, Any]:
            """Simulate fallback coordination logic."""
            for i, provider in enumerate(providers):
                try:
                    start_time = time.time()
                    response = provider.generate_response(prompt)
                    end_time = time.time()
                    
                    token_count = provider.get_last_token_count()
                    
                    return {
                        "provider_index": i,
                        "provider_type": ["primary", "secondary", "tertiary"][i],
                        "response": response,
                        "token_count": token_count,
                        "response_time": end_time - start_time,
                        "success": True
                    }
                    
                except Exception as e:
                    fallback_results.append({
                        "provider_index": i,
                        "provider_type": ["primary", "secondary", "tertiary"][i],
                        "error": str(e),
                        "attempt_time": time.time()
                    })
                    continue
            
            # All providers failed
            return {
                "success": False,
                "error": "All providers failed",
                "fallback_attempts": len(providers)
            }
        
        comm_utils.coordinate_fallback = coordinate_fallback
        
        # Test fallback coordination
        providers = [primary_provider, secondary_provider, tertiary_provider]
        test_prompts = [
            "Generate move for snake game",
            "Analyze current game state",
            "Suggest strategy for high score",
            "Navigate around obstacles",
            "Optimize path to apple",
            "Handle wall collision avoidance",
            "Plan multi-step sequence",
            "React to dynamic game state",
            "Evaluate risk vs reward",
            "Coordinate team strategy",
            "Process complex game rules",
            "Generate creative solution",
            "Adapt to changing conditions"
        ]
        
        coordination_results: List[Dict[str, Any]] = []
        
        for i, prompt in enumerate(test_prompts):
            result = comm_utils.coordinate_fallback(prompt, providers)
            result["prompt_index"] = i
            result["prompt"] = prompt
            coordination_results.append(result)
        
        # Verify fallback coordination
        successful_results = [r for r in coordination_results if r.get("success", False)]
        assert len(successful_results) > 10, "Most requests should succeed with fallback"
        
        # Verify provider usage distribution
        provider_usage = {}
        for result in successful_results:
            provider_type = result["provider_type"]
            provider_usage[provider_type] = provider_usage.get(provider_type, 0) + 1
        
        # Primary should be used most when available
        assert provider_usage.get("primary", 0) > 0, "Primary provider should be used"
        
        # Fallbacks should be used when primary fails
        fallback_usage = provider_usage.get("secondary", 0) + provider_usage.get("tertiary", 0)
        assert fallback_usage > 0, "Fallback providers should be used"
        
        # Verify fallback attempts recorded
        assert len(fallback_results) > 0, "Should record fallback attempts"
        
        # Verify error handling
        failed_attempts = [f for f in fallback_results if "Primary provider failed" in f["error"]]
        assert len(failed_attempts) == 3, "Should record primary provider failures"

    def test_rate_limiting_coordination(self) -> None:
        """Test rate limiting coordination across providers."""
        
        # Mock providers with different rate limits
        fast_provider: Mock = Mock(spec=BaseLLMProvider)
        medium_provider: Mock = Mock(spec=BaseLLMProvider)
        slow_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure rate limits (requests per second)
        rate_limits = {
            "fast": {"limit": 10, "window": 1.0, "calls": []},
            "medium": {"limit": 5, "window": 1.0, "calls": []},
            "slow": {"limit": 2, "window": 1.0, "calls": []}
        }
        
        def create_rate_limited_provider(provider: Mock, provider_name: str) -> None:
            def rate_limited_response(prompt: str) -> str:
                current_time = time.time()
                rate_info = rate_limits[provider_name]
                
                # Clean old calls outside window
                rate_info["calls"] = [
                    call_time for call_time in rate_info["calls"]
                    if current_time - call_time < rate_info["window"]
                ]
                
                # Check rate limit
                if len(rate_info["calls"]) >= rate_info["limit"]:
                    raise Exception(f"{provider_name} provider rate limit exceeded")
                
                # Record call
                rate_info["calls"].append(current_time)
                
                # Simulate processing time
                time.sleep(0.01)
                return f"{provider_name} response: {prompt[:30]}"
            
            provider.generate_response.side_effect = rate_limited_response
            provider.get_last_token_count.return_value = {"total_tokens": 100}
            provider.is_available.return_value = True
        
        create_rate_limited_provider(fast_provider, "fast")
        create_rate_limited_provider(medium_provider, "medium")
        create_rate_limited_provider(slow_provider, "slow")
        
        # Mock communication utils with rate limiting awareness
        comm_utils: Mock = Mock(spec=CommunicationUtils)
        
        provider_stats = {
            "fast": {"successful": 0, "rate_limited": 0},
            "medium": {"successful": 0, "rate_limited": 0},
            "slow": {"successful": 0, "rate_limited": 0}
        }
        
        def coordinate_with_rate_limiting(
            prompts: List[str], 
            providers: Dict[str, BaseLLMProvider]
        ) -> List[Dict[str, Any]]:
            """Coordinate requests with rate limiting awareness."""
            results = []
            
            for i, prompt in enumerate(prompts):
                # Try providers in order of preference (fast -> medium -> slow)
                for provider_name in ["fast", "medium", "slow"]:
                    provider = providers[provider_name]
                    
                    try:
                        start_time = time.time()
                        response = provider.generate_response(prompt)
                        end_time = time.time()
                        
                        provider_stats[provider_name]["successful"] += 1
                        
                        results.append({
                            "prompt_index": i,
                            "provider_used": provider_name,
                            "response": response,
                            "response_time": end_time - start_time,
                            "success": True
                        })
                        break
                        
                    except Exception as e:
                        provider_stats[provider_name]["rate_limited"] += 1
                        
                        if "rate limit" in str(e):
                            # Try next provider
                            continue
                        else:
                            # Non-rate-limit error
                            results.append({
                                "prompt_index": i,
                                "provider_used": provider_name,
                                "error": str(e),
                                "success": False
                            })
                            break
                else:
                    # All providers rate limited
                    results.append({
                        "prompt_index": i,
                        "error": "All providers rate limited",
                        "success": False
                    })
            
            return results
        
        comm_utils.coordinate_with_rate_limiting = coordinate_with_rate_limiting
        
        # Test rate limiting coordination
        providers = {
            "fast": fast_provider,
            "medium": medium_provider,
            "slow": slow_provider
        }
        
        # Generate high-frequency requests
        test_prompts = [f"Request {i}: Generate snake move" for i in range(25)]
        
        # Execute requests rapidly
        start_time = time.time()
        results = comm_utils.coordinate_with_rate_limiting(test_prompts, providers)
        end_time = time.time()
        
        # Verify rate limiting coordination
        successful_results = [r for r in results if r.get("success", False)]
        assert len(successful_results) > 15, "Should handle many requests with rate limiting"
        
        # Verify provider distribution respects rate limits
        provider_usage = {}
        for result in successful_results:
            provider = result["provider_used"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        # Fast provider should handle most requests
        assert provider_usage.get("fast", 0) >= provider_usage.get("medium", 0), \
            "Fast provider should be preferred"
        
        # Rate limiting should cause fallbacks
        total_rate_limited = sum(stats["rate_limited"] for stats in provider_stats.values())
        assert total_rate_limited > 0, "Should encounter rate limiting"
        
        # Verify timing respects rate limits
        total_duration = end_time - start_time
        assert total_duration > 1.0, "Should take time due to rate limiting"

    def test_concurrent_provider_communication(self) -> None:
        """Test concurrent communication across multiple providers."""
        
        # Mock providers for concurrent testing
        providers = {}
        provider_locks = {}
        
        for i in range(4):
            provider = Mock(spec=BaseLLMProvider)
            provider.is_available.return_value = True
            
            # Each provider has different characteristics
            processing_time = 0.05 + (i * 0.02)  # 0.05s to 0.11s
            
            def create_concurrent_response(provider_id: int, delay: float):
                def concurrent_response(prompt: str) -> str:
                    time.sleep(delay)  # Simulate processing
                    return f"Provider {provider_id}: {prompt[:25]}"
                return concurrent_response
            
            provider.generate_response.side_effect = create_concurrent_response(i, processing_time)
            provider.get_last_token_count.return_value = {"total_tokens": 90 + (i * 10)}
            
            providers[f"provider_{i}"] = provider
            provider_locks[f"provider_{i}"] = threading.Lock()
        
        # Mock communication utils for concurrent coordination
        comm_utils: Mock = Mock(spec=CommunicationUtils)
        
        concurrent_results: List[Dict[str, Any]] = []
        result_lock = threading.Lock()
        
        def concurrent_request_handler(
            request_id: int, 
            prompt: str, 
            provider_name: str
        ) -> None:
            """Handle concurrent request to specific provider."""
            try:
                provider = providers[provider_name]
                provider_lock = provider_locks[provider_name]
                
                start_time = time.time()
                
                # Simulate concurrent access control
                with provider_lock:
                    response = provider.generate_response(prompt)
                    token_count = provider.get_last_token_count()
                
                end_time = time.time()
                
                with result_lock:
                    concurrent_results.append({
                        "request_id": request_id,
                        "provider": provider_name,
                        "prompt": prompt,
                        "response": response,
                        "token_count": token_count,
                        "duration": end_time - start_time,
                        "success": True
                    })
                    
            except Exception as e:
                with result_lock:
                    concurrent_results.append({
                        "request_id": request_id,
                        "provider": provider_name,
                        "error": str(e),
                        "success": False
                    })
        
        def coordinate_concurrent_requests(requests: List[Tuple[int, str, str]]) -> None:
            """Coordinate multiple concurrent requests."""
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                for request_id, prompt, provider_name in requests:
                    future = executor.submit(
                        concurrent_request_handler,
                        request_id,
                        prompt,
                        provider_name
                    )
                    futures.append(future)
                
                # Wait for all requests to complete
                for future in futures:
                    future.result(timeout=5.0)
        
        comm_utils.coordinate_concurrent_requests = coordinate_concurrent_requests
        
        # Test concurrent communication
        requests = []
        
        # Create concurrent requests across providers
        for i in range(20):
            provider_name = f"provider_{i % 4}"  # Distribute across providers
            prompt = f"Concurrent request {i}: What's the best move?"
            requests.append((i, prompt, provider_name))
        
        # Execute concurrent requests
        start_time = time.time()
        comm_utils.coordinate_concurrent_requests(requests)
        end_time = time.time()
        
        # Verify concurrent communication
        assert len(concurrent_results) == 20, "Should complete all concurrent requests"
        
        successful_results = [r for r in concurrent_results if r.get("success", False)]
        assert len(successful_results) == 20, "All concurrent requests should succeed"
        
        # Verify concurrency benefits
        total_duration = end_time - start_time
        sequential_duration = sum(0.05 + ((i % 4) * 0.02) for i in range(20))  # Expected sequential time
        
        assert total_duration < sequential_duration * 0.6, \
            f"Concurrent execution should be faster: {total_duration}s vs {sequential_duration}s"
        
        # Verify provider distribution
        provider_usage = {}
        for result in successful_results:
            provider = result["provider"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        # Each provider should handle 5 requests
        for i in range(4):
            provider_name = f"provider_{i}"
            assert provider_usage[provider_name] == 5, f"Provider {i} should handle 5 requests"
        
        # Verify response characteristics
        for result in successful_results:
            assert "duration" in result, "Should record response duration"
            assert result["duration"] > 0, "Should have positive duration"
            assert "token_count" in result, "Should record token count"

    def test_error_propagation_across_communication_layers(self) -> None:
        """Test error propagation across communication layers."""
        
        # Mock providers with different error patterns
        error_provider: Mock = Mock(spec=BaseLLMProvider)
        timeout_provider: Mock = Mock(spec=BaseLLMProvider)
        normal_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure error patterns
        error_call_count = 0
        def error_response(prompt: str) -> str:
            nonlocal error_call_count
            error_call_count += 1
            
            if error_call_count <= 3:
                raise ValueError(f"Provider error {error_call_count}")
            return f"Error provider recovered: {prompt[:20]}"
        
        timeout_call_count = 0
        def timeout_response(prompt: str) -> str:
            nonlocal timeout_call_count
            timeout_call_count += 1
            
            if timeout_call_count % 3 == 0:
                raise TimeoutError(f"Provider timeout {timeout_call_count}")
            time.sleep(0.1)  # Simulate processing
            return f"Timeout provider: {prompt[:20]}"
        
        def normal_response(prompt: str) -> str:
            return f"Normal provider: {prompt[:20]}"
        
        error_provider.generate_response.side_effect = error_response
        timeout_provider.generate_response.side_effect = timeout_response
        normal_provider.generate_response.side_effect = normal_response
        
        for provider in [error_provider, timeout_provider, normal_provider]:
            provider.is_available.return_value = True
            provider.get_last_token_count.return_value = {"total_tokens": 85}
        
        # Mock communication utils with error propagation handling
        comm_utils: Mock = Mock(spec=CommunicationUtils)
        
        error_propagation_log: List[Dict[str, Any]] = []
        
        def handle_error_propagation(
            prompt: str,
            providers: List[Tuple[str, BaseLLMProvider]]
        ) -> Dict[str, Any]:
            """Handle error propagation across communication layers."""
            error_chain = []
            
            for provider_name, provider in providers:
                try:
                    response = provider.generate_response(prompt)
                    token_count = provider.get_last_token_count()
                    
                    # Success - record error chain if any
                    result = {
                        "success": True,
                        "provider_used": provider_name,
                        "response": response,
                        "token_count": token_count,
                        "error_chain": error_chain.copy()
                    }
                    
                    if error_chain:
                        error_propagation_log.append({
                            "prompt": prompt,
                            "error_chain": error_chain,
                            "resolution": provider_name,
                            "recovery_successful": True
                        })
                    
                    return result
                    
                except Exception as e:
                    error_info = {
                        "provider": provider_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": time.time()
                    }
                    error_chain.append(error_info)
                    continue
            
            # All providers failed
            error_propagation_log.append({
                "prompt": prompt,
                "error_chain": error_chain,
                "resolution": None,
                "recovery_successful": False
            })
            
            return {
                "success": False,
                "error_chain": error_chain,
                "total_failures": len(error_chain)
            }
        
        comm_utils.handle_error_propagation = handle_error_propagation
        
        # Test error propagation
        test_scenarios = [
            ("Simple request", [("error", error_provider), ("normal", normal_provider)]),
            ("Timeout scenario", [("timeout", timeout_provider), ("normal", normal_provider)]),
            ("Multi-error", [("error", error_provider), ("timeout", timeout_provider), ("normal", normal_provider)]),
            ("All fail first", [("error", error_provider), ("timeout", timeout_provider), ("normal", normal_provider)]),
            ("Recovery test", [("error", error_provider), ("normal", normal_provider)]),
        ]
        
        propagation_results: List[Dict[str, Any]] = []
        
        for scenario_name, providers in test_scenarios:
            for i in range(3):  # Multiple attempts per scenario
                prompt = f"{scenario_name} attempt {i}: Generate move"
                result = comm_utils.handle_error_propagation(prompt, providers)
                result["scenario"] = scenario_name
                result["attempt"] = i
                propagation_results.append(result)
        
        # Verify error propagation handling
        successful_results = [r for r in propagation_results if r.get("success", False)]
        assert len(successful_results) > 10, "Should successfully handle most scenarios"
        
        # Verify error chain recording
        error_chain_results = [r for r in successful_results if len(r.get("error_chain", [])) > 0]
        assert len(error_chain_results) > 0, "Should record error chains when recovery occurs"
        
        # Verify error propagation log
        assert len(error_propagation_log) > 0, "Should log error propagation events"
        
        successful_recoveries = [log for log in error_propagation_log if log["recovery_successful"]]
        assert len(successful_recoveries) > 0, "Should have successful error recoveries"
        
        # Verify error types are captured
        all_errors = []
        for result in propagation_results:
            if "error_chain" in result:
                all_errors.extend(result["error_chain"])
        
        error_types = set(error["error_type"] for error in all_errors)
        assert "ValueError" in error_types, "Should capture ValueError"
        assert "TimeoutError" in error_types, "Should capture TimeoutError"
        
        # Verify normal provider serves as fallback
        normal_provider_usage = len([r for r in successful_results if r.get("provider_used") == "normal"])
        assert normal_provider_usage > 0, "Normal provider should be used as fallback"

    def test_response_quality_coordination(self) -> None:
        """Test coordination based on response quality across providers."""
        
        # Mock providers with different quality characteristics
        high_quality_provider: Mock = Mock(spec=BaseLLMProvider)
        medium_quality_provider: Mock = Mock(spec=BaseLLMProvider)
        low_quality_provider: Mock = Mock(spec=BaseLLMProvider)
        
        # Configure quality characteristics
        def high_quality_response(prompt: str) -> str:
            return json.dumps({
                "moves": ["UP", "RIGHT"],
                "reasoning": "High quality analysis of game state with strategic planning",
                "confidence": 0.95,
                "alternatives": ["LEFT", "DOWN"],
                "quality_score": 0.9
            })
        
        def medium_quality_response(prompt: str) -> str:
            return json.dumps({
                "moves": ["UP"],
                "reasoning": "Basic move analysis",
                "confidence": 0.7,
                "quality_score": 0.6
            })
        
        def low_quality_response(prompt: str) -> str:
            return '{"moves": ["UP"], "quality_score": 0.3}'
        
        high_quality_provider.generate_response.side_effect = high_quality_response
        medium_quality_provider.generate_response.side_effect = medium_quality_response
        low_quality_provider.generate_response.side_effect = low_quality_response
        
        # Different token costs
        high_quality_provider.get_last_token_count.return_value = {"total_tokens": 200}
        medium_quality_provider.get_last_token_count.return_value = {"total_tokens": 120}
        low_quality_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        for provider in [high_quality_provider, medium_quality_provider, low_quality_provider]:
            provider.is_available.return_value = True
        
        # Mock communication utils with quality-based coordination
        comm_utils: Mock = Mock(spec=CommunicationUtils)
        
        quality_decisions: List[Dict[str, Any]] = []
        
        def coordinate_by_quality(
            prompt: str,
            providers: Dict[str, BaseLLMProvider],
            quality_requirements: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Coordinate provider selection based on quality requirements."""
            
            min_quality = quality_requirements.get("min_quality", 0.5)
            max_tokens = quality_requirements.get("max_tokens", 300)
            prefer_speed = quality_requirements.get("prefer_speed", False)
            
            # Try providers in quality order unless speed preferred
            provider_order = ["high_quality", "medium_quality", "low_quality"]
            if prefer_speed:
                provider_order = ["low_quality", "medium_quality", "high_quality"]
            
            for provider_name in provider_order:
                provider = providers[provider_name]
                
                try:
                    start_time = time.time()
                    response = provider.generate_response(prompt)
                    end_time = time.time()
                    
                    token_count = provider.get_last_token_count()
                    
                    # Check token limit
                    if token_count["total_tokens"] > max_tokens:
                        continue
                    
                    # Parse quality score
                    try:
                        import json
                        parsed = json.loads(response)
                        quality_score = parsed.get("quality_score", 0.5)
                    except:
                        quality_score = 0.3
                    
                    # Check quality requirement
                    if quality_score < min_quality:
                        continue
                    
                    decision = {
                        "provider_selected": provider_name,
                        "quality_score": quality_score,
                        "token_count": token_count["total_tokens"],
                        "response_time": end_time - start_time,
                        "response": response,
                        "success": True,
                        "requirements_met": True
                    }
                    
                    quality_decisions.append(decision)
                    return decision
                    
                except Exception as e:
                    continue
            
            # No provider met requirements
            decision = {
                "success": False,
                "error": "No provider met quality requirements",
                "requirements": quality_requirements
            }
            quality_decisions.append(decision)
            return decision
        
        comm_utils.coordinate_by_quality = coordinate_by_quality
        
        # Test quality-based coordination
        providers = {
            "high_quality": high_quality_provider,
            "medium_quality": medium_quality_provider,
            "low_quality": low_quality_provider
        }
        
        # Various quality requirement scenarios
        quality_scenarios = [
            {"name": "high_quality_required", "min_quality": 0.8, "max_tokens": 300, "prefer_speed": False},
            {"name": "medium_quality_ok", "min_quality": 0.5, "max_tokens": 150, "prefer_speed": False},
            {"name": "speed_prioritized", "min_quality": 0.2, "max_tokens": 100, "prefer_speed": True},
            {"name": "token_constrained", "min_quality": 0.4, "max_tokens": 80, "prefer_speed": False},
            {"name": "balanced", "min_quality": 0.6, "max_tokens": 200, "prefer_speed": False},
        ]
        
        coordination_results: List[Dict[str, Any]] = []
        
        for scenario in quality_scenarios:
            for i in range(3):  # Multiple tests per scenario
                prompt = f"Quality test {scenario['name']} attempt {i}: Generate optimal move"
                requirements = {k: v for k, v in scenario.items() if k != "name"}
                
                result = comm_utils.coordinate_by_quality(prompt, providers, requirements)
                result["scenario"] = scenario["name"]
                result["attempt"] = i
                coordination_results.append(result)
        
        # Verify quality coordination
        successful_results = [r for r in coordination_results if r.get("success", False)]
        assert len(successful_results) > 10, "Should successfully coordinate based on quality"
        
        # Verify quality requirements respected
        for result in successful_results:
            scenario_name = result["scenario"]
            scenario = next(s for s in quality_scenarios if s["name"] == scenario_name)
            
            # Quality threshold met
            assert result["quality_score"] >= scenario["min_quality"], \
                f"Quality requirement not met for {scenario_name}"
            
            # Token limit respected
            assert result["token_count"] <= scenario["max_tokens"], \
                f"Token limit exceeded for {scenario_name}"
        
        # Verify provider selection logic
        high_quality_selections = [r for r in successful_results if r["provider_selected"] == "high_quality"]
        speed_prioritized_results = [r for r in successful_results if r["scenario"] == "speed_prioritized"]
        
        # High quality provider should be selected for high quality requirements
        high_quality_required_results = [r for r in successful_results if r["scenario"] == "high_quality_required"]
        if high_quality_required_results:
            assert all(r["provider_selected"] == "high_quality" for r in high_quality_required_results), \
                "High quality provider should be selected for high quality requirements"
        
        # Speed prioritized should prefer faster providers
        if speed_prioritized_results:
            speed_providers = [r["provider_selected"] for r in speed_prioritized_results]
            assert "low_quality" in speed_providers, "Speed prioritization should use faster providers" 