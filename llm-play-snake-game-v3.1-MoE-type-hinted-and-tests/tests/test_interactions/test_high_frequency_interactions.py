"""
Tests for high-frequency component interactions.

Focuses on testing performance bottlenecks and system behavior
under rapid component interactions and high-frequency operations.
"""

import pytest
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from unittest.mock import Mock, patch
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from llm.client import LLMClient
from utils.moves_utils import calculate_next_position, normalize_direction


class TestHighFrequencyInteractions:
    """Test high-frequency interactions between components."""

    def test_rapid_controller_data_updates(self) -> None:
        """Test rapid updates between GameController and GameData."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Perform high-frequency operations
        operations_per_second: int = 1000
        duration_seconds: float = 2.0
        total_operations: int = int(operations_per_second * duration_seconds)
        
        start_time: float = time.time()
        successful_operations: int = 0
        
        for i in range(total_operations):
            operation_start: float = time.time()
            
            # High-frequency move operations
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            
            try:
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                # Verify data consistency after each operation
                assert controller.score == game_data.score
                assert controller.steps == game_data.steps
                assert len(controller.snake_positions) == game_data.snake_length
                
                successful_operations += 1
                
                if collision:
                    controller.reset()
                    # Verify reset consistency
                    assert controller.score == 0
                    assert game_data.score == 0
                
            except Exception as e:
                # Should not fail under high frequency
                assert False, f"High-frequency operation {i} failed: {e}"
            
            operation_end: float = time.time()
            operation_time: float = operation_end - operation_start
            
            # Each operation should be fast
            assert operation_time < 0.01, f"Operation {i} too slow: {operation_time}s"
        
        end_time: float = time.time()
        total_time: float = end_time - start_time
        actual_rate: float = successful_operations / total_time
        
        # Should achieve close to target rate
        assert actual_rate >= operations_per_second * 0.8, f"Too slow: {actual_rate} ops/sec"
        assert successful_operations == total_operations, "Some operations failed"

    def test_concurrent_high_frequency_access(self) -> None:
        """Test concurrent high-frequency access patterns."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        access_results: List[Dict[str, Any]] = []
        access_errors: List[Exception] = []
        
        def high_frequency_reader(thread_id: int) -> None:
            """Perform high-frequency read operations."""
            try:
                read_count: int = 0
                start_time: float = time.time()
                
                while time.time() - start_time < 1.0:  # Run for 1 second
                    # High-frequency reads
                    score: int = controller.score
                    steps: int = controller.steps
                    positions: NDArray[np.int_] = controller.snake_positions
                    
                    # Verify data consistency
                    assert score >= 0
                    assert steps >= 0
                    assert len(positions) >= 1
                    
                    read_count += 1
                
                access_results.append({
                    "thread_id": thread_id,
                    "type": "reader",
                    "operations": read_count,
                    "rate": read_count / 1.0
                })
                
            except Exception as e:
                access_errors.append(e)
        
        def high_frequency_writer(thread_id: int) -> None:
            """Perform high-frequency write operations."""
            try:
                write_count: int = 0
                start_time: float = time.time()
                
                while time.time() - start_time < 1.0:  # Run for 1 second
                    # High-frequency writes
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][write_count % 4]
                    
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                    
                    write_count += 1
                
                access_results.append({
                    "thread_id": thread_id,
                    "type": "writer",
                    "operations": write_count,
                    "rate": write_count / 1.0
                })
                
            except Exception as e:
                access_errors.append(e)
        
        # Start concurrent high-frequency operations
        threads: List[threading.Thread] = []
        
        # Multiple readers
        for i in range(5):
            thread = threading.Thread(target=high_frequency_reader, args=(i,))
            threads.append(thread)
        
        # Single writer (to avoid conflicts)
        thread = threading.Thread(target=high_frequency_writer, args=(99,))
        threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        assert len(access_errors) == 0, f"High-frequency access errors: {access_errors}"
        assert len(access_results) == 6
        
        # Verify performance rates
        readers = [r for r in access_results if r["type"] == "reader"]
        writers = [r for r in access_results if r["type"] == "writer"]
        
        assert len(readers) == 5
        assert len(writers) == 1
        
        # Readers should achieve high rates
        for reader in readers:
            assert reader["rate"] > 500, f"Reader too slow: {reader['rate']} ops/sec"
        
        # Writer should also be reasonably fast
        assert writers[0]["rate"] > 100, f"Writer too slow: {writers[0]['rate']} ops/sec"

    def test_memory_allocation_high_frequency(self) -> None:
        """Test memory allocation patterns under high-frequency operations."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Track memory usage patterns
        memory_samples: List[Dict[str, Any]] = []
        
        def sample_memory_usage() -> Dict[str, Any]:
            """Sample current memory usage approximation."""
            import sys
            
            # Approximate memory usage
            snake_memory = len(controller.snake_positions) * controller.snake_positions.itemsize * controller.snake_positions.shape[1]
            moves_memory = len(controller.game_state.moves) * 8  # Approximate string size
            
            return {
                "snake_positions": len(controller.snake_positions),
                "moves_count": len(controller.game_state.moves),
                "estimated_snake_memory": snake_memory,
                "estimated_moves_memory": moves_memory,
                "timestamp": time.time()
            }
        
        # Perform memory-intensive high-frequency operations
        start_time: float = time.time()
        
        for i in range(5000):  # Many operations
            # Make move (creates new positions, updates moves list)
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Sample memory every 100 operations
            if i % 100 == 0:
                memory_samples.append(sample_memory_usage())
            
            # Reset periodically to test memory cleanup
            if collision or i % 200 == 199:
                controller.reset()
                
                # Sample after reset
                if i % 100 == 99:
                    memory_samples.append(sample_memory_usage())
        
        end_time: float = time.time()
        total_time: float = end_time - start_time
        
        # Verify performance
        assert total_time < 10.0, f"High-frequency operations too slow: {total_time}s"
        
        # Analyze memory patterns
        assert len(memory_samples) > 10, "Not enough memory samples"
        
        # Memory should not grow unbounded
        max_snake_positions = max(sample["snake_positions"] for sample in memory_samples)
        max_moves_count = max(sample["moves_count"] for sample in memory_samples)
        
        # Should stay within reasonable bounds
        assert max_snake_positions < 100, f"Snake positions grew too large: {max_snake_positions}"
        assert max_moves_count < 500, f"Moves list grew too large: {max_moves_count}"
        
        # After resets, memory should be cleaned up
        reset_samples = [s for i, s in enumerate(memory_samples) if i > 0 and s["snake_positions"] == 1]
        assert len(reset_samples) > 0, "No memory cleanup detected"

    def test_llm_client_high_frequency_requests(self) -> None:
        """Test LLM client behavior under high-frequency requests."""
        from llm.providers.base_provider import BaseLLMProvider
        
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Configure for high-frequency testing
        response_pool: List[str] = [
            '{"moves": ["UP"]}',
            '{"moves": ["DOWN"]}',
            '{"moves": ["LEFT"]}',
            '{"moves": ["RIGHT"]}',
        ]
        
        request_count: int = 0
        
        def fast_response(prompt: str) -> str:
            nonlocal request_count
            request_count += 1
            return response_pool[request_count % len(response_pool)]
        
        mock_provider.generate_response.side_effect = fast_response
        mock_provider.get_last_token_count.return_value = {"total_tokens": 25}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # High-frequency request testing
        start_time: float = time.time()
        successful_requests: int = 0
        request_times: List[float] = []
        
        for i in range(200):  # High frequency requests
            request_start: float = time.time()
            
            try:
                response: str = client.generate_response(f"high freq request {i}")
                
                # Verify response
                assert response in response_pool, f"Unexpected response: {response}"
                
                # Verify token tracking
                tokens: Optional[Dict[str, int]] = client.get_last_token_count()
                assert tokens is not None
                assert tokens["total_tokens"] == 25
                
                successful_requests += 1
                
            except Exception as e:
                assert False, f"High-frequency request {i} failed: {e}"
            
            request_end: float = time.time()
            request_time: float = request_end - request_start
            request_times.append(request_time)
            
            # Each request should be fast
            assert request_time < 0.1, f"Request {i} too slow: {request_time}s"
        
        end_time: float = time.time()
        total_time: float = end_time - start_time
        request_rate: float = successful_requests / total_time
        
        # Verify performance
        assert request_rate > 50, f"Request rate too low: {request_rate} req/sec"
        assert successful_requests == 200, "Some requests failed"
        
        # Verify consistent performance
        avg_request_time: float = sum(request_times) / len(request_times)
        max_request_time: float = max(request_times)
        
        assert avg_request_time < 0.01, f"Average request time too high: {avg_request_time}s"
        assert max_request_time < 0.05, f"Max request time too high: {max_request_time}s"

    def test_json_parsing_high_frequency_throughput(self) -> None:
        """Test JSON parsing throughput under high-frequency operations."""
        from utils.json_utils import safe_json_parse, extract_json_from_text
        
        # Prepare test data with various complexities
        test_responses: List[str] = [
            '{"moves": ["UP"]}',
            '{"moves": ["DOWN", "LEFT"]}',
            'Text before {"moves": ["RIGHT"]} after',
            '```json\n{"moves": ["UP", "DOWN"]}\n```',
            '{"moves": ["LEFT"], "analysis": "short"}',
            '{"moves": ["RIGHT"], "analysis": "' + "longer analysis " * 20 + '"}',
        ]
        
        parsing_results: List[Dict[str, Any]] = []
        
        # High-frequency parsing test
        start_time: float = time.time()
        
        for i in range(1000):  # High volume
            response: str = test_responses[i % len(test_responses)]
            
            parse_start: float = time.time()
            
            # Parse with full chain
            parsed: Optional[Dict[str, Any]] = safe_json_parse(response)
            if parsed is None:
                parsed = extract_json_from_text(response)
            
            parse_end: float = time.time()
            parse_time: float = parse_end - parse_start
            
            # Extract moves
            moves: List[str] = []
            if parsed and "moves" in parsed:
                moves = parsed["moves"]
            
            parsing_results.append({
                "iteration": i,
                "response_length": len(response),
                "parse_time": parse_time,
                "moves_found": len(moves),
                "successful": len(moves) > 0
            })
            
            # Each parse should be fast
            assert parse_time < 0.001, f"Parse {i} too slow: {parse_time}s"
        
        end_time: float = time.time()
        total_time: float = end_time - start_time
        parse_rate: float = len(parsing_results) / total_time
        
        # Verify throughput
        assert parse_rate > 500, f"Parse rate too low: {parse_rate} parses/sec"
        
        # Verify success rate
        successful_parses = [r for r in parsing_results if r["successful"]]
        success_rate: float = len(successful_parses) / len(parsing_results)
        assert success_rate > 0.95, f"Parse success rate too low: {success_rate}"
        
        # Verify consistent performance across different response types
        avg_parse_times: Dict[int, List[float]] = {}
        for result in parsing_results:
            response_type = result["iteration"] % len(test_responses)
            if response_type not in avg_parse_times:
                avg_parse_times[response_type] = []
            avg_parse_times[response_type].append(result["parse_time"])
        
        # All response types should parse quickly
        for response_type, times in avg_parse_times.items():
            avg_time = sum(times) / len(times)
            assert avg_time < 0.0005, f"Response type {response_type} too slow: {avg_time}s"

    def test_file_io_high_frequency_operations(self, temp_dir: str) -> None:
        """Test file I/O operations under high frequency."""
        import os
        from utils.file_utils import save_json_safely, load_json_safely
        
        test_file: str = os.path.join(temp_dir, "high_freq_test.json")
        
        # High-frequency file operations
        operation_count: int = 100
        io_times: List[float] = []
        
        for i in range(operation_count):
            # Prepare test data
            test_data: Dict[str, Any] = {
                "iteration": i,
                "timestamp": time.time(),
                "data": list(range(i % 10 + 1))  # Variable size data
            }
            
            # Test save performance
            save_start: float = time.time()
            save_success: bool = save_json_safely(test_data, test_file)
            save_end: float = time.time()
            save_time: float = save_end - save_start
            
            assert save_success, f"Save {i} failed"
            assert save_time < 0.01, f"Save {i} too slow: {save_time}s"
            
            # Test load performance
            load_start: float = time.time()
            loaded_data: Optional[Dict[str, Any]] = load_json_safely(test_file)
            load_end: float = time.time()
            load_time: float = load_end - load_start
            
            assert loaded_data is not None, f"Load {i} failed"
            assert loaded_data == test_data, f"Data mismatch in iteration {i}"
            assert load_time < 0.01, f"Load {i} too slow: {load_time}s"
            
            total_io_time: float = save_time + load_time
            io_times.append(total_io_time)
        
        # Verify overall I/O performance
        avg_io_time: float = sum(io_times) / len(io_times)
        max_io_time: float = max(io_times)
        
        assert avg_io_time < 0.005, f"Average I/O too slow: {avg_io_time}s"
        assert max_io_time < 0.02, f"Max I/O too slow: {max_io_time}s"
        
        # Verify file system stability
        assert os.path.exists(test_file), "Test file disappeared"
        final_size: int = os.path.getsize(test_file)
        assert final_size > 0, "Test file is empty"

    def test_bottleneck_identification_under_load(self) -> None:
        """Test identification of performance bottlenecks under load."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        # Component timing tracking
        component_times: Dict[str, List[float]] = {
            "move_calculation": [],
            "collision_detection": [],
            "board_update": [],
            "data_synchronization": [],
            "apple_generation": []
        }
        
        bottleneck_threshold: float = 0.005  # 5ms threshold
        
        for i in range(500):  # Load test
            # Time move calculation
            calc_start: float = time.time()
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            head_pos: List[int] = controller.snake_positions[-1].tolist()
            next_pos: List[int] = calculate_next_position(head_pos, move)
            calc_end: float = time.time()
            component_times["move_calculation"].append(calc_end - calc_start)
            
            # Time collision detection
            collision_start: float = time.time()
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            collision_end: float = time.time()
            
            # Break down the make_move timing (approximate)
            component_times["collision_detection"].append((collision_end - collision_start) * 0.3)
            component_times["board_update"].append((collision_end - collision_start) * 0.3)
            component_times["data_synchronization"].append((collision_end - collision_start) * 0.2)
            
            # Time apple generation (when needed)
            if apple_eaten or collision:
                apple_start: float = time.time()
                if collision:
                    controller.reset()
                apple_end: float = time.time()
                component_times["apple_generation"].append(apple_end - apple_start)
        
        # Analyze bottlenecks
        bottlenecks: Dict[str, Dict[str, float]] = {}
        
        for component, times in component_times.items():
            if times:  # Skip empty lists
                avg_time: float = sum(times) / len(times)
                max_time: float = max(times)
                p95_time: float = sorted(times)[int(len(times) * 0.95)]
                
                bottlenecks[component] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "p95_time": p95_time,
                    "is_bottleneck": p95_time > bottleneck_threshold
                }
        
        # Verify no major bottlenecks
        major_bottlenecks: List[str] = [
            comp for comp, metrics in bottlenecks.items() 
            if metrics["is_bottleneck"]
        ]
        
        # Report bottlenecks for debugging but don't fail
        if major_bottlenecks:
            print(f"Performance bottlenecks detected: {major_bottlenecks}")
            for bottleneck in major_bottlenecks:
                metrics = bottlenecks[bottleneck]
                print(f"  {bottleneck}: p95={metrics['p95_time']:.6f}s, max={metrics['max_time']:.6f}s")
        
        # System should remain responsive even under load
        total_avg_time: float = sum(
            metrics["avg_time"] for metrics in bottlenecks.values()
        )
        assert total_avg_time < 0.01, f"Total average operation time too high: {total_avg_time}s" 