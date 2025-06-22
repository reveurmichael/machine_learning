"""
Stress and performance tests for the SnakeGTP system.

This module tests system behavior under heavy load, memory pressure,
and performance-critical scenarios.
"""

import pytest
import numpy as np
import time
import threading
import gc
import psutil
import os
from typing import List, Dict, Any, Optional, Tuple, Callable
from unittest.mock import Mock, patch
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from llm.client import LLMClient
from utils.moves_utils import normalize_direction, calculate_next_position
from utils.json_utils import safe_json_parse, extract_json_from_text


class TestPerformanceStress:
    """Performance and stress testing for all components."""

    def test_high_frequency_moves_performance(self) -> None:
        """Test performance with high-frequency move generation."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Test rapid move execution
        start_time: float = time.time()
        move_count: int = 0
        max_duration: float = 5.0  # 5 seconds max
        
        while time.time() - start_time < max_duration:
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][move_count % 4]
            collision: bool
            _: bool
            collision, _ = controller.make_move(move)
            move_count += 1
            
            if collision:
                controller.reset()
        
        total_time: float = time.time() - start_time
        moves_per_second: float = move_count / total_time
        
        # Performance targets
        assert moves_per_second > 1000  # Should handle 1000+ moves/second
        assert move_count > 5000  # Should execute many moves in 5 seconds
        
        print(f"Performance: {moves_per_second:.1f} moves/second, {move_count} total moves")

    def test_large_snake_collision_detection_performance(self) -> None:
        """Test collision detection performance with very large snakes."""
        grid_size: int = 50
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        
        # Create increasingly large snakes and measure performance
        snake_sizes: List[int] = [10, 50, 100, 500, 1000, 2000]
        performance_results: List[Tuple[int, float]] = []
        
        for snake_size in snake_sizes:
            if snake_size > grid_size * grid_size:
                continue
                
            # Create large snake
            snake_positions: List[List[int]] = []
            for i in range(min(snake_size, grid_size * grid_size - 1)):
                x: int = i % grid_size
                y: int = i // grid_size
                snake_positions.append([x, y])
            
            controller.snake_positions = np.array(snake_positions, dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            # Measure collision detection performance
            start_time = time.time()
            iterations: int = 1000
            
            for _ in range(iterations):
                collision: bool
                _: bool
                collision, _ = controller.make_move("UP")
                if not collision:
                    controller.reset()
                    controller.snake_positions = np.array(snake_positions, dtype=np.int_)
                    controller.head_position = controller.snake_positions[-1]
                    controller._update_board()
            
            elapsed_time: float = time.time() - start_time
            avg_time_per_check: float = elapsed_time / iterations
            performance_results.append((snake_size, avg_time_per_check))
            
            # Performance should scale reasonably
            assert avg_time_per_check < 0.01  # Less than 10ms per collision check
        
        # Verify performance doesn't degrade dramatically with size
        if len(performance_results) > 1:
            first_time: float = performance_results[0][1]
            last_time: float = performance_results[-1][1]
            # Performance shouldn't degrade more than 10x
            assert last_time / first_time < 10

    def test_memory_usage_with_large_datasets(self) -> None:
        """Test memory usage with large game datasets."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory: float = process.memory_info().rss / 1024 / 1024  # MB
        
        controllers: List[GameController] = []
        large_game_states: List[GameData] = []
        
        try:
            # Create many large game instances
            for i in range(100):
                controller: GameController = GameController(grid_size=30, use_gui=False)
                game_data: GameData = GameData()
                
                # Populate with substantial data
                for j in range(1000):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][j % 4]
                    controller.make_move(move)
                    
                    if j % 10 == 0:
                        game_data.add_llm_communication(f"prompt_{j}", f"response_{j}")
                        game_data.add_token_usage(prompt_tokens=100, completion_tokens=50)
                
                controllers.append(controller)
                large_game_states.append(game_data)
                
                # Check memory every 10 instances
                if i % 10 == 9:
                    current_memory: float = process.memory_info().rss / 1024 / 1024
                    memory_growth: float = current_memory - initial_memory
                    
                    # Memory growth should be reasonable (less than 1GB total)
                    if memory_growth > 1024:  # 1GB
                        break
        
        except MemoryError:
            # Expected if we run out of memory
            pass
        
        # Cleanup and verify memory can be reclaimed
        del controllers
        del large_game_states
        gc.collect()
        
        final_memory: float = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: Initial {initial_memory:.1f}MB, Final {final_memory:.1f}MB")

    def test_concurrent_game_instances(self) -> None:
        """Test running multiple game instances concurrently."""
        num_threads: int = 10
        moves_per_thread: int = 1000
        results: List[Dict[str, Any]] = [None] * num_threads  # type: ignore
        errors: List[Exception] = []
        
        def run_game_thread(thread_id: int) -> None:
            try:
                controller: GameController = GameController(grid_size=15, use_gui=False)
                moves_made: int = 0
                collisions: int = 0
                
                for i in range(moves_per_thread):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    _: bool
                    collision, _ = controller.make_move(move)
                    moves_made += 1
                    
                    if collision:
                        collisions += 1
                        controller.reset()
                
                results[thread_id] = {
                    "thread_id": thread_id,
                    "moves_made": moves_made,
                    "collisions": collisions,
                    "final_score": controller.score
                }
                
            except Exception as e:
                errors.append(e)
        
        # Start all threads
        start_time: float = time.time()
        threads: List[threading.Thread] = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=run_game_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30.0)  # 30 second timeout
        
        total_time: float = time.time() - start_time
        
        # Verify results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(result is not None for result in results)
        
        total_moves: int = sum(result["moves_made"] for result in results if result)
        moves_per_second: float = total_moves / total_time
        
        print(f"Concurrent performance: {moves_per_second:.1f} total moves/second across {num_threads} threads")
        assert moves_per_second > 5000  # Should handle high concurrent load

    def test_json_parsing_performance_stress(self) -> None:
        """Test JSON parsing performance with large and complex data."""
        # Test various JSON parsing scenarios
        parsing_scenarios: List[Tuple[str, str]] = [
            # Large array
            ('{"moves": [' + ', '.join(['"UP"'] * 10000) + ']}', "large_array"),
            
            # Deeply nested structure
            ('{"level1": {"level2": {"level3": {"level4": {"level5": {"moves": ["UP"]}}}}}}', "deep_nesting"),
            
            # Many keys
            ('{' + ', '.join([f'"key{i}": "value{i}"' for i in range(1000)]) + ', "moves": ["UP"]}', "many_keys"),
            
            # Large string values
            ('{"moves": ["' + 'A' * 50000 + '"]}', "large_string"),
            
            # Mixed complex structure
            ('{"moves": ["UP"], "metadata": [' + ', '.join([f'{{"step": {i}, "data": "info_{i}"}}' for i in range(1000)]) + ']}', "complex_mixed"),
        ]
        
        performance_results: List[Tuple[str, float, bool]] = []
        
        for json_data, scenario_name in parsing_scenarios:
            # Test safe_json_parse performance
            start_time = time.time()
            iterations: int = 100
            
            for _ in range(iterations):
                result: Optional[Dict[str, Any]] = safe_json_parse(json_data)
            
            parse_time: float = (time.time() - start_time) / iterations
            parse_success: bool = result is not None
            
            performance_results.append((scenario_name, parse_time, parse_success))
            
            # Performance targets
            assert parse_time < 0.1  # Should parse in under 100ms
            
            # Test extract_json_from_text performance
            text_with_json: str = f"LLM Response: {json_data}"
            start_time = time.time()
            
            for _ in range(iterations):
                extract_json_from_text(text_with_json)
            
            extract_time: float = (time.time() - start_time) / iterations
            assert extract_time < 0.1  # Should extract in under 100ms
        
        print("JSON Parsing Performance Results:")
        for scenario, time_ms, success in performance_results:
            print(f"  {scenario}: {time_ms*1000:.2f}ms, success: {success}")

    def test_game_state_serialization_performance(self) -> None:
        """Test performance of game state serialization/deserialization."""
        # Create complex game state
        controller: GameController = GameController(grid_size=25, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Populate with substantial data
        for i in range(5000):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            controller.make_move(move)
            
            if i % 10 == 0:
                game_data.add_llm_communication(
                    f"Complex prompt {i} with lots of context and details",
                    f"Detailed response {i} with comprehensive move analysis"
                )
                game_data.add_token_usage(prompt_tokens=150, completion_tokens=75)
            
            if controller.game_state.game_over:
                controller.reset()
        
        # Test serialization performance
        start_time: float = time.time()
        iterations: int = 100
        
        for _ in range(iterations):
            json_data: Dict[str, Any] = game_data.to_dict()
            json_str: str = game_data.to_json()
        
        serialization_time: float = (time.time() - start_time) / iterations
        
        # Test deserialization performance
        json_str = game_data.to_json()
        start_time = time.time()
        
        for _ in range(iterations):
            new_game_data: GameData = GameData()
            data_dict: Dict[str, Any] = safe_json_parse(json_str)
            if data_dict:
                # Simulate loading from dict
                new_game_data.score = data_dict.get("score", 0)
                new_game_data.steps = data_dict.get("steps", 0)
        
        deserialization_time: float = (time.time() - start_time) / iterations
        
        # Performance targets
        assert serialization_time < 0.01  # Under 10ms per serialization
        assert deserialization_time < 0.01  # Under 10ms per deserialization
        
        print(f"Serialization performance: {serialization_time*1000:.2f}ms")
        print(f"Deserialization performance: {deserialization_time*1000:.2f}ms")

    def test_apple_generation_performance_in_constrained_space(self) -> None:
        """Test apple generation performance when grid is nearly full."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Test with increasing constraint levels
        constraint_levels: List[float] = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        for constraint_level in constraint_levels:
            grid_size: int = controller.grid_size
            total_positions: int = grid_size * grid_size
            snake_length: int = int(total_positions * constraint_level)
            
            # Create large snake
            snake_positions: List[List[int]] = []
            for i in range(snake_length):
                x: int = i % grid_size
                y: int = i // grid_size
                snake_positions.append([x, y])
            
            controller.snake_positions = np.array(snake_positions, dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            # Measure apple generation performance
            start_time: float = time.time()
            iterations: int = 1000
            
            for _ in range(iterations):
                apple: NDArray[np.int_] = controller._generate_apple()
                
                # Verify apple is valid
                apple_list: List[int] = apple.tolist()
                assert 0 <= apple_list[0] < grid_size
                assert 0 <= apple_list[1] < grid_size
                
                # Verify apple is not on snake
                for snake_pos in controller.snake_positions:
                    assert not np.array_equal(apple, snake_pos)
            
            generation_time: float = (time.time() - start_time) / iterations
            
            # Performance should remain reasonable even with high constraint
            max_time: float = 0.001 if constraint_level < 0.9 else 0.01  # 1ms normally, 10ms when very constrained
            assert generation_time < max_time
            
            print(f"Apple generation at {constraint_level*100:.0f}% constraint: {generation_time*1000:.2f}ms")

    def test_llm_client_stress_testing(self) -> None:
        """Test LLM client under stress conditions."""
        mock_provider: Mock = Mock()
        mock_provider.is_available.return_value = True
        
        # Test rapid-fire requests
        start_time: float = time.time()
        request_count: int = 1000
        
        responses: List[str] = [
            '{"moves": ["UP"]}',
            '{"moves": ["RIGHT"]}', 
            '{"moves": ["DOWN"]}',
            '{"moves": ["LEFT"]}',
            'invalid response',
            '{"moves": []}',
        ]
        
        mock_provider.generate_response.side_effect = lambda prompt: responses[hash(prompt) % len(responses)]
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Make many rapid requests
        for i in range(request_count):
            prompt: str = f"Game state {i}: snake at position [5,5], apple at [7,7]"
            response: str = client.generate_response(prompt)
            
            # Verify response handling
            assert isinstance(response, str)
            assert len(response) > 0
        
        total_time: float = time.time() - start_time
        requests_per_second: float = request_count / total_time
        
        print(f"LLM client performance: {requests_per_second:.1f} requests/second")
        assert requests_per_second > 10000  # Should handle very high request rates

    def test_system_integration_under_load(self, temp_dir: str) -> None:
        """Test full system integration under heavy load."""
        # Create multiple game sessions running simultaneously
        session_count: int = 5
        moves_per_session: int = 2000
        
        session_results: List[Dict[str, Any]] = []
        
        for session_id in range(session_count):
            controller: GameController = GameController(grid_size=15, use_gui=False)
            game_data: GameData = controller.game_state
            
            game_data.set_llm_info(
                primary_provider=f"provider_{session_id}",
                primary_model=f"model_{session_id}"
            )
            
            start_time: float = time.time()
            moves_made: int = 0
            files_saved: int = 0
            
            for i in range(moves_per_session):
                # Make move
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                moves_made += 1
                
                # Simulate LLM interaction
                if i % 50 == 0:
                    game_data.add_llm_communication(
                        f"Session {session_id} step {i}",
                        f"Response for session {session_id}"
                    )
                    game_data.add_token_usage(prompt_tokens=100, completion_tokens=50)
                
                # Save state periodically
                if i % 200 == 0:
                    save_path: str = os.path.join(temp_dir, f"session_{session_id}_step_{i}.json")
                    success: bool = game_data.save_to_file(save_path)
                    if success:
                        files_saved += 1
                
                # Reset on collision
                if collision:
                    controller.reset()
            
            session_time: float = time.time() - start_time
            
            session_results.append({
                "session_id": session_id,
                "moves_made": moves_made,
                "time_taken": session_time,
                "files_saved": files_saved,
                "final_score": controller.score,
                "moves_per_second": moves_made / session_time
            })
        
        # Verify all sessions completed successfully
        assert len(session_results) == session_count
        
        total_moves: int = sum(result["moves_made"] for result in session_results)
        total_time: float = max(result["time_taken"] for result in session_results)
        overall_performance: float = total_moves / total_time
        
        print(f"System integration under load: {overall_performance:.1f} moves/second across {session_count} sessions")
        
        # Performance targets
        assert overall_performance > 1000  # Should maintain good performance under load
        assert all(result["files_saved"] > 0 for result in session_results)  # File I/O should work 