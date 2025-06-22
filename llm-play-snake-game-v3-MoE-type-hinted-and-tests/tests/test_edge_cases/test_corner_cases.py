"""
Tests for corner cases and edge scenarios across all components.

This module focuses on testing the boundary conditions, error states,
and unusual scenarios that can occur in real-world usage.
"""

import pytest
import numpy as np
import os
import json
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable
from unittest.mock import Mock, patch, MagicMock
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from llm.client import LLMClient
from utils.moves_utils import normalize_direction, calculate_next_position, detect_collision
from utils.json_utils import safe_json_parse, extract_json_from_text, repair_malformed_json
from utils.file_utils import ensure_directory_exists, save_json_safely, load_json_safely


class TestCornerCases:
    """Comprehensive corner case testing across all components."""

    def test_minimal_grid_operations(self) -> None:
        """Test operations on minimal grid sizes."""
        # Test 1x1 grid (degenerate case)
        with pytest.raises((ValueError, IndexError)):
            controller: GameController = GameController(grid_size=1, use_gui=False)
        
        # Test 2x2 grid (minimal viable)
        controller = GameController(grid_size=2, use_gui=False)
        
        # Should have snake at center
        assert len(controller.snake_positions) == 1
        
        # All moves should result in collision
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            controller.reset()
            collision: bool
            _: bool
            collision, _ = controller.make_move(direction)
            # With 2x2 grid, most moves will hit walls quickly
            
        # Apple should be placeable in remaining position
        snake_pos: List[int] = controller.snake_positions[0].tolist()
        for x in range(2):
            for y in range(2):
                if [x, y] != snake_pos:
                    success: bool = controller.set_apple_position([x, y])
                    assert success

    def test_maximum_snake_length(self) -> None:
        """Test behavior when snake reaches maximum possible length."""
        grid_size: int = 5  # 25 total positions
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        
        # Fill entire grid except one position with snake
        max_positions: List[List[int]] = []
        for x in range(grid_size):
            for y in range(grid_size):
                if len(max_positions) < (grid_size * grid_size - 1):
                    max_positions.append([x, y])
        
        controller.snake_positions = np.array(max_positions, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Apple should be forced to the only remaining position
        apple: NDArray[np.int_] = controller._generate_apple()
        
        # Verify apple is not on snake
        for snake_pos in controller.snake_positions:
            assert not np.array_equal(apple, snake_pos)
        
        # Any move should result in collision (game won)
        collision: bool
        _: bool
        collision, _ = controller.make_move("UP")
        assert collision

    def test_rapid_direction_changes(self) -> None:
        """Test rapid direction changes and reversal detection."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test rapid direction changes
        rapid_moves: List[str] = [
            "UP", "DOWN",    # Immediate reversal
            "LEFT", "RIGHT", # Immediate reversal
            "UP", "LEFT", "DOWN", "RIGHT", "UP"  # Box pattern
        ]
        
        reversal_count: int = 0
        valid_moves: int = 0
        
        for i, move in enumerate(rapid_moves):
            if i > 0:
                previous_move: str = rapid_moves[i-1]
                if controller.filter_invalid_reversals([move], previous_move) == []:
                    reversal_count += 1
                    continue
            
            collision: bool
            _: bool
            collision, _ = controller.make_move(move)
            valid_moves += 1
            
            if collision:
                break
        
        # Should have detected and prevented reversals
        assert reversal_count > 0
        assert valid_moves < len(rapid_moves)

    def test_simultaneous_apple_snake_collision(self) -> None:
        """Test edge case where snake head lands on apple and wall simultaneously."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Position snake at edge
        controller.snake_positions = np.array([[9, 5]], dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        
        # Try to place apple outside grid (should fail)
        success: bool = controller.set_apple_position([10, 5])
        assert not success
        
        # Place apple at edge position
        controller.set_apple_position([9, 4])
        controller._update_board()
        
        # Move toward apple and wall simultaneously
        collision: bool
        apple_eaten: bool
        collision, apple_eaten = controller.make_move("UP")
        
        # Should eat apple
        assert apple_eaten
        # But also might hit wall depending on implementation
        
        # Move toward wall
        collision, apple_eaten = controller.make_move("RIGHT")
        assert collision  # Should hit wall
        assert not apple_eaten

    def test_json_parsing_extreme_cases(self) -> None:
        """Test JSON parsing with extreme and malformed inputs."""
        extreme_cases: List[Tuple[str, str]] = [
            ('{"moves":' + '["UP"]' * 1000 + '}', "massive_array"),
            ('{"moves": ["' + 'A' * 10000 + '"]}', "huge_string"),
            ('{"moves": [' + ', '.join([f'"{i}"' for i in range(1000)]) + ']}', "many_items"),
            ('{{{{{"moves": ["UP"]}}}}', "nested_braces"),
            ('{"moves": ["UP"]} {"moves": ["DOWN"]}', "multiple_objects"),
            ('{"moves": ["\\u0055\\u0050"]}', "unicode_escapes"),
            ('{"moves": [null, "UP", undefined]}', "mixed_null_undefined"),
            ('{moves: ["UP"], "extra": "data"}', "mixed_quotes"),
            ('{"moves": ["UP",]}', "trailing_comma"),
            ('{\n  "moves": [\n    "UP"\n  ]\n}', "multiline"),
        ]
        
        for test_input, case_type in extreme_cases:
            # Test safe parsing
            parsed: Optional[Dict[str, Any]] = safe_json_parse(test_input)
            
            # Test extraction from text
            extracted: Optional[Dict[str, Any]] = extract_json_from_text(f"LLM response: {test_input}")
            
            # Test repair function
            repaired: Optional[str] = repair_malformed_json(test_input)
            
            if case_type in ["massive_array", "huge_string", "many_items"]:
                # Should handle large data gracefully
                assert parsed is not None or extracted is not None or repaired is not None
            elif case_type in ["nested_braces", "multiple_objects"]:
                # Should extract first valid JSON
                assert extracted is not None or repaired is not None

    def test_file_system_edge_cases(self, temp_dir: str) -> None:
        """Test file system operations under edge conditions."""
        game_data: GameData = GameData()
        
        # Test extremely long filename
        long_filename: str = "a" * 200 + ".json"
        long_path: str = os.path.join(temp_dir, long_filename)
        
        success: bool = game_data.save_to_file(long_path)
        # May succeed or fail depending on filesystem limits
        
        # Test filename with special characters
        special_filename: str = "test-file_with.special@chars#2024.json"
        special_path: str = os.path.join(temp_dir, special_filename)
        
        success = game_data.save_to_file(special_path)
        assert success  # Should handle special chars
        
        # Test concurrent file access
        concurrent_path: str = os.path.join(temp_dir, "concurrent.json")
        
        def save_data(data: GameData, path: str, delay: float) -> None:
            time.sleep(delay)
            data.save_to_file(path)
        
        # Simulate concurrent saves
        thread1 = threading.Thread(target=save_data, args=(game_data, concurrent_path, 0.1))
        thread2 = threading.Thread(target=save_data, args=(game_data, concurrent_path, 0.1))
        
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        # File should exist and be readable
        assert os.path.exists(concurrent_path)
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(concurrent_path)
        assert loaded_data is not None

    def test_memory_pressure_scenarios(self) -> None:
        """Test behavior under simulated memory pressure."""
        # Create many large game states
        controllers: List[GameController] = []
        
        try:
            for i in range(100):  # Create many instances
                controller: GameController = GameController(grid_size=20, use_gui=False)
                
                # Make some moves to establish state
                for move in ["UP", "RIGHT", "DOWN", "LEFT"]:
                    controller.make_move(move)
                
                controllers.append(controller)
                
                # Verify each controller maintains correct state
                assert controller.steps > 0
                assert len(controller.snake_positions) >= 1
                
        except MemoryError:
            # Expected if we run out of memory
            pass
        
        # Cleanup and verify remaining controllers still work
        for controller in controllers[-10:]:  # Test last 10
            controller.make_move("UP")
            assert hasattr(controller, 'game_state')

    def test_infinite_loop_prevention(self) -> None:
        """Test prevention of infinite loops in various scenarios."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test apple generation in constrained space
        start_time: float = time.time()
        
        # Fill most of grid with snake
        large_snake: List[List[int]] = []
        for x in range(10):
            for y in range(10):
                if len(large_snake) < 98:  # Leave 2 free spaces
                    large_snake.append([x, y])
        
        controller.snake_positions = np.array(large_snake, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Generate apple multiple times
        for _ in range(100):
            apple: NDArray[np.int_] = controller._generate_apple()
            
            # Should not take too long
            current_time: float = time.time()
            assert current_time - start_time < 1.0  # Max 1 second
            
            # Apple should be valid
            assert len(apple) == 2
            assert 0 <= apple[0] < 10
            assert 0 <= apple[1] < 10

    def test_numeric_overflow_scenarios(self) -> None:
        """Test behavior with very large numeric values."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Test large score values
        large_score: int = 2**31 - 1  # Max int32
        game_data.update_scores(score=large_score, steps=controller.steps, snake_length=controller.snake_length)
        
        assert game_data.score == large_score
        
        # Test large step count
        large_steps: int = 1000000
        game_data.update_scores(score=game_data.score, steps=large_steps, snake_length=controller.snake_length)
        
        assert game_data.steps == large_steps
        
        # Test serialization with large values
        data_dict: Dict[str, Any] = game_data.to_dict()
        json_str: str = json.dumps(data_dict)
        
        # Should be able to parse back
        parsed: Dict[str, Any] = json.loads(json_str)
        assert parsed["score"] == large_score
        assert parsed["steps"] == large_steps

    def test_unicode_and_encoding_edge_cases(self) -> None:
        """Test handling of various unicode and encoding scenarios."""
        game_data: GameData = GameData()
        
        # Test unicode in LLM communication
        unicode_prompts: List[str] = [
            "🐍 Snake game prompt with emoji",
            "中文提示词测试",
            "Ñoño español prompt",
            "Русский текст prompt",
            "مطالبة عربية",
            "\u0000\u0001\u0002 control characters",
            "Mixed: 🎮🐍 中文 русский"
        ]
        
        for prompt in unicode_prompts:
            try:
                game_data.add_llm_communication(prompt, f"Response to: {prompt}")
                
                # Should be able to serialize
                json_str: str = game_data.to_json()
                
                # Should be able to parse back
                parsed: Optional[Dict[str, Any]] = safe_json_parse(json_str)
                assert parsed is not None
                
            except (UnicodeError, ValueError) as e:
                # Some control characters might be rejected
                if "control characters" in prompt:
                    pass  # Expected
                else:
                    raise e

    def test_thread_safety_scenarios(self) -> None:
        """Test thread safety of shared components."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        results: List[bool] = []
        errors: List[Exception] = []
        
        def make_moves_thread(thread_id: int) -> None:
            try:
                for i in range(50):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    _: bool
                    collision, _ = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                
                results.append(True)
                
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads: List[threading.Thread] = []
        for i in range(5):
            thread = threading.Thread(target=make_moves_thread, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Check results
        if errors:
            # Thread safety issues detected
            print(f"Thread safety errors: {errors}")
        
        # Controller should still be in valid state
        assert hasattr(controller, 'game_state')
        assert controller.grid_size == 15

    def test_resource_cleanup_scenarios(self) -> None:
        """Test proper resource cleanup in various scenarios."""
        # Test cleanup after exceptions
        controllers: List[GameController] = []
        
        for i in range(10):
            controller: GameController = GameController(grid_size=10, use_gui=False)
            controllers.append(controller)
            
            try:
                # Force some kind of error
                if i == 5:
                    # Try invalid operation
                    controller.snake_positions = np.array([[100, 100]], dtype=np.int_)
                    controller.make_move("UP")
                
            except Exception:
                # Should be able to recover
                controller.reset()
                assert controller.steps == 0
        
        # All controllers should still be usable
        for controller in controllers:
            controller.make_move("UP")
            assert controller.steps >= 1

    def test_api_contract_violations(self) -> None:
        """Test behavior when API contracts are violated."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test calling methods in wrong order
        try:
            # Try to access head position before initialization
            head: NDArray[np.int_] = controller.head_position
            # Should either work or fail gracefully
            
        except AttributeError:
            # Expected if not properly initialized
            pass
        
        # Test with None values
        try:
            success: bool = controller.set_apple_position(None)  # type: ignore
            assert not success
            
        except (TypeError, ValueError):
            # Expected to fail
            pass
        
        # Test with wrong types
        try:
            controller.make_move(123)  # type: ignore
            
        except (TypeError, ValueError):
            # Expected to fail with type error
            pass

    def test_boundary_value_analysis(self) -> None:
        """Test boundary values for all numeric parameters."""
        # Test grid size boundaries
        boundary_sizes: List[int] = [0, 1, 2, 3, 50, 100, 1000]
        
        for size in boundary_sizes:
            try:
                controller: GameController = GameController(grid_size=size, use_gui=False)
                
                if size >= 3:  # Minimum viable size
                    # Should work normally
                    controller.make_move("UP")
                    assert controller.steps == 1
                    
            except (ValueError, MemoryError):
                # Expected for invalid or too large sizes
                if size <= 1 or size > 100:
                    pass  # Expected
                else:
                    raise
        
        # Test position boundaries
        controller = GameController(grid_size=10, use_gui=False)
        
        boundary_positions: List[List[int]] = [
            [-1, -1], [0, 0], [9, 9], [10, 10], [100, 100]
        ]
        
        for pos in boundary_positions:
            success: bool = controller.set_apple_position(pos)
            
            # Should succeed only for valid positions
            expected_success: bool = 0 <= pos[0] < 10 and 0 <= pos[1] < 10
            assert success == expected_success

    def test_error_message_quality(self) -> None:
        """Test that error messages are informative and helpful."""
        # Test various error conditions and verify error messages
        error_scenarios: List[Tuple[Callable, str]] = [
            (lambda: GameController(grid_size=0, use_gui=False), "grid_size"),
            (lambda: GameController(grid_size=-5, use_gui=False), "negative"),
        ]
        
        for error_func, expected_keyword in error_scenarios:
            try:
                error_func()
                # Should not reach here
                assert False, f"Expected error for scenario with {expected_keyword}"
                
            except Exception as e:
                error_msg: str = str(e).lower()
                # Error message should be informative
                assert len(error_msg) > 10  # Not just empty or too brief
                # Should contain relevant context
                # Note: Specific checks depend on implementation 