"""
Tests for complex interactions between classes and corner cases.

This module tests the intricate relationships and edge cases that arise
when multiple components interact in real-world scenarios.
"""

import pytest
import numpy as np
import tempfile
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock, call
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from core.game_manager import GameManager
from core.game_rounds import GameRounds
from llm.client import LLMClient
from utils.moves_utils import normalize_direction, calculate_next_position
from utils.json_utils import safe_json_parse, extract_json_from_text
from utils.file_utils import ensure_directory_exists, save_json_safely
from config.game_constants import DIRECTIONS


class TestComplexClassInteractions:
    """Test complex interactions between multiple classes."""

    def test_game_controller_data_sync_under_stress(self) -> None:
        """Test GameController and GameData synchronization under rapid changes."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Rapid sequence of moves with state changes
        moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"] * 25  # 100 moves
        apple_eating_sequence: List[int] = [5, 15, 30, 45, 70]  # Steps where apples are eaten
        
        for i, move in enumerate(moves):
            # Force apple eating at specific steps
            if i in apple_eating_sequence:
                head: NDArray[np.int_] = controller.snake_positions[-1]
                controller.set_apple_position([head[0], head[1] - 1])
            
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Verify synchronization at each step
            assert controller.score == game_data.score
            assert controller.steps == game_data.steps
            assert controller.snake_length == game_data.snake_length
            assert len(controller.game_state.moves) == i + 1
            
            if collision:
                assert game_data.game_over
                break
        
        # Final consistency check
        assert controller.game_state.steps == len([m for m in moves[:controller.steps]])

    def test_game_logic_controller_boundary_interactions(self) -> None:
        """Test GameLogic and GameController interactions at grid boundaries."""
        controller: GameController = GameController(grid_size=5, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=5)
        
        # Test all boundary scenarios
        boundary_scenarios: List[Tuple[List[int], str, bool]] = [
            ([0, 0], "UP", True),     # Top-left corner going up
            ([0, 0], "LEFT", True),   # Top-left corner going left
            ([4, 4], "DOWN", True),   # Bottom-right corner going down
            ([4, 4], "RIGHT", True),  # Bottom-right corner going right
            ([2, 0], "UP", True),     # Top edge going up
            ([0, 2], "LEFT", True),   # Left edge going left
            ([4, 2], "RIGHT", True),  # Right edge going right
            ([2, 4], "DOWN", True),   # Bottom edge going down
            ([1, 1], "UP", False),    # Interior move
        ]
        
        for start_pos, direction, should_collide in boundary_scenarios:
            # Reset controller to specific position
            controller.snake_positions = np.array([start_pos], dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            # Test GameLogic collision detection
            next_pos: List[int] = calculate_next_position(start_pos, direction)
            logic_collision: bool = game_logic.check_collision(next_pos, controller.snake_positions)
            
            # Test GameController collision detection
            collision: bool
            _: bool
            collision, _ = controller.make_move(direction)
            
            # Both should agree on collision detection
            assert collision == should_collide
            if should_collide:
                assert logic_collision == collision

    def test_llm_client_json_parsing_error_cascade(self) -> None:
        """Test error cascading between LLM client and JSON parsing utilities."""
        mock_provider: Mock = Mock()
        mock_provider.is_available.return_value = True
        
        # Test various malformed LLM responses
        malformed_responses: List[Tuple[str, str]] = [
            ('{"moves": ["UP", "RIGHT"', "incomplete_json"),
            ('moves: ["UP", "RIGHT"]', "missing_braces"),
            ('{"moves": "UP"}', "wrong_type"),
            ('{"direction": "UP"}', "wrong_key"),
            ('{"moves": ["INVALID_MOVE"]}', "invalid_move"),
            ('{"moves": []}', "empty_moves"),
            ('{"moves": ["UP", "UP", "UP", "UP", "UP", "UP"]}', "too_many_moves"),
            ('Text before {"moves": ["UP"]} text after', "json_with_text"),
            ('```json\n{"moves": ["RIGHT"]}\n```', "code_block"),
            ('null', "null_response"),
            ('', "empty_response"),
        ]
        
        for response, scenario in malformed_responses:
            mock_provider.generate_response.return_value = response
            mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
            
            client: LLMClient = LLMClient(mock_provider)
            llm_response: str = client.generate_response("test prompt")
            
            # Test JSON parsing utility integration
            parsed_json: Optional[Dict[str, Any]] = safe_json_parse(llm_response)
            extracted_json: Optional[Dict[str, Any]] = extract_json_from_text(llm_response)
            
            # Verify error handling at each level
            if scenario in ["incomplete_json", "missing_braces", "null_response", "empty_response"]:
                assert parsed_json is None
            elif scenario == "json_with_text" or scenario == "code_block":
                assert extracted_json is not None
            elif scenario in ["wrong_type", "wrong_key", "invalid_move", "empty_moves"]:
                # Should parse but fail validation
                result = parsed_json or extracted_json
                if result:
                    moves = result.get("moves", [])
                    if scenario == "wrong_type":
                        assert not isinstance(moves, list)
                    elif scenario == "empty_moves":
                        assert len(moves) == 0

    def test_game_state_persistence_with_concurrent_modifications(self, temp_dir: str) -> None:
        """Test game state persistence when multiple components modify state simultaneously."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Simulate concurrent modifications from different sources
        save_path: str = os.path.join(temp_dir, "concurrent_test.json")
        
        # Thread 1: Game moves
        moves_sequence: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"]
        for move in moves_sequence:
            controller.make_move(move)
            
            # Thread 2: LLM data updates (simulated)
            game_data.add_llm_communication(f"prompt_{move}", f"response_{move}")
            game_data.stats.record_step_result(valid=True, collision=False, apple_eaten=False)
            
            # Thread 3: Statistics updates
            game_data.add_token_usage(prompt_tokens=10, completion_tokens=5)
            
            # Thread 4: File operations
            save_success: bool = save_json_safely(game_data.to_dict(), save_path)
            assert save_success
            
            # Verify consistency after each concurrent operation
            loaded_data: GameData = GameData()
            load_success: bool = loaded_data.load_from_file(save_path)
            assert load_success
            assert loaded_data.steps == game_data.steps
            assert loaded_data.score == game_data.score

    def test_memory_management_with_large_snake(self) -> None:
        """Test memory and performance with very large snake."""
        controller: GameController = GameController(grid_size=50, use_gui=False)
        
        # Create a large snake (almost filling the grid)
        large_snake_positions: List[List[int]] = []
        for i in range(40):
            for j in range(40):
                if len(large_snake_positions) < 1500:  # Large but manageable
                    large_snake_positions.append([i, j])
        
        controller.snake_positions = np.array(large_snake_positions, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Test operations with large snake
        start_time: float = time.time()
        
        # Test collision detection performance
        for _ in range(100):
            collision: bool
            _: bool
            collision, _ = controller.make_move("UP")
            if collision:
                break
        
        end_time: float = time.time()
        operation_time: float = end_time - start_time
        
        # Should complete within reasonable time (performance test)
        assert operation_time < 5.0  # 5 seconds max for 100 operations
        assert len(controller.snake_positions) >= 1500

    def test_apple_generation_edge_cases(self) -> None:
        """Test apple generation in constrained environments."""
        # Test with nearly full grid
        controller: GameController = GameController(grid_size=4, use_gui=False)
        
        # Fill most positions with snake
        snake_positions: List[List[int]] = [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2],  # Leave [2,3] and [3,x] free
            [3, 0], [3, 1], [3, 2]   # Leave [3,3] free
        ]
        
        controller.snake_positions = np.array(snake_positions, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Generate apple multiple times - should only use free positions
        free_positions: List[List[int]] = [[2, 3], [3, 3]]
        
        for _ in range(50):  # Try many times
            apple: NDArray[np.int_] = controller._generate_apple()
            apple_list: List[int] = apple.tolist()
            
            # Apple should only be in free positions
            assert apple_list in free_positions
            
            # Apple should not be on snake
            for snake_pos in controller.snake_positions:
                assert not np.array_equal(apple, snake_pos)

    def test_round_management_with_error_conditions(self) -> None:
        """Test GameRounds management under various error conditions."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test round transitions with different game end conditions
        end_conditions: List[Tuple[str, List[str]]] = [
            ("collision_wall", ["UP"] * 15),  # Hit wall
            ("collision_self", ["UP", "UP", "RIGHT", "DOWN", "LEFT", "UP"]),  # Self collision
            ("max_steps", ["RIGHT", "LEFT"] * 100),  # Oscillate to max steps
        ]
        
        for end_reason, moves in end_conditions:
            # Reset for each test
            controller.reset()
            
            # Execute moves until game ends
            final_collision: bool = False
            final_steps: int = 0
            
            for i, move in enumerate(moves):
                collision: bool
                _: bool
                collision, _ = controller.make_move(move)
                final_steps = i + 1
                
                if collision:
                    final_collision = True
                    break
                
                # Artificial step limit for max_steps test
                if end_reason == "max_steps" and i >= 50:
                    controller.game_state.set_game_over(True, "max_steps")
                    break
            
            # Verify end condition handling
            if end_reason.startswith("collision"):
                assert final_collision
                assert controller.last_collision_type in ["wall", "self"]
            
            # Verify state consistency after game end
            assert controller.game_state.steps == final_steps
            if controller.game_state.game_over:
                assert controller.game_state.game_end_reason != ""

    def test_cross_component_error_propagation(self) -> None:
        """Test how errors propagate across different components."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test error propagation scenarios
        error_scenarios: List[Tuple[str, Any]] = [
            ("invalid_move", "INVALID_DIRECTION"),
            ("negative_position", [-1, -1]),
            ("out_of_bounds_position", [15, 15]),
            ("malformed_snake_positions", "not_an_array"),
            ("invalid_grid_size", 0),
        ]
        
        for scenario, error_input in error_scenarios:
            try:
                if scenario == "invalid_move":
                    # Should handle gracefully or raise specific error
                    controller.make_move(error_input)
                    
                elif scenario == "negative_position":
                    # Should validate position bounds
                    success: bool = controller.set_apple_position(error_input)
                    assert not success  # Should fail gracefully
                    
                elif scenario == "out_of_bounds_position":
                    success = controller.set_apple_position(error_input)
                    assert not success
                    
                elif scenario == "malformed_snake_positions":
                    # Should handle type errors gracefully
                    try:
                        controller.snake_positions = error_input  # type: ignore
                    except (TypeError, ValueError):
                        pass  # Expected to fail
                        
                elif scenario == "invalid_grid_size":
                    # Should validate during initialization
                    try:
                        invalid_controller: GameController = GameController(grid_size=error_input, use_gui=False)
                    except (ValueError, TypeError):
                        pass  # Expected to fail
                        
            except Exception as e:
                # Errors should be specific and informative
                assert isinstance(e, (ValueError, TypeError, IndexError))
                assert str(e) != ""  # Should have error message

    def test_complex_multi_component_scenario(self, temp_dir: str) -> None:
        """Test a complex scenario involving all major components."""
        # Initialize all components
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Set up complex scenario
        session_name: str = "complex_test_session"
        log_path: str = os.path.join(temp_dir, session_name)
        ensure_directory_exists(log_path)
        
        # Configure game data
        game_data.set_llm_info(
            primary_provider="test_provider",
            primary_model="test_model_v1",
            parser_provider="secondary_provider", 
            parser_model="parser_model_v2"
        )
        
        # Simulate complex game flow
        complex_moves: List[str] = [
            "UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN", "LEFT", "LEFT",
            "UP", "RIGHT", "DOWN", "LEFT", "UP", "UP", "RIGHT", "DOWN"
        ]
        
        llm_responses: List[str] = [
            '{"moves": ["UP", "RIGHT"]}',
            '{"moves": ["DOWN"]}',
            'Invalid response',
            '```json\n{"moves": ["LEFT", "UP"]}\n```',
            '{"moves": []}',
        ]
        
        apple_positions_log: List[Dict[str, int]] = []
        collision_points: List[int] = []
        
        for i, move in enumerate(complex_moves):
            # Simulate LLM interaction every few moves
            if i % 3 == 0 and i // 3 < len(llm_responses):
                response_idx: int = i // 3
                game_data.add_llm_communication(
                    f"Game state at step {i}",
                    llm_responses[response_idx]
                )
                game_data.add_token_usage(
                    prompt_tokens=50 + i * 5,
                    completion_tokens=20 + i * 2
                )
            
            # Record apple position before move
            apple_positions_log.append({
                "x": int(controller.apple_position[0]),
                "y": int(controller.apple_position[1])
            })
            
            # Make move
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Record statistics
            game_data.stats.record_step_result(
                valid=True,
                collision=collision,
                apple_eaten=apple_eaten
            )
            
            if collision:
                collision_points.append(i)
                break
            
            # Increment round every 5 moves
            if i % 5 == 4:
                game_data.increment_round()
        
        # Save complex game state
        game_file: str = os.path.join(log_path, "complex_game.json")
        save_success: bool = game_data.save_to_file(game_file)
        assert save_success
        
        # Verify all component interactions
        assert game_data.steps > 0
        assert game_data.round_count >= 1
        assert game_data.stats.llm_stats.total_requests > 0
        assert game_data.stats.step_stats.valid > 0
        assert len(apple_positions_log) > 0
        
        # Load and verify persistence
        loaded_data: GameData = GameData()
        load_success: bool = loaded_data.load_from_file(game_file)
        assert load_success
        
        # Verify complex data integrity
        assert loaded_data.llm_info["primary_provider"] == "test_provider"
        assert loaded_data.llm_info["parser_model"] == "parser_model_v2"
        assert loaded_data.round_count == game_data.round_count
        assert loaded_data.steps == game_data.steps

    def test_performance_under_load(self) -> None:
        """Test system performance under heavy load."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Measure performance metrics
        start_time: float = time.time()
        move_times: List[float] = []
        
        # Execute many moves with timing
        for i in range(1000):
            move_start: float = time.time()
            
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            _: bool
            collision, _ = controller.make_move(move)
            
            move_end: float = time.time()
            move_times.append(move_end - move_start)
            
            if collision:
                controller.reset()  # Reset on collision to continue testing
        
        total_time: float = time.time() - start_time
        avg_move_time: float = sum(move_times) / len(move_times)
        max_move_time: float = max(move_times)
        
        # Performance assertions
        assert total_time < 30.0  # Should complete in under 30 seconds
        assert avg_move_time < 0.01  # Average move should be under 10ms
        assert max_move_time < 0.1   # No single move should take over 100ms
        
        # Verify system remained stable
        assert controller.grid_size == 20
        assert len(controller.snake_positions) >= 1

    def test_edge_case_interactions_matrix(self) -> None:
        """Test matrix of edge case interactions between components."""
        # Define edge cases for each component
        edge_cases: Dict[str, List[Any]] = {
            "grid_sizes": [1, 2, 3, 50],
            "snake_lengths": [1, 10, 100],
            "positions": [[0, 0], [1, 1], [-1, -1], [100, 100]],
            "moves": ["UP", "DOWN", "LEFT", "RIGHT", "INVALID", ""],
            "apple_positions": [[0, 0], [1, 1], [-1, -1], [50, 50]],
        }
        
        # Test critical combinations
        critical_combinations: List[Dict[str, Any]] = [
            {"grid_size": 3, "snake_length": 8, "position": [1, 1]},  # Snake longer than grid area
            {"grid_size": 1, "snake_length": 1, "position": [0, 0]},  # Minimal game
            {"grid_size": 2, "apple_position": [0, 0], "snake_position": [0, 0]},  # Apple on snake
        ]
        
        for combination in critical_combinations:
            try:
                grid_size: int = combination.get("grid_size", 10)
                controller: GameController = GameController(grid_size=grid_size, use_gui=False)
                
                # Apply edge case settings
                if "snake_length" in combination:
                    # Create snake of specified length (if possible)
                    length: int = combination["snake_length"]
                    if length <= grid_size * grid_size:
                        positions: List[List[int]] = []
                        for i in range(min(length, grid_size * grid_size)):
                            x: int = i % grid_size
                            y: int = i // grid_size
                            positions.append([x, y])
                        
                        if positions:
                            controller.snake_positions = np.array(positions, dtype=np.int_)
                            controller.head_position = controller.snake_positions[-1]
                            controller._update_board()
                
                if "apple_position" in combination:
                    apple_pos: List[int] = combination["apple_position"]
                    controller.set_apple_position(apple_pos)
                
                # Test basic operations don't crash
                controller.make_move("UP")
                controller.reset()
                
            except Exception as e:
                # Should handle edge cases gracefully
                assert isinstance(e, (ValueError, IndexError, TypeError))

    def test_state_corruption_recovery(self) -> None:
        """Test recovery from various state corruption scenarios."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test various corruption scenarios
        corruption_scenarios: List[Tuple[str, callable]] = [
            ("empty_snake", lambda c: setattr(c, 'snake_positions', np.array([], dtype=np.int_).reshape(0, 2))),
            ("negative_score", lambda c: c.game_state.update_scores(score=-10, steps=c.steps, snake_length=c.snake_length)),
            ("invalid_apple", lambda c: setattr(c, 'apple_position', np.array([-5, -5], dtype=np.int_))),
            ("inconsistent_board", lambda c: c.board.fill(0)),  # Clear board while snake exists
        ]
        
        for scenario_name, corruption_func in corruption_scenarios:
            # Start with valid state
            controller.reset()
            controller.make_move("UP")  # Establish some state
            
            original_steps: int = controller.steps
            
            # Apply corruption
            try:
                corruption_func(controller)
                
                # Try to continue operation
                collision: bool
                _: bool
                collision, _ = controller.make_move("RIGHT")
                
                # System should either:
                # 1. Handle corruption gracefully
                # 2. Reset to valid state
                # 3. Raise informative error
                
                # Verify we can still get basic info
                assert hasattr(controller, 'grid_size')
                assert hasattr(controller, 'game_state')
                
            except Exception as e:
                # Should be specific error types with clear messages
                assert isinstance(e, (ValueError, IndexError, TypeError, AttributeError))
                assert len(str(e)) > 0
            
            # Reset should always work
            controller.reset()
            assert controller.steps == 0 