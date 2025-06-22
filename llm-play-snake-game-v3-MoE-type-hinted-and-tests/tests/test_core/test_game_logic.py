"""Tests for core.game_logic module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import cast, List, Tuple

from core.game_logic import GameLogic
from config.ui_constants import GRID_SIZE


class TestGameLogic:
    """Test suite for GameLogic class."""

    def test_initialization(self) -> None:
        """Test GameLogic initialization."""
        game = GameLogic()
        
        assert game.grid_size == GRID_SIZE
        assert game.use_gui is True
        assert game.planned_moves == []
        assert game.processed_response == ""

    def test_initialization_custom_params(self) -> None:
        """Test GameLogic initialization with custom parameters."""
        game = GameLogic(grid_size=15, use_gui=False)
        
        assert game.grid_size == 15
        assert game.use_gui is False
        assert game.planned_moves == []
        assert game.processed_response == ""

    def test_head_property(self) -> None:
        """Test head property returns tuple of head position."""
        game = GameLogic()
        # Set up a head position
        game.head_position = [5, 6]
        
        head = game.head
        assert isinstance(head, tuple)
        assert head == (5, 6)

    def test_apple_property(self) -> None:
        """Test apple property returns tuple of apple position."""
        game = GameLogic()
        # Set up an apple position
        game.apple_position = [3, 4]
        
        apple = game.apple
        assert isinstance(apple, tuple)
        assert apple == (3, 4)

    def test_body_property(self) -> None:
        """Test body property returns list of tuples excluding head."""
        game = GameLogic()
        # Set up snake positions: body segments + head
        game.snake_positions = [[1, 2], [2, 2], [3, 2], [4, 2]]
        
        body = game.body
        assert isinstance(body, list)
        # Should return all but the last (head) in reverse order
        expected = [(3, 2), (2, 2), (1, 2)]
        assert body == expected

    def test_body_property_empty(self) -> None:
        """Test body property with only head position."""
        game = GameLogic()
        game.snake_positions = [[5, 5]]  # Only head
        
        body = game.body
        assert body == []

    def test_draw_with_gui(self) -> None:
        """Test draw method when GUI is available."""
        game = GameLogic(use_gui=True)
        mock_gui = Mock()
        game.gui = mock_gui
        game.board = [[0] * 10 for _ in range(10)]
        game.board_info = "test info"
        game.head_position = [5, 5]
        game.score = 10
        game.steps = 20
        game.planned_moves = ["UP", "DOWN"]
        game.processed_response = "test response"
        
        game.draw()
        
        mock_gui.draw_board.assert_called_once_with(
            game.board, game.board_info, game.head_position
        )
        expected_game_info = {
            'score': 10,
            'steps': 20,
            'planned_moves': ["UP", "DOWN"],
            'llm_response': "test response"
        }
        mock_gui.draw_game_info.assert_called_once_with(expected_game_info)

    def test_draw_without_gui(self) -> None:
        """Test draw method when GUI is not available."""
        game = GameLogic(use_gui=False)
        # Should not raise any errors
        game.draw()

    def test_draw_with_gui_disabled(self) -> None:
        """Test draw method when use_gui is False."""
        game = GameLogic(use_gui=False)
        mock_gui = Mock()
        game.gui = mock_gui
        
        game.draw()
        
        # GUI methods should not be called
        mock_gui.draw_board.assert_not_called()
        mock_gui.draw_game_info.assert_not_called()

    def test_reset(self) -> None:
        """Test reset method clears LLM-specific state."""
        game = GameLogic()
        # Set up some state
        game.planned_moves = ["UP", "DOWN", "LEFT"]
        game.processed_response = "test response"
        
        with patch.object(game.__class__.__bases__[0], 'reset') as mock_super_reset:
            game.reset()
            
            mock_super_reset.assert_called_once()
            assert game.planned_moves == []
            assert game.processed_response == ""

    @patch('core.game_logic.prepare_snake_prompt')
    def test_get_state_representation(self, mock_prepare_prompt: Mock) -> None:
        """Test get_state_representation method."""
        game = GameLogic()
        game.head_position = [5, 5]
        game.apple_position = [3, 3]
        game.current_direction = [0, -1]  # UP
        
        with patch.object(game, '_get_current_direction_key', return_value="UP"):
            mock_prepare_prompt.return_value = "test prompt"
            
            result = game.get_state_representation()
            
            mock_prepare_prompt.assert_called_once_with(
                head_position=[5, 5],
                body_positions=game.body,
                apple_position=[3, 3],
                current_direction="UP"
            )
            assert result == "test prompt"

    @patch('core.game_logic.prepare_snake_prompt')
    def test_get_state_representation_no_direction(self, mock_prepare_prompt: Mock) -> None:
        """Test get_state_representation with no current direction."""
        game = GameLogic()
        game.head_position = [5, 5]
        game.apple_position = [3, 3]
        game.current_direction = None
        
        mock_prepare_prompt.return_value = "test prompt"
        
        result = game.get_state_representation()
        
        mock_prepare_prompt.assert_called_once_with(
            head_position=[5, 5],
            body_positions=game.body,
            apple_position=[3, 3],
            current_direction="NONE"
        )
        assert result == "test prompt"

    @patch('core.game_logic.parse_llm_response')
    @patch('core.game_logic.process_response_for_display')
    def test_parse_llm_response_success(self, mock_process_response: Mock, mock_parse: Mock) -> None:
        """Test successful LLM response parsing."""
        game = GameLogic()
        response = '{"moves": ["UP", "DOWN"]}'
        mock_parse.return_value = "UP"
        
        result = game.parse_llm_response(response)
        
        mock_parse.assert_called_once_with(response, mock_process_response, game)
        assert result == "UP"

    @patch('core.game_logic.parse_llm_response')
    @patch('core.game_logic.process_response_for_display')
    def test_parse_llm_response_error(self, mock_process_response: Mock, mock_parse: Mock) -> None:
        """Test LLM response parsing with error."""
        game = GameLogic()
        game.game_state = Mock()
        response = 'invalid json'
        mock_parse.side_effect = ValueError("Invalid JSON")
        
        result = game.parse_llm_response(response)
        
        assert result is None
        assert "ERROR: Failed to parse LLM response" in game.processed_response
        assert response[:200] in game.processed_response
        assert game.planned_moves == []
        game.game_state.record_something_is_wrong_move.assert_called_once()

    def test_get_next_planned_move_with_moves(self) -> None:
        """Test getting next planned move when moves are available."""
        game = GameLogic()
        game.planned_moves = ["UP", "DOWN", "LEFT"]
        
        next_move = game.get_next_planned_move()
        
        assert next_move == "UP"
        assert game.planned_moves == ["DOWN", "LEFT"]

    def test_get_next_planned_move_empty(self) -> None:
        """Test getting next planned move when no moves available."""
        game = GameLogic()
        game.planned_moves = []
        
        next_move = game.get_next_planned_move()
        
        assert next_move is None
        assert game.planned_moves == []

    def test_get_next_planned_move_single(self) -> None:
        """Test getting next planned move when only one move available."""
        game = GameLogic()
        game.planned_moves = ["RIGHT"]
        
        next_move = game.get_next_planned_move()
        
        assert next_move == "RIGHT"
        assert game.planned_moves == []


class TestGameLogicIntegration:
    """Integration tests for GameLogic with other components."""

    def test_full_game_cycle(self) -> None:
        """Test a complete game cycle with LLM integration."""
        game = GameLogic(use_gui=False)
        
        # Initialize game state
        game.reset()
        
        # Test state representation
        state = game.get_state_representation()
        assert isinstance(state, str)
        assert len(state) > 0
        
        # Test move planning
        game.planned_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Execute planned moves
        moves_executed = []
        while True:
            next_move = game.get_next_planned_move()
            if next_move is None:
                break
            moves_executed.append(next_move)
        
        assert moves_executed == ["UP", "DOWN", "LEFT", "RIGHT"]
        assert game.planned_moves == []

    @patch('core.game_logic.prepare_snake_prompt')
    @patch('core.game_logic.parse_llm_response')
    def test_llm_interaction_flow(self, mock_parse: Mock, mock_prepare: Mock) -> None:
        """Test complete LLM interaction flow."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Mock prompt preparation
        mock_prepare.return_value = "game state prompt"
        
        # Mock successful parsing
        mock_parse.return_value = "UP"
        
        # Get state representation
        state = game.get_state_representation()
        assert state == "game state prompt"
        
        # Parse LLM response
        result = game.parse_llm_response('{"moves": ["UP", "DOWN"]}')
        assert result == "UP"
        
        # Verify no errors were recorded
        game.game_state.record_something_is_wrong_move.assert_not_called()

    def test_error_recovery(self) -> None:
        """Test error recovery mechanisms."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Test with invalid response
        result = game.parse_llm_response("not json")
        
        assert result is None
        assert game.planned_moves == []
        assert "ERROR" in game.processed_response
        game.game_state.record_something_is_wrong_move.assert_called_once()

    def test_property_consistency(self) -> None:
        """Test that properties return consistent types and values."""
        game = GameLogic()
        
        # Set up game state
        game.head_position = [7, 8]
        game.apple_position = [2, 3]
        game.snake_positions = [[5, 6], [6, 6], [7, 8]]  # body + head
        
        # Test property types and consistency
        head = game.head
        apple = game.apple
        body = game.body
        
        assert isinstance(head, tuple)
        assert isinstance(apple, tuple)
        assert isinstance(body, list)
        assert all(isinstance(pos, tuple) for pos in body)
        
        # Test values
        assert head == (7, 8)
        assert apple == (2, 3)
        assert body == [(6, 6), (5, 6)]  # reversed, excluding head

    def test_comprehensive_llm_integration(self) -> None:
        """Test comprehensive LLM integration scenarios."""
        game = GameLogic(grid_size=10, use_gui=False)
        
        # Test state representation with different snake configurations
        game.snake_positions = [[5, 5], [5, 4], [5, 3]]
        game.head_position = [5, 3]
        game.apple_position = [7, 7]
        game.current_direction = [0, -1]  # UP
        
        with patch('core.game_logic.prepare_snake_prompt') as mock_prepare:
            mock_prepare.return_value = "test_prompt_output"
            
            state_repr = game.get_state_representation()
            
            # Verify prompt preparation was called with correct parameters
            mock_prepare.assert_called_once()
            args = mock_prepare.call_args
            assert args[1]['head_position'] == [5, 3]
            assert args[1]['apple_position'] == [7, 7]
            assert args[1]['current_direction'] == "UP"
            assert state_repr == "test_prompt_output"

    def test_llm_response_parsing_comprehensive(self) -> None:
        """Test comprehensive LLM response parsing scenarios."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Test successful parsing with various response formats
        test_responses = [
            '{"moves": ["UP", "RIGHT", "DOWN"]}',
            '{"move": "LEFT", "reasoning": "Moving towards apple"}',
            '{"direction": "UP", "confidence": 0.9}'
        ]
        
        for response in test_responses:
            with patch('core.game_logic.parse_llm_response') as mock_parse:
                mock_parse.return_value = "UP"
                
                result = game.parse_llm_response(response)
                assert result == "UP"
                mock_parse.assert_called_once()

    def test_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Test various error types
        error_scenarios = [
            (ValueError("Invalid JSON"), "Invalid JSON"),
            (KeyError("Missing key"), "Missing key"),
            (TypeError("Type error"), "Type error"),
            (Exception("Generic error"), "Generic error")
        ]
        
        for error, error_msg in error_scenarios:
            with patch('core.game_logic.parse_llm_response', side_effect=error):
                result = game.parse_llm_response('{"invalid": "json"}')
                
                assert result is None
                assert "ERROR: Failed to parse LLM response" in game.processed_response
                assert game.planned_moves == []
                game.game_state.record_something_is_wrong_move.assert_called()
                
                # Reset for next iteration
                game.game_state.reset_mock()
                game.processed_response = ""
                game.planned_moves = []

    def test_planned_moves_management(self) -> None:
        """Test comprehensive planned moves management."""
        game = GameLogic()
        
        # Test with empty planned moves
        assert game.get_next_planned_move() is None
        
        # Test with single move
        game.planned_moves = ["UP"]
        assert game.get_next_planned_move() == "UP"
        assert game.planned_moves == []
        
        # Test with multiple moves
        game.planned_moves = ["UP", "RIGHT", "DOWN", "LEFT"]
        moves = []
        while True:
            move = game.get_next_planned_move()
            if move is None:
                break
            moves.append(move)
        
        assert moves == ["UP", "RIGHT", "DOWN", "LEFT"]
        assert game.planned_moves == []

    def test_gui_integration_comprehensive(self) -> None:
        """Test comprehensive GUI integration scenarios."""
        game = GameLogic(use_gui=True)
        
        # Test with different GUI states
        mock_gui = Mock()
        game.gui = mock_gui
        
        # Set up game state
        game.board = [[0] * 10 for _ in range(10)]
        game.board_info = {"empty": 0, "snake": 1, "apple": 2}
        game.head_position = [5, 5]
        game.score = 15
        game.steps = 30
        game.planned_moves = ["UP", "RIGHT"]
        game.processed_response = "Moving towards apple"
        
        game.draw()
        
        # Verify GUI methods were called correctly
        mock_gui.draw_board.assert_called_once_with(
            game.board, game.board_info, game.head_position
        )
        
        expected_info = {
            'score': 15,
            'steps': 30,
            'planned_moves': ["UP", "RIGHT"],
            'llm_response': "Moving towards apple"
        }
        mock_gui.draw_game_info.assert_called_once_with(expected_info)

    def test_reset_functionality_comprehensive(self) -> None:
        """Test comprehensive reset functionality."""
        game = GameLogic()
        
        # Set up complex state
        game.planned_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        game.processed_response = "Complex LLM response with reasoning"
        game.score = 10
        game.steps = 25
        
        with patch.object(GameLogic.__bases__[0], 'reset') as mock_super_reset:
            game.reset()
            
            # Verify parent reset was called
            mock_super_reset.assert_called_once()
            
            # Verify LLM-specific state was reset
            assert game.planned_moves == []
            assert game.processed_response == ""

    def test_property_edge_cases(self) -> None:
        """Test property edge cases and boundary conditions."""
        game = GameLogic()
        
        # Test with minimal snake (head only)
        game.snake_positions = [[5, 5]]
        game.head_position = [5, 5]
        game.apple_position = [3, 3]
        
        assert game.head == (5, 5)
        assert game.apple == (3, 3)
        assert game.body == []
        
        # Test with maximum length snake
        max_positions = [[i, 0] for i in range(20)]  # 20-segment snake
        game.snake_positions = max_positions
        game.head_position = max_positions[-1]
        
        assert game.head == tuple(max_positions[-1])
        assert len(game.body) == 19  # All but head
        
        # Test body order (should be reversed excluding head)
        expected_body = [tuple(pos) for pos in max_positions[:-1]][::-1]
        assert game.body == expected_body

    def test_state_representation_edge_cases(self) -> None:
        """Test state representation with edge cases."""
        game = GameLogic()
        
        # Test with no current direction
        game.current_direction = None
        
        with patch('core.game_logic.prepare_snake_prompt') as mock_prepare:
            mock_prepare.return_value = "no_direction_prompt"
            
            game.get_state_representation()
            
            args = mock_prepare.call_args
            assert args[1]['current_direction'] == "NONE"
        
        # Test with invalid direction
        game.current_direction = [1, 1]  # Diagonal - invalid
        
        with patch.object(game, '_get_current_direction_key', return_value="UNKNOWN"):
            with patch('core.game_logic.prepare_snake_prompt') as mock_prepare:
                mock_prepare.return_value = "unknown_direction_prompt"
                
                game.get_state_representation()
                
                args = mock_prepare.call_args
                assert args[1]['current_direction'] == "UNKNOWN"

    def test_performance_characteristics(self) -> None:
        """Test performance characteristics of key operations."""
        import time
        
        game = GameLogic(grid_size=20, use_gui=False)
        
        # Test property access performance
        game.snake_positions = [[i, 0] for i in range(100)]
        game.head_position = game.snake_positions[-1]
        
        start_time = time.time()
        for _ in range(1000):
            _ = game.head
            _ = game.body
        property_time = time.time() - start_time
        
        assert property_time < 0.1  # Should be very fast
        
        # Test state representation performance
        start_time = time.time()
        for _ in range(100):
            with patch('core.game_logic.prepare_snake_prompt', return_value="test"):
                game.get_state_representation()
        state_time = time.time() - start_time
        
        assert state_time < 1.0  # Should complete quickly

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency with large game states."""
        game = GameLogic(grid_size=50, use_gui=False)
        
        # Create large snake
        large_snake = [[i % 50, i // 50] for i in range(1000)]
        game.snake_positions = large_snake
        game.head_position = large_snake[-1]
        
        # Test that properties work efficiently
        head = game.head
        body = game.body
        
        assert head == tuple(large_snake[-1])
        assert len(body) == 999
        
        # Test planned moves with large list
        game.planned_moves = ["UP", "DOWN", "LEFT", "RIGHT"] * 250  # 1000 moves
        
        # Should handle large move lists efficiently
        moves_retrieved = []
        for _ in range(100):
            move = game.get_next_planned_move()
            if move:
                moves_retrieved.append(move)
        
        assert len(moves_retrieved) == 100
        assert len(game.planned_moves) == 900

    def test_concurrent_operations_simulation(self) -> None:
        """Test behavior under simulated concurrent operations."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Simulate rapid state changes
        states = [
            {"snake": [[1, 1]], "apple": [2, 2], "direction": [0, -1]},
            {"snake": [[1, 1], [1, 2]], "apple": [3, 3], "direction": [1, 0]},
            {"snake": [[1, 1], [1, 2], [2, 2]], "apple": [4, 4], "direction": [0, 1]}
        ]
        
        for state in states:
            game.snake_positions = state["snake"]
            game.head_position = state["snake"][-1]
            game.apple_position = state["apple"]
            game.current_direction = state["direction"]
            
            # Test that properties remain consistent
            assert game.head == tuple(state["snake"][-1])
            assert game.apple == tuple(state["apple"])
            
            # Test state representation
            with patch('core.game_logic.prepare_snake_prompt', return_value="test"):
                repr_str = game.get_state_representation()
                assert isinstance(repr_str, str)

    def test_error_recovery_mechanisms(self) -> None:
        """Test error recovery mechanisms."""
        game = GameLogic()
        game.game_state = Mock()
        
        # Test recovery from parsing errors
        invalid_responses = [
            "not json at all",
            '{"incomplete": json',
            '{"empty": {}}',
            '',
            None
        ]
        
        for response in invalid_responses:
            with patch('core.game_logic.parse_llm_response', side_effect=Exception("Parse error")):
                result = game.parse_llm_response(str(response) if response else "")
                
                assert result is None
                assert game.planned_moves == []
                assert "ERROR" in game.processed_response
                
                # Reset state for next test
                game.planned_moves = []
                game.processed_response = ""
                game.game_state.reset_mock()

    def test_integration_with_parent_class(self) -> None:
        """Test integration with parent GameController class."""
        game = GameLogic(grid_size=15, use_gui=False)
        
        # Test that parent functionality is preserved
        assert hasattr(game, 'make_move')
        assert hasattr(game, 'reset')
        assert hasattr(game, 'set_apple_position')
        assert hasattr(game, 'filter_invalid_reversals')
        
        # Test that parent properties work
        assert game.grid_size == 15
        assert isinstance(game.board, np.ndarray)
        assert hasattr(game, 'game_state')
        
        # Test that LLM-specific functionality is added
        assert hasattr(game, 'planned_moves')
        assert hasattr(game, 'processed_response')
        assert hasattr(game, 'get_state_representation')
        assert hasattr(game, 'parse_llm_response')

    def test_serialization_compatibility(self) -> None:
        """Test compatibility with serialization for game state persistence."""
        game = GameLogic()
        
        # Set up complex state
        game.snake_positions = [[1, 1], [2, 1], [3, 1]]
        game.head_position = [3, 1]
        game.apple_position = [5, 5]
        game.planned_moves = ["UP", "RIGHT", "DOWN"]
        game.processed_response = "LLM reasoning text"
        game.score = 5
        game.steps = 10
        
        # Test that state can be extracted for serialization
        state_dict = {
            'head': game.head,
            'apple': game.apple,
            'body': game.body,
            'planned_moves': game.planned_moves,
            'processed_response': game.processed_response,
            'score': game.score,
            'steps': game.steps
        }
        
        # Verify all values are serializable
        import json
        json_str = json.dumps(state_dict)
        restored_state = json.loads(json_str)
        
        assert restored_state['head'] == list(game.head)
        assert restored_state['apple'] == list(game.apple)
        assert restored_state['planned_moves'] == game.planned_moves
        assert restored_state['score'] == game.score

    def test_boundary_conditions(self) -> None:
        """Test boundary conditions and edge cases."""
        game = GameLogic(grid_size=3, use_gui=False)  # Very small grid
        
        # Test with snake filling most of the grid
        game.snake_positions = [[0, 0], [0, 1], [0, 2], [1, 2]]
        game.head_position = [1, 2]
        game.apple_position = [2, 2]
        
        # Properties should still work correctly
        assert game.head == (1, 2)
        assert game.apple == (2, 2)
        assert len(game.body) == 3
        
        # Test state representation with crowded grid
        with patch('core.game_logic.prepare_snake_prompt') as mock_prepare:
            mock_prepare.return_value = "crowded_grid_prompt"
            result = game.get_state_representation()
            assert result == "crowded_grid_prompt" 