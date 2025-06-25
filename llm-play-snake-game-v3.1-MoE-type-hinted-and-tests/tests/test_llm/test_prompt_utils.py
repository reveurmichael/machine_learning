"""Tests for llm.prompt_utils module."""

import pytest
from unittest.mock import Mock, patch
from typing import List, Sequence

from llm.prompt_utils import (
    prepare_snake_prompt,
    create_parser_prompt,
    format_body_cells_str,
)


class TestPrepareSnakePrompt:
    """Test class for prepare_snake_prompt function."""

    def test_prepare_snake_prompt_basic(self):
        """Test basic snake prompt preparation."""
        head_pos = (5, 5)
        current_direction = "UP"
        body_cells = [(4, 5), (3, 5)]
        apple_pos = (7, 7)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain game state information
        assert "(5, 5)" in prompt or "5, 5" in prompt
        assert "UP" in prompt
        assert "(7, 7)" in prompt or "7, 7" in prompt

    def test_prepare_snake_prompt_empty_body(self):
        """Test prompt preparation with empty body (snake length 1)."""
        head_pos = (3, 3)
        current_direction = "RIGHT"
        body_cells = []
        apple_pos = (6, 3)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(prompt, str)
        assert "(3, 3)" in prompt or "3, 3" in prompt
        assert "RIGHT" in prompt

    def test_prepare_snake_prompt_none_direction(self):
        """Test prompt preparation with None current direction."""
        head_pos = (1, 1)
        current_direction = None
        body_cells = [(0, 1)]
        apple_pos = (9, 9)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(prompt, str)
        # Should handle None direction appropriately
        assert "None" in prompt or "NONE" in prompt

    def test_prepare_snake_prompt_long_body(self):
        """Test prompt preparation with long snake body."""
        head_pos = (5, 5)
        current_direction = "DOWN"
        body_cells = [(5, 6), (5, 7), (5, 8), (4, 8), (3, 8), (2, 8)]
        apple_pos = (0, 0)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(prompt, str)
        # Should include body information
        assert len(prompt) > 100  # Should be substantial with long body

    def test_prepare_snake_prompt_edge_positions(self):
        """Test prompt preparation with edge positions (grid boundaries)."""
        head_pos = (0, 0)
        current_direction = "UP"
        body_cells = [(0, 1)]
        apple_pos = (9, 9)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(prompt, str)
        assert "(0, 0)" in prompt or "0, 0" in prompt
        assert "(9, 9)" in prompt or "9, 9" in prompt

    def test_prepare_snake_prompt_all_directions(self):
        """Test prompt preparation with all possible directions."""
        directions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
        
        for direction in directions:
            prompt = prepare_snake_prompt(
                head_pos=(5, 5),
                current_direction=direction,
                body_cells=[(4, 5)],
                apple_pos=(7, 7)
            )
            
            assert isinstance(prompt, str)
            assert direction in prompt

    def test_prepare_snake_prompt_coordinates_format(self):
        """Test that coordinates are formatted correctly in prompt."""
        head_pos = (1, 2)
        current_direction = "LEFT"
        body_cells = [(2, 2), (3, 2)]
        apple_pos = (8, 9)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        # Should contain coordinate information in some format
        assert "1" in prompt and "2" in prompt  # Head coordinates
        assert "8" in prompt and "9" in prompt  # Apple coordinates


class TestCreateParserPrompt:
    """Test class for create_parser_prompt function."""

    def test_create_parser_prompt_basic(self):
        """Test basic parser prompt creation."""
        primary_response = "I think the snake should move UP to get closer to the apple."
        head_pos = (3, 3)
        body_cells = [(2, 3)]
        apple_pos = (3, 6)
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=head_pos,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(parser_prompt, str)
        assert primary_response in parser_prompt
        assert len(parser_prompt) > len(primary_response)  # Should add parsing instructions

    def test_create_parser_prompt_json_response(self):
        """Test parser prompt creation with JSON primary response."""
        primary_response = '{"moves": ["UP", "RIGHT"], "reasoning": "Move to apple"}'
        head_pos = (1, 1)
        body_cells = []
        apple_pos = (5, 5)
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=head_pos,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        assert isinstance(parser_prompt, str)
        assert primary_response in parser_prompt

    def test_create_parser_prompt_complex_response(self):
        """Test parser prompt creation with complex primary response."""
        primary_response = """
        Looking at the current state, I need to analyze the situation:
        - Snake head is at (2, 2)
        - Apple is at (5, 5)
        - Best path is UP, RIGHT, RIGHT, UP, UP
        {"moves": ["UP", "RIGHT", "RIGHT", "UP", "UP"], "reasoning": "Optimal path"}
        """
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=(2, 2),
            body_cells=[(1, 2)],
            apple_pos=(5, 5)
        )
        
        assert isinstance(parser_prompt, str)
        assert "Snake head is at (2, 2)" in parser_prompt

    def test_create_parser_prompt_error_response(self):
        """Test parser prompt creation with error primary response."""
        primary_response = "ERROR: Cannot determine safe move due to network issues"
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=(0, 0),
            body_cells=[],
            apple_pos=(9, 9)
        )
        
        assert isinstance(parser_prompt, str)
        assert "ERROR:" in parser_prompt

    def test_create_parser_prompt_no_path_response(self):
        """Test parser prompt creation with NO_PATH_FOUND response."""
        primary_response = "After analysis, NO_PATH_FOUND - snake is trapped"
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=(5, 5),
            body_cells=[(4, 5), (3, 5)],
            apple_pos=(1, 1)
        )
        
        assert isinstance(parser_prompt, str)
        assert "NO_PATH_FOUND" in parser_prompt

    def test_create_parser_prompt_unicode(self):
        """Test parser prompt creation with unicode in primary response."""
        primary_response = "Move towards the 🍎 apple! Path: UP ⬆️"
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=(3, 3),
            body_cells=[(3, 2)],
            apple_pos=(3, 6)
        )
        
        assert isinstance(parser_prompt, str)
        assert "🍎" in parser_prompt
        assert "⬆️" in parser_prompt

    def test_create_parser_prompt_includes_context(self):
        """Test that parser prompt includes game context."""
        primary_response = "Simple response"
        head_pos = (4, 5)
        body_cells = [(3, 5), (2, 5)]
        apple_pos = (8, 8)
        
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=head_pos,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        # Should include context about the game state
        assert "(4, 5)" in parser_prompt or "4, 5" in parser_prompt
        assert "(8, 8)" in parser_prompt or "8, 8" in parser_prompt


class TestFormatBodyCellsStr:
    """Test class for format_body_cells_str function."""

    def test_format_body_cells_empty(self):
        """Test formatting empty body cells."""
        body_cells = []
        
        result = format_body_cells_str(body_cells)
        
        assert isinstance(result, str)
        # Should handle empty list appropriately
        assert result in ["[]", "", "None"] or "empty" in result.lower()

    def test_format_body_cells_single(self):
        """Test formatting single body cell."""
        body_cells = [(3, 4)]
        
        result = format_body_cells_str(body_cells)
        
        assert isinstance(result, str)
        assert "3" in result and "4" in result

    def test_format_body_cells_multiple(self):
        """Test formatting multiple body cells."""
        body_cells = [(1, 2), (3, 4), (5, 6)]
        
        result = format_body_cells_str(body_cells)
        
        assert isinstance(result, str)
        # Should contain all coordinates
        assert "1" in result and "2" in result
        assert "3" in result and "4" in result
        assert "5" in result and "6" in result

    def test_format_body_cells_long_snake(self):
        """Test formatting long snake body."""
        body_cells = [(i, 5) for i in range(10)]  # 10 cells in a row
        
        result = format_body_cells_str(body_cells)
        
        assert isinstance(result, str)
        assert len(result) > 20  # Should be substantial for long snake

    def test_format_body_cells_various_positions(self):
        """Test formatting body cells at various grid positions."""
        body_cells = [(0, 0), (9, 9), (5, 3), (2, 8)]
        
        result = format_body_cells_str(body_cells)
        
        assert isinstance(result, str)
        # Should handle edge and middle positions
        assert "0" in result and "9" in result

    def test_format_body_cells_sequence_types(self):
        """Test formatting with different sequence types."""
        # Test with tuples
        body_tuples = [(1, 2), (3, 4)]
        result_tuples = format_body_cells_str(body_tuples)
        
        # Test with lists
        body_lists = [[1, 2], [3, 4]]
        result_lists = format_body_cells_str(body_lists)
        
        # Both should work and produce similar results
        assert isinstance(result_tuples, str)
        assert isinstance(result_lists, str)
        assert "1" in result_tuples and "2" in result_tuples
        assert "1" in result_lists and "2" in result_lists

    def test_format_body_cells_coordinate_format(self):
        """Test that coordinates are formatted in readable format."""
        body_cells = [(7, 8)]
        
        result = format_body_cells_str(body_cells)
        
        # Should be in a readable format (e.g., "(7, 8)" or "[7, 8]" or "7,8")
        assert isinstance(result, str)
        assert "7" in result and "8" in result
        # Common formatting patterns
        common_patterns = ["(7, 8)", "[7, 8]", "7,8", "7, 8"]
        assert any(pattern in result for pattern in common_patterns)


class TestPromptUtilsIntegration:
    """Test class for integration scenarios."""

    def test_full_prompt_preparation_pipeline(self):
        """Test complete prompt preparation pipeline."""
        # Simulate complete game state
        head_pos = (4, 6)
        current_direction = "RIGHT"
        body_cells = [(3, 6), (2, 6), (1, 6)]
        apple_pos = (8, 6)
        
        # Prepare primary prompt
        primary_prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        # Simulate primary LLM response
        primary_response = '{"moves": ["RIGHT", "RIGHT", "RIGHT"], "reasoning": "Direct path to apple"}'
        
        # Create parser prompt
        parser_prompt = create_parser_prompt(
            primary_response=primary_response,
            head_pos=head_pos,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        # Both prompts should be valid
        assert isinstance(primary_prompt, str) and len(primary_prompt) > 0
        assert isinstance(parser_prompt, str) and len(parser_prompt) > 0
        
        # Parser prompt should contain primary response
        assert primary_response in parser_prompt

    def test_body_cells_formatting_integration(self):
        """Test body cells formatting integration."""
        body_cells = [(2, 3), (3, 3), (4, 3)]
        
        # Format body cells
        formatted_body = format_body_cells_str(body_cells)
        
        # Use in prompt preparation
        prompt = prepare_snake_prompt(
            head_pos=(1, 3),
            current_direction="RIGHT",
            body_cells=body_cells,
            apple_pos=(6, 3)
        )
        
        # Formatted body should be represented in the prompt
        assert isinstance(formatted_body, str)
        assert isinstance(prompt, str)
        # Body information should be in the prompt
        assert "2" in prompt and "3" in prompt  # Body coordinates

    def test_error_handling_across_functions(self):
        """Test error handling across prompt utility functions."""
        # Test with various problematic inputs
        problematic_inputs = [
            # Valid but edge case inputs
            ((-1, -1), "INVALID", [], (10, 10)),  # Negative coordinates
            ((10, 10), "UP", [(9, 10)], (0, 0)),  # Edge coordinates
        ]
        
        for head_pos, direction, body_cells, apple_pos in problematic_inputs:
            try:
                # These should handle edge cases gracefully
                prompt = prepare_snake_prompt(
                    head_pos=head_pos,
                    current_direction=direction,
                    body_cells=body_cells,
                    apple_pos=apple_pos
                )
                assert isinstance(prompt, str)
                
                formatted_body = format_body_cells_str(body_cells)
                assert isinstance(formatted_body, str)
                
            except (ValueError, TypeError):
                # Some edge cases might raise exceptions - that's acceptable
                pass

    def test_unicode_handling_across_functions(self):
        """Test unicode handling across all functions."""
        # Test with unicode in responses
        unicode_response = "Move towards 🍎 apple! 方向: UP"
        
        parser_prompt = create_parser_prompt(
            primary_response=unicode_response,
            head_pos=(3, 3),
            body_cells=[(3, 2)],
            apple_pos=(3, 6)
        )
        
        # Should handle unicode properly
        assert isinstance(parser_prompt, str)
        assert "🍎" in parser_prompt

    def test_large_game_state_handling(self):
        """Test handling of large game states."""
        # Large snake body
        large_body = [(i, 5) for i in range(50)]
        
        # Should handle large inputs efficiently
        formatted_body = format_body_cells_str(large_body)
        assert isinstance(formatted_body, str)
        
        prompt = prepare_snake_prompt(
            head_pos=(51, 5),
            current_direction="RIGHT",
            body_cells=large_body,
            apple_pos=(60, 5)
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_consistency(self):
        """Test that prompts are consistent across multiple calls."""
        # Same inputs should produce same outputs
        head_pos = (2, 3)
        current_direction = "UP"
        body_cells = [(2, 2), (2, 1)]
        apple_pos = (5, 5)
        
        prompts = []
        for _ in range(3):
            prompt = prepare_snake_prompt(
                head_pos=head_pos,
                current_direction=current_direction,
                body_cells=body_cells,
                apple_pos=apple_pos
            )
            prompts.append(prompt)
        
        # All prompts should be identical
        assert all(p == prompts[0] for p in prompts)

    def test_prompt_content_validation(self):
        """Test that prompts contain expected content."""
        head_pos = (3, 4)
        current_direction = "LEFT"
        body_cells = [(4, 4), (5, 4)]
        apple_pos = (1, 2)
        
        prompt = prepare_snake_prompt(
            head_pos=head_pos,
            current_direction=current_direction,
            body_cells=body_cells,
            apple_pos=apple_pos
        )
        
        # Should contain game rules or instructions
        game_keywords = ["snake", "move", "apple", "direction", "position", "grid"]
        assert any(keyword in prompt.lower() for keyword in game_keywords)
        
        # Should contain current state
        assert str(head_pos[0]) in prompt and str(head_pos[1]) in prompt
        assert current_direction in prompt
        assert str(apple_pos[0]) in prompt and str(apple_pos[1]) in prompt 