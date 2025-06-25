"""Tests for config.game_constants module."""

import pytest
from unittest.mock import patch, MagicMock

from config.game_constants import (
    AVAILABLE_PROVIDERS,
    PAUSE_BETWEEN_MOVES_SECONDS,
    MAX_GAMES_ALLOWED,
    MAX_STEPS_ALLOWED,
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
    SLEEP_AFTER_EMPTY_STEP,
    SENTINEL_MOVES,
    VALID_MOVES,
    DIRECTIONS,
    END_REASON_MAP,
)


class TestGameConstants:
    """Test class for basic game constants."""

    def test_pause_between_moves_seconds_type(self):
        """Test that PAUSE_BETWEEN_MOVES_SECONDS is a float."""
        assert isinstance(PAUSE_BETWEEN_MOVES_SECONDS, float)
        assert PAUSE_BETWEEN_MOVES_SECONDS > 0

    def test_max_games_allowed_type_and_value(self):
        """Test that MAX_GAMES_ALLOWED is a positive integer."""
        assert isinstance(MAX_GAMES_ALLOWED, int)
        assert MAX_GAMES_ALLOWED > 0

    def test_max_steps_allowed_type_and_value(self):
        """Test that MAX_STEPS_ALLOWED is a positive integer."""
        assert isinstance(MAX_STEPS_ALLOWED, int)
        assert MAX_STEPS_ALLOWED > 0

    def test_max_consecutive_limits_are_positive(self):
        """Test that all max consecutive limits are positive integers."""
        limits = [
            MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
        ]
        for limit in limits:
            assert isinstance(limit, int)
            assert limit > 0

    def test_sleep_after_empty_step_type(self):
        """Test that SLEEP_AFTER_EMPTY_STEP is a number."""
        assert isinstance(SLEEP_AFTER_EMPTY_STEP, (int, float))
        assert SLEEP_AFTER_EMPTY_STEP >= 0


class TestSentinelMoves:
    """Test class for sentinel moves configuration."""

    def test_sentinel_moves_is_tuple(self):
        """Test that SENTINEL_MOVES is a tuple."""
        assert isinstance(SENTINEL_MOVES, tuple)

    def test_sentinel_moves_contents(self):
        """Test that SENTINEL_MOVES contains expected sentinel values."""
        expected_sentinels = {
            "INVALID_REVERSAL",
            "EMPTY",
            "SOMETHING_IS_WRONG",
            "NO_PATH_FOUND",
        }
        assert set(SENTINEL_MOVES) == expected_sentinels

    def test_sentinel_moves_are_strings(self):
        """Test that all sentinel moves are strings."""
        for move in SENTINEL_MOVES:
            assert isinstance(move, str)
            assert len(move) > 0

    def test_sentinel_moves_no_duplicates(self):
        """Test that SENTINEL_MOVES has no duplicates."""
        assert len(SENTINEL_MOVES) == len(set(SENTINEL_MOVES))


class TestValidMoves:
    """Test class for valid moves configuration."""

    def test_valid_moves_is_list(self):
        """Test that VALID_MOVES is a list."""
        assert isinstance(VALID_MOVES, list)

    def test_valid_moves_contents(self):
        """Test that VALID_MOVES contains exactly the four directions."""
        expected_moves = {"UP", "DOWN", "LEFT", "RIGHT"}
        assert set(VALID_MOVES) == expected_moves

    def test_valid_moves_are_strings(self):
        """Test that all valid moves are strings."""
        for move in VALID_MOVES:
            assert isinstance(move, str)
            assert len(move) > 0

    def test_valid_moves_no_duplicates(self):
        """Test that VALID_MOVES has no duplicates."""
        assert len(VALID_MOVES) == len(set(VALID_MOVES))

    def test_valid_moves_uppercase(self):
        """Test that all valid moves are uppercase."""
        for move in VALID_MOVES:
            assert move.isupper()


class TestDirections:
    """Test class for directions mapping."""

    def test_directions_is_dict(self):
        """Test that DIRECTIONS is a dictionary."""
        assert isinstance(DIRECTIONS, dict)

    def test_directions_keys_match_valid_moves(self):
        """Test that DIRECTIONS keys match VALID_MOVES."""
        assert set(DIRECTIONS.keys()) == set(VALID_MOVES)

    def test_directions_values_are_tuples(self):
        """Test that all direction values are tuples of two integers."""
        for direction, (dx, dy) in DIRECTIONS.items():
            assert isinstance(dx, int)
            assert isinstance(dy, int)
            assert -1 <= dx <= 1
            assert -1 <= dy <= 1
            assert abs(dx) + abs(dy) == 1  # Exactly one component is non-zero

    def test_directions_mappings(self):
        """Test specific direction mappings."""
        assert DIRECTIONS["UP"] == (0, 1)
        assert DIRECTIONS["DOWN"] == (0, -1)
        assert DIRECTIONS["LEFT"] == (-1, 0)
        assert DIRECTIONS["RIGHT"] == (1, 0)

    def test_directions_opposite_pairs(self):
        """Test that opposite directions cancel out."""
        up_x, up_y = DIRECTIONS["UP"]
        down_x, down_y = DIRECTIONS["DOWN"]
        left_x, left_y = DIRECTIONS["LEFT"]
        right_x, right_y = DIRECTIONS["RIGHT"]

        assert up_x + down_x == 0 and up_y + down_y == 0
        assert left_x + right_x == 0 and left_y + right_y == 0


class TestEndReasonMap:
    """Test class for end reason mapping."""

    def test_end_reason_map_is_dict(self):
        """Test that END_REASON_MAP is a dictionary."""
        assert isinstance(END_REASON_MAP, dict)

    def test_end_reason_map_keys_are_strings(self):
        """Test that all end reason keys are strings."""
        for key in END_REASON_MAP.keys():
            assert isinstance(key, str)
            assert len(key) > 0

    def test_end_reason_map_values_are_strings(self):
        """Test that all end reason values are strings."""
        for value in END_REASON_MAP.values():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_end_reason_map_expected_keys(self):
        """Test that END_REASON_MAP contains expected end reasons."""
        expected_keys = {
            "WALL",
            "SELF",
            "MAX_STEPS_REACHED",
            "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED",
            "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED",
            "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED",
            "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED",
        }
        assert set(END_REASON_MAP.keys()) == expected_keys

    def test_end_reason_map_values_are_user_friendly(self):
        """Test that end reason values are user-friendly."""
        # All values should be title case or contain spaces for readability
        for reason_code, user_message in END_REASON_MAP.items():
            # Should either be title case or contain spaces
            assert " " in user_message or user_message.istitle()

    def test_end_reason_map_no_duplicate_values(self):
        """Test that END_REASON_MAP has no duplicate user messages."""
        values = list(END_REASON_MAP.values())
        assert len(values) == len(set(values))


class TestAvailableProviders:
    """Test class for available providers."""

    @patch('config.game_constants.list_providers')
    def test_available_providers_calls_list_providers(self, mock_list_providers):
        """Test that AVAILABLE_PROVIDERS is populated by calling list_providers."""
        # Import again to trigger the function call
        mock_list_providers.return_value = ["provider1", "provider2"]
        
        # Re-import to test the function call
        import importlib
        import config.game_constants
        importlib.reload(config.game_constants)
        
        # Verify the function was called
        mock_list_providers.assert_called_once()

    def test_available_providers_is_list_or_tuple(self):
        """Test that AVAILABLE_PROVIDERS is a list or tuple."""
        assert isinstance(AVAILABLE_PROVIDERS, (list, tuple))

    @patch('config.game_constants.list_providers')
    def test_available_providers_handles_empty_list(self, mock_list_providers):
        """Test handling when list_providers returns empty list."""
        mock_list_providers.return_value = []
        
        import importlib
        import config.game_constants
        importlib.reload(config.game_constants)
        
        assert config.game_constants.AVAILABLE_PROVIDERS == []

    @patch('config.game_constants.list_providers')
    def test_available_providers_handles_exception(self, mock_list_providers):
        """Test handling when list_providers raises an exception."""
        mock_list_providers.side_effect = ImportError("Module not found")
        
        # Should not crash when importing
        try:
            import importlib
            import config.game_constants
            importlib.reload(config.game_constants)
        except ImportError:
            pytest.fail("Should handle list_providers exceptions gracefully")


class TestConstantsIntegrity:
    """Test class for cross-constant validation."""

    def test_max_consecutive_limits_reasonable_values(self):
        """Test that max consecutive limits have reasonable values."""
        # They should be small positive integers (not huge values)
        limits = [
            MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
        ]
        for limit in limits:
            assert 1 <= limit <= 100  # Reasonable range

    def test_max_steps_reasonable_value(self):
        """Test that MAX_STEPS_ALLOWED has a reasonable value."""
        # Should be large enough for meaningful games but not infinite
        assert 50 <= MAX_STEPS_ALLOWED <= 10000

    def test_max_games_reasonable_value(self):
        """Test that MAX_GAMES_ALLOWED has a reasonable value."""
        # Should be positive but not too large
        assert 1 <= MAX_GAMES_ALLOWED <= 1000

    def test_pause_between_moves_reasonable_value(self):
        """Test that PAUSE_BETWEEN_MOVES_SECONDS has a reasonable value."""
        # Should be positive but not too long
        assert 0.1 <= PAUSE_BETWEEN_MOVES_SECONDS <= 10.0

    def test_sleep_after_empty_step_reasonable_value(self):
        """Test that SLEEP_AFTER_EMPTY_STEP has a reasonable value."""
        # Should be non-negative and not too long (in minutes)
        assert 0 <= SLEEP_AFTER_EMPTY_STEP <= 60  # Up to 1 hour

    def test_sentinel_moves_not_in_valid_moves(self):
        """Test that sentinel moves don't overlap with valid moves."""
        sentinel_set = set(SENTINEL_MOVES)
        valid_set = set(VALID_MOVES)
        assert sentinel_set.isdisjoint(valid_set)

    def test_directions_cover_all_valid_moves(self):
        """Test that DIRECTIONS covers all VALID_MOVES."""
        assert set(DIRECTIONS.keys()) == set(VALID_MOVES)

    def test_end_reason_keys_logical_consistency(self):
        """Test that end reason keys are logically consistent with constants."""
        # Should have reasons for all the max consecutive limits
        expected_consecutive_reasons = [
            "MAX_CONSECUTIVE_EMPTY_MOVES_REACHED",
            "MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED", 
            "MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED",
            "MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED",
        ]
        
        for reason in expected_consecutive_reasons:
            assert reason in END_REASON_MAP
        
        # Should have basic game end reasons
        assert "WALL" in END_REASON_MAP
        assert "SELF" in END_REASON_MAP
        assert "MAX_STEPS_REACHED" in END_REASON_MAP


class TestConstantsImmutability:
    """Test class for constants immutability expectations."""

    def test_sentinel_moves_immutable(self):
        """Test that SENTINEL_MOVES is immutable (tuple)."""
        assert isinstance(SENTINEL_MOVES, tuple)
        # Tuples are immutable, so this is good

    def test_valid_moves_mutable_but_documented(self):
        """Test that VALID_MOVES is a list (mutable but should not be changed)."""
        # This documents that it's a list - in production code,
        # it should not be modified
        assert isinstance(VALID_MOVES, list)
        original_length = len(VALID_MOVES)
        
        # Verify it can be modified (but shouldn't be in practice)
        VALID_MOVES.append("TEST")
        assert len(VALID_MOVES) == original_length + 1
        
        # Clean up
        VALID_MOVES.pop()
        assert len(VALID_MOVES) == original_length

    def test_directions_mutable_but_documented(self):
        """Test that DIRECTIONS is a dict (mutable but should not be changed)."""
        assert isinstance(DIRECTIONS, dict)
        original_keys = set(DIRECTIONS.keys())
        
        # Verify it can be modified (but shouldn't be in practice)
        DIRECTIONS["TEST"] = (0, 0)
        assert "TEST" in DIRECTIONS
        
        # Clean up
        del DIRECTIONS["TEST"]
        assert set(DIRECTIONS.keys()) == original_keys

    def test_end_reason_map_mutable_but_documented(self):
        """Test that END_REASON_MAP is a dict (mutable but should not be changed)."""
        assert isinstance(END_REASON_MAP, dict)
        original_keys = set(END_REASON_MAP.keys())
        
        # Verify it can be modified (but shouldn't be in practice)
        END_REASON_MAP["TEST"] = "Test Reason"
        assert "TEST" in END_REASON_MAP
        
        # Clean up
        del END_REASON_MAP["TEST"]
        assert set(END_REASON_MAP.keys()) == original_keys 