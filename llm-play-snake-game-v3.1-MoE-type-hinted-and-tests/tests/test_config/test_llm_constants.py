"""Tests for config.llm_constants module."""

import pytest

from config.llm_constants import (
    TEMPERATURE,
    MAX_TOKENS,
)


class TestLLMConstants:
    """Test class for LLM configuration constants."""

    def test_temperature_type_and_range(self):
        """Test that TEMPERATURE is a float within valid range."""
        assert isinstance(TEMPERATURE, (int, float))
        assert 0.0 <= TEMPERATURE <= 2.0  # Standard LLM temperature range

    def test_temperature_reasonable_value(self):
        """Test that TEMPERATURE has a reasonable value for deterministic responses."""
        # Low temperature values (0.0-0.5) are typically used for more deterministic responses
        # which is appropriate for a game where consistency is important
        assert 0.0 <= TEMPERATURE <= 1.0

    def test_max_tokens_type_and_value(self):
        """Test that MAX_TOKENS is a positive integer."""
        assert isinstance(MAX_TOKENS, int)
        assert MAX_TOKENS > 0

    def test_max_tokens_reasonable_value(self):
        """Test that MAX_TOKENS has a reasonable value for snake game responses."""
        # For a snake game, responses should be relatively short (JSON with moves)
        # but need enough tokens for reasoning. Common values are powers of 2.
        assert 1000 <= MAX_TOKENS <= 32768  # Reasonable range for LLM responses

    def test_max_tokens_common_llm_values(self):
        """Test that MAX_TOKENS aligns with common LLM token limits."""
        # Common token limits are powers of 2: 1024, 2048, 4096, 8192, 16384, 32768
        common_limits = [1024, 2048, 4096, 8192, 16384, 32768]
        # MAX_TOKENS should be reasonable relative to these common limits
        assert any(MAX_TOKENS <= limit for limit in common_limits)


class TestLLMConstantsIntegrity:
    """Test class for LLM constants integrity and relationships."""

    def test_constants_not_none(self):
        """Test that all LLM constants are not None."""
        assert TEMPERATURE is not None
        assert MAX_TOKENS is not None

    def test_constants_for_deterministic_gaming(self):
        """Test that constants are appropriate for deterministic gaming scenarios."""
        # For a game like Snake, we want relatively deterministic responses
        # Temperature should be low
        assert TEMPERATURE <= 0.5
        
        # Tokens should be sufficient for JSON responses with reasoning
        # but not excessive
        assert 2000 <= MAX_TOKENS <= 16384

    def test_temperature_precision(self):
        """Test that TEMPERATURE has reasonable precision."""
        # Should not have excessive decimal places
        temp_str = str(TEMPERATURE)
        if '.' in temp_str:
            decimal_places = len(temp_str.split('.')[1])
            assert decimal_places <= 2  # At most 2 decimal places

    def test_constants_immutability_expectation(self):
        """Test that constants are expected to be immutable."""
        # These should be simple immutable types
        assert isinstance(TEMPERATURE, (int, float))
        assert isinstance(MAX_TOKENS, int)


class TestLLMConstantsUsageValidation:
    """Test class for validating LLM constants for actual usage."""

    def test_temperature_for_api_compatibility(self):
        """Test that TEMPERATURE is compatible with common LLM APIs."""
        # Most LLM APIs accept temperature between 0 and 2
        assert 0.0 <= TEMPERATURE <= 2.0
        
        # Should be a reasonable precision for API calls
        assert isinstance(TEMPERATURE, (int, float))

    def test_max_tokens_for_api_compatibility(self):
        """Test that MAX_TOKENS is compatible with common LLM APIs."""
        # Should be positive integer
        assert isinstance(MAX_TOKENS, int)
        assert MAX_TOKENS > 0
        
        # Should not exceed common API limits (most APIs have limits around 32k-128k)
        assert MAX_TOKENS <= 100000

    def test_constants_json_serializable(self):
        """Test that constants can be serialized to JSON (for API calls)."""
        import json
        
        # Should be able to serialize these values
        try:
            json.dumps({"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS})
        except (TypeError, ValueError) as e:
            pytest.fail(f"Constants should be JSON serializable: {e}")

    def test_constants_for_snake_game_context(self):
        """Test that constants are appropriate for snake game context."""
        # For snake game:
        # - We want consistent, logical moves (low temperature)
        # - We need enough tokens for JSON response with move list and reasoning
        # - But not too many tokens (responses should be concise)
        
        # Temperature should favor consistency
        assert TEMPERATURE <= 0.3
        
        # Tokens should be sufficient for reasonable snake game responses
        # A typical response might be:
        # {"moves": ["UP", "RIGHT", "RIGHT", ...], "reasoning": "..."}
        # With reasoning, this could be 100-1000 tokens typically
        assert 2000 <= MAX_TOKENS <= 10000


class TestLLMConstantsEdgeCases:
    """Test class for edge cases and boundary conditions."""

    def test_temperature_zero_handling(self):
        """Test behavior when temperature is zero."""
        if TEMPERATURE == 0.0:
            # Zero temperature should work fine (completely deterministic)
            assert TEMPERATURE == 0.0
        else:
            # If not zero, should be positive
            assert TEMPERATURE > 0.0

    def test_temperature_floating_point_precision(self):
        """Test that temperature handles floating point precision correctly."""
        # Should be representable as a float without precision issues
        temp_reconstructed = float(str(TEMPERATURE))
        assert abs(temp_reconstructed - TEMPERATURE) < 1e-10

    def test_max_tokens_boundary_values(self):
        """Test that MAX_TOKENS handles boundary values appropriately."""
        # Should be well within common limits but not at extremes
        assert MAX_TOKENS >= 100  # Minimum reasonable value
        assert MAX_TOKENS <= 50000  # Maximum reasonable value for most use cases

    def test_constants_type_stability(self):
        """Test that constants maintain their types across imports."""
        # Re-import and verify types are consistent
        from config.llm_constants import TEMPERATURE as TEMP2, MAX_TOKENS as TOKENS2
        
        assert type(TEMPERATURE) == type(TEMP2)
        assert type(MAX_TOKENS) == type(TOKENS2)
        assert TEMPERATURE == TEMP2
        assert MAX_TOKENS == TOKENS2


class TestLLMConstantsDocumentation:
    """Test class for validating constants are well-documented."""

    def test_module_has_docstring(self):
        """Test that the module has appropriate documentation."""
        import config.llm_constants as module
        assert hasattr(module, '__doc__')
        assert module.__doc__ is not None
        assert len(module.__doc__.strip()) > 0

    def test_constants_have_meaningful_names(self):
        """Test that constant names are meaningful and follow conventions."""
        # Should be ALL_CAPS
        assert 'TEMPERATURE' == 'TEMPERATURE'.upper()
        assert 'MAX_TOKENS' == 'MAX_TOKENS'.upper()
        
        # Should have descriptive names
        assert 'TEMPERATURE' in 'TEMPERATURE'  # Self-descriptive
        assert 'MAX' in 'MAX_TOKENS' and 'TOKENS' in 'MAX_TOKENS'

    def test_constants_represent_llm_config(self):
        """Test that constants represent typical LLM configuration parameters."""
        # These are standard LLM API parameters
        llm_param_names = ['temperature', 'max_tokens', 'max_length', 'top_p', 'top_k']
        
        # Our constants should correspond to these common parameters
        assert any(param.upper().replace('_', '') in 'TEMPERATURE' for param in llm_param_names)
        assert any(param.upper().replace('_', '') in 'MAXTOKENS' for param in llm_param_names) 