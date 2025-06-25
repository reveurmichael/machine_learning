"""Tests for config.ui_constants module."""

import pytest

from config.ui_constants import (
    COLORS,
    GRID_SIZE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    TIME_DELAY,
    TIME_TICK,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_PARSER_PROVIDER,
    DEFAULT_PARSER_MODEL,
)


class TestColors:
    """Test class for color configuration."""

    def test_colors_is_dict(self):
        """Test that COLORS is a dictionary."""
        assert isinstance(COLORS, dict)

    def test_colors_not_empty(self):
        """Test that COLORS is not empty."""
        assert len(COLORS) > 0

    def test_color_values_are_rgb_tuples(self):
        """Test that all color values are RGB tuples."""
        for color_name, color_value in COLORS.items():
            assert isinstance(color_value, tuple), f"Color {color_name} should be a tuple"
            assert len(color_value) == 3, f"Color {color_name} should have 3 RGB components"
            
            for component in color_value:
                assert isinstance(component, int), f"RGB component should be integer for {color_name}"
                assert 0 <= component <= 255, f"RGB component should be 0-255 for {color_name}"

    def test_expected_color_names(self):
        """Test that expected color names are present."""
        expected_colors = {
            "SNAKE_HEAD",
            "SNAKE_BODY", 
            "APPLE",
            "BACKGROUND",
            "GRID",
            "TEXT",
            "ERROR",
            "BLACK",
            "WHITE",
            "GREY",
            "APP_BG",
        }
        
        assert set(COLORS.keys()) == expected_colors

    def test_color_names_are_strings(self):
        """Test that all color names are strings."""
        for color_name in COLORS.keys():
            assert isinstance(color_name, str)
            assert len(color_name) > 0

    def test_color_names_uppercase(self):
        """Test that color names follow uppercase convention."""
        for color_name in COLORS.keys():
            assert color_name.isupper(), f"Color name {color_name} should be uppercase"

    def test_basic_colors_defined(self):
        """Test that basic colors (BLACK, WHITE) are correctly defined."""
        assert COLORS["BLACK"] == (0, 0, 0)
        assert COLORS["WHITE"] == (255, 255, 255)

    def test_game_specific_colors_defined(self):
        """Test that game-specific colors are defined."""
        game_colors = ["SNAKE_HEAD", "SNAKE_BODY", "APPLE", "BACKGROUND", "GRID"]
        for game_color in game_colors:
            assert game_color in COLORS
            # Should be valid RGB tuples (already tested above)

    def test_ui_colors_defined(self):
        """Test that UI-specific colors are defined."""
        ui_colors = ["TEXT", "ERROR", "APP_BG", "GREY"]
        for ui_color in ui_colors:
            assert ui_color in COLORS

    def test_colors_visually_distinct(self):
        """Test that key colors are visually distinct."""
        # Snake head and body should be different
        assert COLORS["SNAKE_HEAD"] != COLORS["SNAKE_BODY"]
        
        # Apple should be distinct from snake
        assert COLORS["APPLE"] != COLORS["SNAKE_HEAD"]
        assert COLORS["APPLE"] != COLORS["SNAKE_BODY"]
        
        # Background should be distinct from foreground elements
        assert COLORS["BACKGROUND"] != COLORS["SNAKE_HEAD"]
        assert COLORS["BACKGROUND"] != COLORS["APPLE"]
        
        # Text should be distinct from background
        assert COLORS["TEXT"] != COLORS["BACKGROUND"]


class TestDimensions:
    """Test class for dimension constants."""

    def test_grid_size_type_and_value(self):
        """Test that GRID_SIZE is a positive integer."""
        assert isinstance(GRID_SIZE, int)
        assert GRID_SIZE > 0

    def test_grid_size_reasonable_value(self):
        """Test that GRID_SIZE has a reasonable value for snake game."""
        # Snake games typically use grids between 10x10 and 30x30
        assert 5 <= GRID_SIZE <= 50

    def test_window_dimensions_type_and_value(self):
        """Test that window dimensions are positive integers."""
        assert isinstance(WINDOW_WIDTH, int)
        assert isinstance(WINDOW_HEIGHT, int)
        assert WINDOW_WIDTH > 0
        assert WINDOW_HEIGHT > 0

    def test_window_dimensions_reasonable_values(self):
        """Test that window dimensions are reasonable for desktop applications."""
        # Should be large enough to be usable but not too large
        assert 400 <= WINDOW_WIDTH <= 2000
        assert 300 <= WINDOW_HEIGHT <= 1500

    def test_window_aspect_ratio(self):
        """Test that window has a reasonable aspect ratio."""
        aspect_ratio = WINDOW_WIDTH / WINDOW_HEIGHT
        # Should be between 1:2 and 3:1 (reasonable for game windows)
        assert 0.5 <= aspect_ratio <= 3.0


class TestTimingConstants:
    """Test class for timing-related constants."""

    def test_time_delay_type_and_value(self):
        """Test that TIME_DELAY is a positive number."""
        assert isinstance(TIME_DELAY, (int, float))
        assert TIME_DELAY > 0

    def test_time_tick_type_and_value(self):
        """Test that TIME_TICK is a positive number."""
        assert isinstance(TIME_TICK, (int, float))
        assert TIME_TICK > 0

    def test_timing_values_reasonable(self):
        """Test that timing values are reasonable for game performance."""
        # TIME_DELAY should be small (milliseconds for responsiveness)
        assert 1 <= TIME_DELAY <= 1000
        
        # TIME_TICK should be reasonable for game loop timing
        assert 10 <= TIME_TICK <= 5000

    def test_timing_relationship(self):
        """Test that timing constants have logical relationship."""
        # TIME_TICK is typically larger than TIME_DELAY
        # (game tick vs general delay)
        assert TIME_TICK >= TIME_DELAY


class TestDefaultProviderConstants:
    """Test class for default provider configuration."""

    def test_default_provider_type(self):
        """Test that DEFAULT_PROVIDER is a string."""
        assert isinstance(DEFAULT_PROVIDER, str)
        assert len(DEFAULT_PROVIDER) > 0

    def test_default_model_type(self):
        """Test that DEFAULT_MODEL is a string."""
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

    def test_default_parser_provider_type(self):
        """Test that DEFAULT_PARSER_PROVIDER is a string."""
        assert isinstance(DEFAULT_PARSER_PROVIDER, str)
        assert len(DEFAULT_PARSER_PROVIDER) > 0

    def test_default_parser_model_type(self):
        """Test that DEFAULT_PARSER_MODEL is a string."""
        assert isinstance(DEFAULT_PARSER_MODEL, str)
        assert len(DEFAULT_PARSER_MODEL) > 0

    def test_provider_names_format(self):
        """Test that provider names follow expected format."""
        providers = [DEFAULT_PROVIDER, DEFAULT_PARSER_PROVIDER]
        for provider in providers:
            # Should be lowercase (common convention)
            assert provider.islower(), f"Provider {provider} should be lowercase"
            # Should not contain spaces
            assert " " not in provider, f"Provider {provider} should not contain spaces"

    def test_model_names_format(self):
        """Test that model names follow expected format."""
        models = [DEFAULT_MODEL, DEFAULT_PARSER_MODEL]
        for model in models:
            # Should not be empty
            assert len(model.strip()) > 0
            # Common model name patterns (letters, numbers, hyphens, colons)
            import re
            pattern = re.compile(r'^[a-zA-Z0-9\-:._]+$')
            assert pattern.match(model), f"Model name {model} should follow naming conventions"

    def test_default_values_consistency(self):
        """Test that default values are consistent."""
        # Primary and parser providers could be the same (that's valid)
        # Models should be different if they serve different purposes
        # (but this is not strictly required)
        
        # At minimum, they should all be valid strings
        defaults = [DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_PARSER_PROVIDER, DEFAULT_PARSER_MODEL]
        for default in defaults:
            assert isinstance(default, str)
            assert len(default.strip()) > 0


class TestUIConstantsIntegration:
    """Test class for integration and cross-validation of UI constants."""

    def test_colors_compatible_with_pygame(self):
        """Test that colors are compatible with pygame color format."""
        # Pygame expects RGB tuples with values 0-255
        for color_name, (r, g, b) in COLORS.items():
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
            assert isinstance(r, int)
            assert isinstance(g, int)
            assert isinstance(b, int)

    def test_dimensions_compatible_with_grid(self):
        """Test that window dimensions work well with grid size."""
        # Window should be large enough to display the grid
        min_cell_size = 10  # Minimum reasonable cell size in pixels
        min_window_width = GRID_SIZE * min_cell_size
        min_window_height = GRID_SIZE * min_cell_size
        
        assert WINDOW_WIDTH >= min_window_width
        assert WINDOW_HEIGHT >= min_window_height

    def test_timing_constants_for_smooth_gameplay(self):
        """Test that timing constants support smooth gameplay."""
        # Frame rate calculation (approximate)
        if TIME_TICK > 0:
            approx_fps = 1000 / TIME_TICK  # Assuming TIME_TICK is in milliseconds
            # Should support reasonable frame rates (10-60 FPS)
            assert 5 <= approx_fps <= 100

    def test_default_providers_realistic(self):
        """Test that default providers are realistic choices."""
        common_providers = ["openai", "ollama", "anthropic", "cohere", "huggingface", "mistral", "deepseek"]
        
        # DEFAULT_PROVIDER should be a known provider type
        assert any(provider in DEFAULT_PROVIDER.lower() for provider in common_providers)
        
        # DEFAULT_PARSER_PROVIDER should also be reasonable
        assert any(provider in DEFAULT_PARSER_PROVIDER.lower() for provider in common_providers)

    def test_constants_for_streamlit_compatibility(self):
        """Test that default constants are noted as Streamlit-specific."""
        # Based on the comment in the module, these are for Streamlit only
        # They should be reasonable for web interface defaults
        
        # Provider names should be web-compatible
        assert " " not in DEFAULT_PROVIDER
        assert " " not in DEFAULT_PARSER_PROVIDER
        
        # Model names should be reasonable for web display
        assert len(DEFAULT_MODEL) <= 100  # Not too long for UI
        assert len(DEFAULT_PARSER_MODEL) <= 100


class TestUIConstantsEdgeCases:
    """Test class for edge cases and boundary conditions."""

    def test_color_tuple_immutability(self):
        """Test that color tuples are immutable."""
        for color_name in COLORS:
            color_tuple = COLORS[color_name]
            assert isinstance(color_tuple, tuple)
            # Tuples are immutable, which is good for constants

    def test_zero_values_handling(self):
        """Test handling of zero values where appropriate."""
        # TIME_DELAY and TIME_TICK should not be zero (would cause issues)
        assert TIME_DELAY != 0
        assert TIME_TICK != 0
        
        # GRID_SIZE should not be zero
        assert GRID_SIZE != 0
        
        # Window dimensions should not be zero
        assert WINDOW_WIDTH != 0
        assert WINDOW_HEIGHT != 0

    def test_negative_values_not_present(self):
        """Test that negative values are not present where inappropriate."""
        assert GRID_SIZE > 0
        assert WINDOW_WIDTH > 0
        assert WINDOW_HEIGHT > 0
        assert TIME_DELAY > 0
        assert TIME_TICK > 0

    def test_extreme_values_not_present(self):
        """Test that extreme values are avoided."""
        # Colors should not be extreme unless intentional (like pure black/white)
        non_extreme_colors = [name for name in COLORS if name not in ["BLACK", "WHITE"]]
        for color_name in non_extreme_colors:
            r, g, b = COLORS[color_name]
            # Should not be too dark (all components near 0) unless it's background
            if color_name != "BACKGROUND":
                assert not (r < 50 and g < 50 and b < 50), f"Color {color_name} might be too dark"

    def test_string_constants_not_empty(self):
        """Test that string constants are not empty or whitespace-only."""
        string_constants = [DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_PARSER_PROVIDER, DEFAULT_PARSER_MODEL]
        for constant in string_constants:
            assert len(constant.strip()) > 0
            assert constant != ""


class TestUIConstantsDocumentation:
    """Test class for validating UI constants documentation."""

    def test_module_has_docstring(self):
        """Test that the module has appropriate documentation."""
        import config.ui_constants as module
        assert hasattr(module, '__doc__')
        assert module.__doc__ is not None
        assert len(module.__doc__.strip()) > 0

    def test_constants_grouped_logically(self):
        """Test that constants are logically grouped (implied by naming)."""
        # Color constants should have 'COLOR' or color names
        color_related = [name for name in COLORS.keys()]
        assert len(color_related) > 0
        
        # Dimension constants should be about size/dimensions
        dimension_names = ["GRID_SIZE", "WINDOW_WIDTH", "WINDOW_HEIGHT"]
        for dim_name in dimension_names:
            assert any(keyword in dim_name for keyword in ["SIZE", "WIDTH", "HEIGHT", "DIMENSION"])
        
        # Timing constants should be about time
        timing_names = ["TIME_DELAY", "TIME_TICK"]
        for time_name in timing_names:
            assert "TIME" in time_name
        
        # Default constants should have 'DEFAULT'
        default_names = ["DEFAULT_PROVIDER", "DEFAULT_MODEL", "DEFAULT_PARSER_PROVIDER", "DEFAULT_PARSER_MODEL"]
        for default_name in default_names:
            assert "DEFAULT" in default_name 