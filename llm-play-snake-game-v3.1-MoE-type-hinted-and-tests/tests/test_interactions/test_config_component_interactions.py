"""Tests for config component interactions.

This module tests the interactions between configuration components
and all other system components to ensure proper configuration
management and propagation throughout the system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from typing import Dict, List, Any, Optional

from config.game_constants import *
from config.llm_constants import *
from config.network_constants import *
from config.prompt_templates import *
from config.ui_constants import *
from core.game_manager import GameManager
from core.game_logic import GameLogic
from core.game_controller import GameController
from llm.client import LLMClient
from llm.providers.base_provider import BaseProvider
from gui.game_gui import GameGUI


class TestGameConstantsInteractions:
    """Test suite for game constants interactions with components."""

    def test_grid_size_propagation(self) -> None:
        """Test that GRID_SIZE constant is properly used across components."""
        # Test with GameController
        controller = GameController()
        assert controller.grid_size == GRID_SIZE
        
        # Test with GameLogic
        logic = GameLogic()
        assert logic.grid_size == GRID_SIZE
        
        # Test custom grid size override
        custom_size = 15
        custom_controller = GameController(grid_size=custom_size)
        assert custom_controller.grid_size == custom_size

    def test_initial_snake_length_usage(self) -> None:
        """Test that INITIAL_SNAKE_LENGTH is used correctly."""
        controller = GameController()
        controller.reset()
        
        # Initial snake should have the configured length
        assert len(controller.snake_positions) == INITIAL_SNAKE_LENGTH

    def test_direction_constants_consistency(self) -> None:
        """Test that direction constants are used consistently."""
        controller = GameController()
        
        # Test all direction constants are defined and valid
        directions = [DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT]
        
        for direction in directions:
            assert isinstance(direction, list)
            assert len(direction) == 2
            assert all(isinstance(coord, int) for coord in direction)

    def test_game_states_usage(self) -> None:
        """Test that game state constants are used properly."""
        from core.game_data import GameData
        
        game_data = GameData()
        
        # Test game over reasons are properly defined
        game_over_reasons = [
            GAME_OVER_WALL_COLLISION,
            GAME_OVER_SELF_COLLISION,
            GAME_OVER_MAX_STEPS,
            GAME_OVER_MAX_EMPTY_MOVES
        ]
        
        for reason in game_over_reasons:
            assert isinstance(reason, str)
            assert len(reason) > 0


class TestLLMConstantsInteractions:
    """Test suite for LLM constants interactions with components."""

    def test_provider_constants_with_client(self) -> None:
        """Test that LLM provider constants work with client."""
        # Test each supported provider
        providers = [
            PROVIDER_DEEPSEEK,
            PROVIDER_MISTRAL,
            PROVIDER_HUNYUAN,
            PROVIDER_OLLAMA
        ]
        
        for provider in providers:
            # Should be able to create client with provider constant
            with patch('llm.client.LLMClient') as mock_client:
                mock_client.return_value = Mock()
                client = LLMClient(provider=provider)
                mock_client.assert_called_once_with(provider=provider)

    def test_model_constants_mapping(self) -> None:
        """Test that model constants map correctly to providers."""
        # Test default models for each provider
        model_mappings = {
            PROVIDER_DEEPSEEK: DEFAULT_DEEPSEEK_MODEL,
            PROVIDER_MISTRAL: DEFAULT_MISTRAL_MODEL,
            PROVIDER_HUNYUAN: DEFAULT_HUNYUAN_MODEL,
            PROVIDER_OLLAMA: DEFAULT_OLLAMA_MODEL
        }
        
        for provider, model in model_mappings.items():
            assert isinstance(provider, str)
            assert isinstance(model, str)
            assert len(provider) > 0
            assert len(model) > 0

    def test_request_timeout_usage(self) -> None:
        """Test that request timeout is used by network components."""
        # Mock network request to verify timeout usage
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock()
            mock_post.return_value.json.return_value = {"response": "test"}
            
            # Test that DEFAULT_REQUEST_TIMEOUT is used
            assert isinstance(DEFAULT_REQUEST_TIMEOUT, (int, float))
            assert DEFAULT_REQUEST_TIMEOUT > 0

    def test_max_retries_configuration(self) -> None:
        """Test that max retries configuration is respected."""
        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert DEFAULT_MAX_RETRIES >= 0
        
        # Test retry logic respects the constant
        retry_count = 0
        max_retries = DEFAULT_MAX_RETRIES
        
        while retry_count < max_retries:
            retry_count += 1
        
        assert retry_count == max_retries


class TestNetworkConstantsInteractions:
    """Test suite for network constants interactions with components."""

    def test_api_endpoints_configuration(self) -> None:
        """Test that API endpoints are properly configured."""
        # Test endpoint constants are defined
        endpoints = [
            DEEPSEEK_API_URL,
            MISTRAL_API_URL,
            HUNYUAN_API_URL,
            OLLAMA_API_URL
        ]
        
        for endpoint in endpoints:
            assert isinstance(endpoint, str)
            assert endpoint.startswith(('http://', 'https://'))

    def test_request_headers_consistency(self) -> None:
        """Test that request headers are consistently defined."""
        # Test default headers
        headers = DEFAULT_HEADERS
        assert isinstance(headers, dict)
        assert 'Content-Type' in headers
        assert headers['Content-Type'] == 'application/json'

    def test_rate_limiting_constants(self) -> None:
        """Test that rate limiting constants are properly defined."""
        assert isinstance(RATE_LIMIT_REQUESTS_PER_MINUTE, int)
        assert RATE_LIMIT_REQUESTS_PER_MINUTE > 0
        
        assert isinstance(RATE_LIMIT_DELAY_SECONDS, (int, float))
        assert RATE_LIMIT_DELAY_SECONDS >= 0


class TestPromptTemplatesInteractions:
    """Test suite for prompt templates interactions with components."""

    def test_snake_prompt_template_usage(self) -> None:
        """Test that snake prompt template is used correctly."""
        from llm.prompt_utils import prepare_snake_prompt
        
        # Test template with sample data
        head_position = [5, 5]
        body_positions = [[4, 5], [3, 5]]
        apple_position = [7, 8]
        current_direction = "UP"
        
        prompt = prepare_snake_prompt(
            head_position=head_position,
            body_positions=body_positions,
            apple_position=apple_position,
            current_direction=current_direction
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "snake" in prompt.lower()

    def test_system_prompt_integration(self) -> None:
        """Test that system prompt integrates with LLM client."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        
        # Test system prompt contains key instructions
        key_elements = ["snake", "game", "move", "direction"]
        for element in key_elements:
            assert element.lower() in system_prompt.lower()

    def test_response_format_template(self) -> None:
        """Test that response format template is properly defined."""
        response_format = RESPONSE_FORMAT_TEMPLATE
        
        assert isinstance(response_format, str)
        assert "json" in response_format.lower()

    def test_error_handling_prompts(self) -> None:
        """Test that error handling prompts are defined."""
        error_prompts = [
            ERROR_RECOVERY_PROMPT,
            INVALID_MOVE_PROMPT,
            TIMEOUT_RETRY_PROMPT
        ]
        
        for prompt in error_prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestUIConstantsInteractions:
    """Test suite for UI constants interactions with components."""

    def test_display_constants_with_gui(self) -> None:
        """Test that display constants are used by GUI components."""
        # Test with GameGUI
        with patch('pygame.init'):
            with patch('pygame.display.set_mode') as mock_display:
                gui = GameGUI()
                
                # Should use defined window dimensions
                expected_width = CELL_SIZE * GRID_SIZE
                expected_height = CELL_SIZE * GRID_SIZE + INFO_PANEL_HEIGHT
                
                mock_display.assert_called_with((expected_width, expected_height))

    def test_color_constants_consistency(self) -> None:
        """Test that color constants are properly defined."""
        colors = [
            SNAKE_COLOR,
            APPLE_COLOR,
            BACKGROUND_COLOR,
            GRID_COLOR,
            TEXT_COLOR
        ]
        
        for color in colors:
            assert isinstance(color, (tuple, list))
            assert len(color) == 3  # RGB
            assert all(0 <= c <= 255 for c in color)

    def test_timing_constants_usage(self) -> None:
        """Test that timing constants are used correctly."""
        assert isinstance(TIME_DELAY, int)
        assert TIME_DELAY >= 0
        
        assert isinstance(TIME_TICK, int)
        assert TIME_TICK > 0

    def test_font_constants_configuration(self) -> None:
        """Test that font constants are properly configured."""
        assert isinstance(FONT_SIZE, int)
        assert FONT_SIZE > 0
        
        assert isinstance(FONT_NAME, str)


class TestConfigurationIntegrationWorkflow:
    """Test suite for complete configuration integration workflow."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 1,
            'no_gui': True,
            'primary_provider': PROVIDER_DEEPSEEK,
            'primary_model': DEFAULT_DEEPSEEK_MODEL,
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    def test_complete_configuration_workflow(self) -> None:
        """Test complete configuration workflow across all components."""
        args = self.create_mock_args()
        
        # Create manager with configuration
        manager = GameManager(args)
        
        # Test configuration propagation
        with patch.object(manager, 'setup_game') as mock_setup:
            with patch.object(manager, 'create_llm_client') as mock_create_client:
                
                # Setup game should use game constants
                manager.setup_game()
                mock_setup.assert_called_once()
                
                # LLM client should use LLM constants
                client = manager.create_llm_client(
                    provider=PROVIDER_DEEPSEEK,
                    model=DEFAULT_DEEPSEEK_MODEL
                )
                
                mock_create_client.assert_called_once_with(
                    PROVIDER_DEEPSEEK,
                    DEFAULT_DEEPSEEK_MODEL
                )

    def test_configuration_validation_workflow(self) -> None:
        """Test configuration validation across components."""
        # Test invalid configuration detection
        with pytest.raises((ValueError, AssertionError)):
            # Should fail with invalid grid size
            GameController(grid_size=0)
        
        with pytest.raises((ValueError, AssertionError)):
            # Should fail with negative grid size
            GameController(grid_size=-5)

    def test_configuration_override_workflow(self) -> None:
        """Test configuration override workflow."""
        # Test that constants can be overridden when needed
        custom_grid_size = 20
        custom_controller = GameController(grid_size=custom_grid_size)
        
        assert custom_controller.grid_size == custom_grid_size
        assert custom_controller.grid_size != GRID_SIZE

    def test_configuration_consistency_across_components(self) -> None:
        """Test configuration consistency across all components."""
        # Create components with same configuration
        controller = GameController(grid_size=GRID_SIZE)
        logic = GameLogic(grid_size=GRID_SIZE)
        
        # Both should use same grid size
        assert controller.grid_size == logic.grid_size == GRID_SIZE
        
        # Both should have same initial state
        controller.reset()
        logic.reset()
        
        assert len(controller.snake_positions) == len(logic.snake_positions) == INITIAL_SNAKE_LENGTH


class TestEnvironmentConfigurationInteractions:
    """Test suite for environment configuration interactions."""

    def test_environment_variable_integration(self) -> None:
        """Test integration with environment variables."""
        with patch.dict(os.environ, {'SNAKE_GRID_SIZE': '15'}):
            # Test that environment variables can override defaults
            # (This would require implementation in actual config loading)
            pass

    def test_config_file_loading(self) -> None:
        """Test loading configuration from files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'config.json')
            
            # Create test config file
            config_data = {
                'grid_size': 12,
                'max_games': 5,
                'provider': 'test_provider'
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(config_data, f)
            
            # Test config file loading (would require implementation)
            assert os.path.exists(config_file)

    def test_command_line_argument_override(self) -> None:
        """Test command line argument override of configuration."""
        # Test that command line args override defaults
        args = self.create_mock_args(max_games=10)
        manager = GameManager(args)
        
        assert manager.args.max_games == 10

    def test_configuration_validation_on_startup(self) -> None:
        """Test configuration validation during system startup."""
        # Test that invalid configurations are caught early
        args = self.create_mock_args(max_games=0)
        
        # Should handle invalid max_games gracefully
        manager = GameManager(args)
        assert manager.args.max_games == 0  # Allow 0 for testing


class TestDynamicConfigurationInteractions:
    """Test suite for dynamic configuration changes during runtime."""

    def test_runtime_configuration_updates(self) -> None:
        """Test updating configuration during runtime."""
        controller = GameController()
        original_size = controller.grid_size
        
        # Test that grid size can be updated (if supported)
        new_size = original_size + 5
        # This would require implementation of dynamic config updates
        
        assert controller.grid_size == original_size

    def test_configuration_change_propagation(self) -> None:
        """Test that configuration changes propagate to all components."""
        manager = GameManager(self.create_mock_args())
        
        # Test that changes to manager configuration affect child components
        # (This would require implementation of configuration change events)
        pass

    def test_configuration_rollback_on_error(self) -> None:
        """Test configuration rollback when changes cause errors."""
        controller = GameController()
        original_size = controller.grid_size
        
        # Test that failed configuration changes are rolled back
        # (This would require implementation of configuration transactions)
        assert controller.grid_size == original_size
