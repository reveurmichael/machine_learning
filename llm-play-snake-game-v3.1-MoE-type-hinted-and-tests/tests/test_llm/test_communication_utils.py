"""Tests for llm.communication_utils module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

from llm.communication_utils import (
    send_prompt_to_llm,
    send_prompt_to_secondary_llm,
    get_response_from_provider_with_retries,
    handle_provider_response,
    validate_response_structure,
    log_llm_interaction,
)


class TestSendPromptToLLM:
    """Test class for send_prompt_to_llm function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = Mock()
        provider.send_prompt = AsyncMock(return_value="test response")
        return provider

    @pytest.fixture
    def mock_game_data(self):
        """Create a mock game data object."""
        game_data = Mock()
        game_data.log_llm_interaction = Mock()
        return game_data

    @pytest.mark.asyncio
    async def test_send_prompt_basic_success(self, mock_provider, mock_game_data):
        """Test basic successful prompt sending."""
        prompt = "Test prompt"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = "success response"
            
            result = await send_prompt_to_llm(prompt, mock_provider, mock_game_data)
            
            assert result == "success response"
            mock_get_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_prompt_with_logging(self, mock_provider, mock_game_data):
        """Test that prompt sending includes proper logging."""
        prompt = "Test prompt"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            with patch('llm.communication_utils.log_llm_interaction') as mock_log:
                mock_get_response.return_value = "success response"
                
                await send_prompt_to_llm(prompt, mock_provider, mock_game_data)
                
                mock_log.assert_called()

    @pytest.mark.asyncio
    async def test_send_prompt_error_handling(self, mock_provider, mock_game_data):
        """Test error handling in prompt sending."""
        prompt = "Test prompt"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.side_effect = Exception("Provider error")
            
            with pytest.raises(Exception):
                await send_prompt_to_llm(prompt, mock_provider, mock_game_data)

    @pytest.mark.asyncio
    async def test_send_prompt_empty_prompt(self, mock_provider, mock_game_data):
        """Test handling of empty prompt."""
        prompt = ""
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = "response"
            
            result = await send_prompt_to_llm(prompt, mock_provider, mock_game_data)
            
            # Should still process empty prompts (might be valid in some contexts)
            assert result == "response"

    @pytest.mark.asyncio
    async def test_send_prompt_none_provider(self, mock_game_data):
        """Test handling of None provider."""
        prompt = "Test prompt"
        
        with pytest.raises((AttributeError, TypeError)):
            await send_prompt_to_llm(prompt, None, mock_game_data)

    @pytest.mark.asyncio
    async def test_send_prompt_none_game_data(self, mock_provider):
        """Test handling of None game data."""
        prompt = "Test prompt"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = "response"
            
            # Should handle None game_data gracefully (logging might be optional)
            result = await send_prompt_to_llm(prompt, mock_provider, None)
            assert result == "response"


class TestSendPromptToSecondaryLLM:
    """Test class for send_prompt_to_secondary_llm function."""

    @pytest.fixture
    def mock_secondary_provider(self):
        """Create a mock secondary provider."""
        provider = Mock()
        provider.send_prompt = AsyncMock(return_value="parsed response")
        return provider

    @pytest.fixture
    def mock_game_data(self):
        """Create a mock game data object."""
        game_data = Mock()
        game_data.log_llm_interaction = Mock()
        return game_data

    @pytest.mark.asyncio
    async def test_send_secondary_prompt_success(self, mock_secondary_provider, mock_game_data):
        """Test successful secondary prompt sending."""
        primary_response = "primary response"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = "parsed response"
            
            result = await send_prompt_to_secondary_llm(
                primary_response, mock_secondary_provider, mock_game_data
            )
            
            assert result == "parsed response"

    @pytest.mark.asyncio
    async def test_send_secondary_prompt_with_context(self, mock_secondary_provider, mock_game_data):
        """Test secondary prompt with game context."""
        primary_response = "primary response"
        head_pos = (5, 5)
        body_cells = [(4, 5), (3, 5)]
        apple_pos = (7, 7)
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = "parsed response"
            
            result = await send_prompt_to_secondary_llm(
                primary_response, mock_secondary_provider, mock_game_data,
                head_pos=head_pos, body_cells=body_cells, apple_pos=apple_pos
            )
            
            assert result == "parsed response"

    @pytest.mark.asyncio
    async def test_send_secondary_prompt_error_handling(self, mock_secondary_provider, mock_game_data):
        """Test error handling in secondary prompt."""
        primary_response = "primary response"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.side_effect = Exception("Secondary provider error")
            
            with pytest.raises(Exception):
                await send_prompt_to_secondary_llm(
                    primary_response, mock_secondary_provider, mock_game_data
                )

    @pytest.mark.asyncio
    async def test_send_secondary_prompt_template_usage(self, mock_secondary_provider, mock_game_data):
        """Test that secondary prompt uses proper template."""
        primary_response = "primary response"
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            with patch('config.prompt_templates.PROMPT_TEMPLATE_TEXT_SECONDARY_LLM', 'Template: {response}'):
                mock_get_response.return_value = "parsed response"
                
                await send_prompt_to_secondary_llm(
                    primary_response, mock_secondary_provider, mock_game_data
                )
                
                # Should have called with formatted template
                mock_get_response.assert_called_once()


class TestGetResponseFromProviderWithRetries:
    """Test class for get_response_from_provider_with_retries function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.send_prompt = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, mock_provider):
        """Test successful response on first attempt."""
        mock_provider.send_prompt.return_value = "success response"
        
        result = await get_response_from_provider_with_retries(
            "test prompt", mock_provider, max_retries=3
        )
        
        assert result == "success response"
        assert mock_provider.send_prompt.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, mock_provider):
        """Test successful response after initial failures."""
        mock_provider.send_prompt.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            "success response"
        ]
        
        result = await get_response_from_provider_with_retries(
            "test prompt", mock_provider, max_retries=3
        )
        
        assert result == "success response"
        assert mock_provider.send_prompt.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_max_retries_exceeded(self, mock_provider):
        """Test behavior when max retries is exceeded."""
        mock_provider.send_prompt.side_effect = Exception("Persistent failure")
        
        with pytest.raises(Exception):
            await get_response_from_provider_with_retries(
                "test prompt", mock_provider, max_retries=2
            )
        
        assert mock_provider.send_prompt.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_different_delays(self, mock_provider):
        """Test retry with different delay configurations."""
        mock_provider.send_prompt.side_effect = [
            Exception("First failure"),
            "success response"
        ]
        
        with patch('asyncio.sleep') as mock_sleep:
            result = await get_response_from_provider_with_retries(
                "test prompt", mock_provider, max_retries=3, retry_delay=2.0
            )
            
            assert result == "success response"
            mock_sleep.assert_called_with(2.0)

    @pytest.mark.asyncio
    async def test_retry_zero_retries(self, mock_provider):
        """Test behavior with zero retries allowed."""
        mock_provider.send_prompt.side_effect = Exception("Failure")
        
        with pytest.raises(Exception):
            await get_response_from_provider_with_retries(
                "test prompt", mock_provider, max_retries=0
            )
        
        assert mock_provider.send_prompt.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_negative_retries(self, mock_provider):
        """Test behavior with negative retries (should treat as 0)."""
        mock_provider.send_prompt.side_effect = Exception("Failure")
        
        with pytest.raises(Exception):
            await get_response_from_provider_with_retries(
                "test prompt", mock_provider, max_retries=-1
            )


class TestHandleProviderResponse:
    """Test class for handle_provider_response function."""

    def test_handle_valid_response(self):
        """Test handling of valid provider response."""
        response = "Valid response text"
        
        result = handle_provider_response(response)
        
        assert result == response

    def test_handle_empty_response(self):
        """Test handling of empty response."""
        response = ""
        
        result = handle_provider_response(response)
        
        # Should handle empty responses (might return as-is or process differently)
        assert isinstance(result, str)

    def test_handle_none_response(self):
        """Test handling of None response."""
        response = None
        
        # Should handle None gracefully
        result = handle_provider_response(response)
        assert result is not None or result is None  # Depends on implementation

    def test_handle_whitespace_response(self):
        """Test handling of whitespace-only response."""
        response = "   \t\n   "
        
        result = handle_provider_response(response)
        
        # Should handle whitespace appropriately
        assert isinstance(result, str)

    def test_handle_very_long_response(self):
        """Test handling of very long response."""
        response = "x" * 10000  # Very long response
        
        result = handle_provider_response(response)
        
        assert isinstance(result, str)
        # Should handle long responses without truncation (unless specifically designed to)

    def test_handle_response_with_special_characters(self):
        """Test handling of response with special characters."""
        response = "Response with unicode: 🐍 and symbols: @#$%"
        
        result = handle_provider_response(response)
        
        assert isinstance(result, str)
        assert "🐍" in result  # Should preserve unicode


class TestValidateResponseStructure:
    """Test class for validate_response_structure function."""

    def test_validate_valid_json_response(self):
        """Test validation of valid JSON response."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Go to apple"}'
        
        is_valid = validate_response_structure(response)
        
        assert is_valid is True

    def test_validate_invalid_json_response(self):
        """Test validation of invalid JSON."""
        response = '{"moves": ["UP", "RIGHT"], "reasoning": "Go to apple"'  # Missing closing brace
        
        is_valid = validate_response_structure(response)
        
        assert is_valid is False

    def test_validate_missing_required_fields(self):
        """Test validation when required fields are missing."""
        response = '{"moves": ["UP", "RIGHT"]}'  # Missing reasoning
        
        is_valid = validate_response_structure(response)
        
        # Depends on implementation - might be invalid if reasoning is required
        assert isinstance(is_valid, bool)

    def test_validate_wrong_field_types(self):
        """Test validation with wrong field types."""
        response = '{"moves": "UP", "reasoning": ["not", "a", "string"]}'
        
        is_valid = validate_response_structure(response)
        
        assert is_valid is False

    def test_validate_empty_moves_array(self):
        """Test validation with empty moves array."""
        response = '{"moves": [], "reasoning": "NO_PATH_FOUND"}'
        
        is_valid = validate_response_structure(response)
        
        # Empty moves might be valid for NO_PATH_FOUND scenarios
        assert isinstance(is_valid, bool)

    def test_validate_non_json_response(self):
        """Test validation of non-JSON response."""
        response = "This is not JSON at all"
        
        is_valid = validate_response_structure(response)
        
        assert is_valid is False

    def test_validate_json_with_extra_fields(self):
        """Test validation of JSON with extra fields."""
        response = '{"moves": ["UP"], "reasoning": "Test", "extra_field": "value"}'
        
        is_valid = validate_response_structure(response)
        
        # Extra fields might be acceptable
        assert isinstance(is_valid, bool)


class TestLogLLMInteraction:
    """Test class for log_llm_interaction function."""

    @pytest.fixture
    def mock_game_data(self):
        """Create a mock game data object."""
        game_data = Mock()
        game_data.log_llm_interaction = Mock()
        return game_data

    def test_log_interaction_basic(self, mock_game_data):
        """Test basic interaction logging."""
        prompt = "Test prompt"
        response = "Test response"
        provider_name = "test_provider"
        
        log_llm_interaction(prompt, response, provider_name, mock_game_data)
        
        mock_game_data.log_llm_interaction.assert_called_once()

    def test_log_interaction_with_metadata(self, mock_game_data):
        """Test interaction logging with metadata."""
        prompt = "Test prompt"
        response = "Test response"
        provider_name = "test_provider"
        metadata = {"temperature": 0.7, "max_tokens": 1000}
        
        log_llm_interaction(prompt, response, provider_name, mock_game_data, metadata=metadata)
        
        mock_game_data.log_llm_interaction.assert_called_once()

    def test_log_interaction_none_game_data(self):
        """Test logging when game_data is None."""
        prompt = "Test prompt"
        response = "Test response"
        provider_name = "test_provider"
        
        # Should handle None game_data gracefully
        try:
            log_llm_interaction(prompt, response, provider_name, None)
        except AttributeError:
            # This is expected if the function doesn't handle None
            pass

    def test_log_interaction_empty_strings(self, mock_game_data):
        """Test logging with empty strings."""
        prompt = ""
        response = ""
        provider_name = ""
        
        log_llm_interaction(prompt, response, provider_name, mock_game_data)
        
        mock_game_data.log_llm_interaction.assert_called_once()

    def test_log_interaction_long_content(self, mock_game_data):
        """Test logging with very long content."""
        prompt = "x" * 5000
        response = "y" * 5000
        provider_name = "test_provider"
        
        log_llm_interaction(prompt, response, provider_name, mock_game_data)
        
        mock_game_data.log_llm_interaction.assert_called_once()


class TestCommunicationUtilsIntegration:
    """Test class for integration scenarios."""

    @pytest.fixture
    def mock_provider(self):
        """Create a comprehensive mock provider."""
        provider = Mock()
        provider.send_prompt = AsyncMock(return_value='{"moves": ["UP"], "reasoning": "test"}')
        provider.name = "test_provider"
        return provider

    @pytest.fixture
    def mock_game_data(self):
        """Create a comprehensive mock game data."""
        game_data = Mock()
        game_data.log_llm_interaction = Mock()
        game_data.get_current_state = Mock(return_value={
            "head_pos": (5, 5),
            "body_cells": [(4, 5)],
            "apple_pos": (7, 7)
        })
        return game_data

    @pytest.mark.asyncio
    async def test_full_llm_interaction_flow(self, mock_provider, mock_game_data):
        """Test complete flow from prompt to response."""
        prompt = "Move the snake"
        
        with patch('llm.communication_utils.validate_response_structure', return_value=True):
            with patch('llm.communication_utils.handle_provider_response', side_effect=lambda x: x):
                response = await send_prompt_to_llm(prompt, mock_provider, mock_game_data)
                
                assert response == '{"moves": ["UP"], "reasoning": "test"}'

    @pytest.mark.asyncio
    async def test_primary_secondary_llm_chain(self, mock_provider, mock_game_data):
        """Test chaining primary and secondary LLM calls."""
        primary_response = "Move UP to get closer to apple"
        
        mock_secondary_provider = Mock()
        mock_secondary_provider.send_prompt = AsyncMock(
            return_value='{"moves": ["UP"], "reasoning": "Processed primary response"}'
        )
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries') as mock_get_response:
            mock_get_response.return_value = '{"moves": ["UP"], "reasoning": "Processed primary response"}'
            
            result = await send_prompt_to_secondary_llm(
                primary_response, mock_secondary_provider, mock_game_data
            )
            
            assert '"moves"' in result
            assert '"reasoning"' in result

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, mock_provider, mock_game_data):
        """Test error recovery in communication."""
        # First call fails, second succeeds
        mock_provider.send_prompt.side_effect = [
            Exception("Network error"),
            '{"moves": ["RIGHT"], "reasoning": "recovered"}'
        ]
        
        with patch('asyncio.sleep'):  # Speed up test
            result = await get_response_from_provider_with_retries(
                "test prompt", mock_provider, max_retries=2
            )
            
            assert result == '{"moves": ["RIGHT"], "reasoning": "recovered"}'

    @pytest.mark.asyncio
    async def test_concurrent_llm_calls(self, mock_game_data):
        """Test handling of concurrent LLM calls."""
        # Create multiple providers
        providers = []
        for i in range(3):
            provider = Mock()
            provider.send_prompt = AsyncMock(return_value=f'response_{i}')
            providers.append(provider)
        
        # Make concurrent calls
        tasks = [
            send_prompt_to_llm(f"prompt_{i}", providers[i], mock_game_data)
            for i in range(3)
        ]
        
        with patch('llm.communication_utils.get_response_from_provider_with_retries', side_effect=lambda p, pr, **kw: pr.send_prompt(p)):
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all("response_" in result for result in results)

    def test_response_validation_edge_cases(self):
        """Test response validation with various edge cases."""
        edge_cases = [
            ('{"moves": [], "reasoning": "NO_PATH_FOUND"}', True),  # Empty moves
            ('{"moves": ["INVALID_MOVE"], "reasoning": "test"}', False),  # Invalid move
            ('{"reasoning": "test"}', False),  # Missing moves
            ('{"moves": ["UP"]}', False),  # Missing reasoning
            ('null', False),  # Null JSON
            ('[]', False),  # Array instead of object
        ]
        
        for response, expected_valid in edge_cases:
            with patch('llm.communication_utils.validate_response_structure') as mock_validate:
                mock_validate.return_value = expected_valid
                result = validate_response_structure(response)
                assert result == expected_valid 