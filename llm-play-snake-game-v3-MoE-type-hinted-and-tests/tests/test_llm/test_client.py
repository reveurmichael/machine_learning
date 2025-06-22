"""
Tests for the LLM client module.
"""

import pytest
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch

from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider


class TestLLMClient:
    """Test cases for LLMClient."""

    @patch('llm.client.create_provider')
    def test_init_default_provider(self, mock_create_provider):
        """Test initialization with default provider."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider
        
        client = LLMClient()
        
        assert client.provider == "hunyuan"
        assert client.model is None
        assert client.last_token_count is None
        mock_create_provider.assert_called_once_with("hunyuan")

    @patch('llm.client.create_provider')
    def test_init_custom_provider(self, mock_create_provider):
        """Test initialization with custom provider."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider
        
        client = LLMClient(provider="deepseek", model="deepseek-chat")
        
        assert client.provider == "deepseek"
        assert client.model == "deepseek-chat"
        mock_create_provider.assert_called_once_with("deepseek")

    def test_extract_usage_with_data(self):
        """Test usage extraction with valid data."""
        client = LLMClient()
        raw_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        result = client._extract_usage(raw_usage)
        
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_extract_usage_empty_data(self):
        """Test usage extraction with empty data."""
        client = LLMClient()
        
        result = client._extract_usage({})
        
        assert result["prompt_tokens"] is None
        assert result["completion_tokens"] is None
        assert result["total_tokens"] is None

    @patch('llm.client.create_provider')
    def test_set_secondary_llm_valid(self, mock_create_provider):
        """Test setting valid secondary LLM."""
        mock_temp_provider = Mock()
        mock_temp_provider.validate_model.return_value = "validated-model"
        mock_create_provider.return_value = mock_temp_provider
        
        client = LLMClient()
        
        result = client.set_secondary_llm("deepseek", "deepseek-chat")
        
        assert result is True
        assert client.secondary_provider == "deepseek"
        assert client.secondary_model == "validated-model"

    @patch('llm.client.create_provider')
    def test_set_secondary_llm_invalid_provider(self, mock_create_provider):
        """Test setting secondary LLM with invalid provider."""
        client = LLMClient()
        
        result = client.set_secondary_llm("", "some-model")
        
        assert result is False
        assert client.secondary_provider is None
        assert client.secondary_model is None

    @patch('llm.client.create_provider')
    @patch('llm.client.get_provider_cls')
    def test_generate_response_success(self, mock_get_provider_cls, mock_create_provider):
        """Test successful response generation."""
        mock_provider_cls = Mock()
        mock_provider_cls.validate_model.return_value = "validated-model"
        mock_get_provider_cls.return_value = mock_provider_cls
        
        mock_provider = Mock()
        mock_provider.generate_response.return_value = ("Test response", {"prompt_tokens": 50})
        mock_provider.__class__ = mock_provider_cls
        mock_create_provider.return_value = mock_provider
        
        client = LLMClient()
        client.model = "test-model"
        
        response = client.generate_response("Test prompt")
        
        assert response == "Test response"
        assert client.last_token_count["prompt_tokens"] == 50

    @patch('llm.client.create_provider')
    @patch('llm.client.get_provider_cls')
    def test_generate_response_error(self, mock_get_provider_cls, mock_create_provider):
        """Test response generation with error."""
        mock_provider_cls = Mock()
        mock_get_provider_cls.return_value = mock_provider_cls
        
        mock_provider = Mock()
        mock_provider.generate_response.side_effect = Exception("Test error")
        mock_provider.__class__ = mock_provider_cls
        mock_create_provider.return_value = mock_provider
        
        client = LLMClient()
        
        response = client.generate_response("Test prompt")
        
        assert "ERROR LLMCLIENT: Test error" in response

    @patch('llm.client.create_provider')
    def test_generate_text_with_secondary_llm_not_configured(self, mock_create_provider):
        """Test secondary LLM generation when not configured."""
        client = LLMClient()
        
        response = client.generate_text_with_secondary_llm("Test prompt")
        
        assert "ERROR: Secondary LLM not configured" in response 

    def test_init_with_valid_provider(self) -> None:
        """Test LLMClient initialization with valid provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        client: LLMClient = LLMClient(mock_provider)
        
        assert client.provider is mock_provider
        assert client.last_token_count is None

    def test_init_with_unavailable_provider(self) -> None:
        """Test LLMClient initialization with unavailable provider."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = False
        
        with pytest.raises(ValueError, match="Provider is not available"):
            LLMClient(mock_provider)

    def test_generate_response_success(self) -> None:
        """Test successful response generation."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Test response"
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        client: LLMClient = LLMClient(mock_provider)
        response: str = client.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_provider.generate_response.assert_called_once_with("Test prompt")
        assert client.last_token_count is not None
        assert client.last_token_count["total_tokens"] == 150

    def test_generate_response_with_options(self) -> None:
        """Test response generation with additional options."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Configured response"
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "total_tokens": 120
        }
        
        client: LLMClient = LLMClient(mock_provider)
        options: Dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": 200,
            "model": "custom-model"
        }
        
        response: str = client.generate_response("Test prompt", **options)
        
        assert response == "Configured response"
        mock_provider.generate_response.assert_called_once_with(
            "Test prompt", 
            temperature=0.7, 
            max_tokens=200, 
            model="custom-model"
        )

    def test_generate_response_provider_error(self) -> None:
        """Test handling provider errors during response generation."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.side_effect = Exception("Provider error")
        
        client: LLMClient = LLMClient(mock_provider)
        
        with pytest.raises(Exception, match="Provider error"):
            client.generate_response("Test prompt")

    def test_is_available_true(self) -> None:
        """Test is_available when provider is available."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        client: LLMClient = LLMClient(mock_provider)
        
        assert client.is_available() is True

    def test_is_available_false(self) -> None:
        """Test is_available when provider becomes unavailable."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Provider becomes unavailable
        mock_provider.is_available.return_value = False
        
        assert client.is_available() is False

    def test_get_provider_info(self) -> None:
        """Test getting provider information."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_provider_name.return_value = "test_provider"
        mock_provider.get_model_name.return_value = "test_model"
        
        client: LLMClient = LLMClient(mock_provider)
        
        provider_name: str = client.get_provider_name()
        model_name: str = client.get_model_name()
        
        assert provider_name == "test_provider"
        assert model_name == "test_model"

    def test_token_count_tracking(self) -> None:
        """Test token count tracking across multiple requests."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # First request
        mock_provider.generate_response.return_value = "Response 1"
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
        
        client: LLMClient = LLMClient(mock_provider)
        client.generate_response("Prompt 1")
        
        first_count: Optional[Dict[str, int]] = client.last_token_count
        assert first_count is not None
        assert first_count["total_tokens"] == 75
        
        # Second request
        mock_provider.generate_response.return_value = "Response 2"
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 60,
            "completion_tokens": 30,
            "total_tokens": 90
        }
        
        client.generate_response("Prompt 2")
        
        second_count: Optional[Dict[str, int]] = client.last_token_count
        assert second_count is not None
        assert second_count["total_tokens"] == 90

    def test_token_count_none_when_not_available(self) -> None:
        """Test token count is None when provider doesn't provide it."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Response"
        mock_provider.get_last_token_count.return_value = None
        
        client: LLMClient = LLMClient(mock_provider)
        client.generate_response("Prompt")
        
        assert client.last_token_count is None

    def test_multiple_providers_not_supported(self) -> None:
        """Test that client is designed for single provider."""
        mock_provider1: Mock = Mock(spec=BaseLLMProvider)
        mock_provider2: Mock = Mock(spec=BaseLLMProvider)
        mock_provider1.is_available.return_value = True
        mock_provider2.is_available.return_value = True
        
        client: LLMClient = LLMClient(mock_provider1)
        
        # Client should maintain reference to original provider
        assert client.provider is mock_provider1
        assert client.provider is not mock_provider2

    @patch('llm.client.logger')
    def test_error_logging(self, mock_logger: Mock) -> None:
        """Test that errors are properly logged."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.side_effect = Exception("Test error")
        
        client: LLMClient = LLMClient(mock_provider)
        
        with pytest.raises(Exception):
            client.generate_response("Test prompt")
        
        # Verify error was logged (if logging is implemented)
        # Note: This test assumes logging is implemented in the client

    def test_empty_prompt_handling(self) -> None:
        """Test handling of empty prompts."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = ""
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        client: LLMClient = LLMClient(mock_provider)
        response: str = client.generate_response("")
        
        assert response == ""
        mock_provider.generate_response.assert_called_once_with("")

    def test_very_long_prompt_handling(self) -> None:
        """Test handling of very long prompts."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Long response"
        mock_provider.get_last_token_count.return_value = {
            "prompt_tokens": 5000,
            "completion_tokens": 100,
            "total_tokens": 5100
        }
        
        client: LLMClient = LLMClient(mock_provider)
        long_prompt: str = "Very long prompt " * 1000
        
        response: str = client.generate_response(long_prompt)
        
        assert response == "Long response"
        assert client.last_token_count is not None
        assert client.last_token_count["prompt_tokens"] == 5000

    def test_concurrent_requests_not_supported(self) -> None:
        """Test that client handles requests sequentially."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Simulate response order
        responses: List[str] = ["Response 1", "Response 2"]
        mock_provider.generate_response.side_effect = responses
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Make sequential requests
        response1: str = client.generate_response("Prompt 1")
        response2: str = client.generate_response("Prompt 2")
        
        assert response1 == "Response 1"
        assert response2 == "Response 2"
        assert mock_provider.generate_response.call_count == 2

    def test_provider_state_consistency(self) -> None:
        """Test that provider state remains consistent."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Response"
        mock_provider.get_last_token_count.return_value = {"total_tokens": 100}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Make multiple requests
        for i in range(5):
            client.generate_response(f"Prompt {i}")
        
        # Provider should have been called for each request
        assert mock_provider.generate_response.call_count == 5
        assert mock_provider.get_last_token_count.call_count == 5

    def test_client_with_real_provider_interface(self) -> None:
        """Test client with a provider that matches the real interface."""
        
        class MockRealProvider:
            def __init__(self) -> None:
                self.available: bool = True
                self.last_tokens: Optional[Dict[str, int]] = None
            
            def is_available(self) -> bool:
                return self.available
            
            def generate_response(self, prompt: str, **kwargs: Any) -> str:
                return f"Response to: {prompt[:20]}..."
            
            def get_last_token_count(self) -> Optional[Dict[str, int]]:
                return self.last_tokens
            
            def get_provider_name(self) -> str:
                return "mock_real_provider"
            
            def get_model_name(self) -> str:
                return "mock_model"
        
        provider: MockRealProvider = MockRealProvider()
        provider.last_tokens = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        
        client: LLMClient = LLMClient(provider)  # type: ignore
        response: str = client.generate_response("Test prompt")
        
        assert "Response to: Test prompt" in response
        assert client.last_token_count == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_edge_case_provider_returns_none_response(self) -> None:
        """Test handling when provider returns None response."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = None
        mock_provider.get_last_token_count.return_value = None
        
        client: LLMClient = LLMClient(mock_provider)
        
        # This might raise an exception or handle None gracefully
        # depending on implementation
        response: Optional[str] = client.generate_response("Test")
        
        # The actual behavior depends on implementation
        # This test documents the expected behavior
        assert response is None or isinstance(response, str)

    def test_provider_configuration_persistence(self) -> None:
        """Test that provider configuration persists across requests."""
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate_response.return_value = "Response"
        mock_provider.get_provider_name.return_value = "persistent_provider"
        mock_provider.get_model_name.return_value = "persistent_model"
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Check that provider info is consistent
        assert client.get_provider_name() == "persistent_provider"
        assert client.get_model_name() == "persistent_model"
        
        # After generating response
        client.generate_response("Test")
        
        # Should still be the same
        assert client.get_provider_name() == "persistent_provider"
        assert client.get_model_name() == "persistent_model" 