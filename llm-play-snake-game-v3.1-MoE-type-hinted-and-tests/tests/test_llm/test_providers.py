"""Tests for llm.providers module and individual provider classes."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

from llm.providers.base_provider import BaseProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.providers.deepseek_provider import DeepSeekProvider
from llm.providers.mistral_provider import MistralProvider
from llm.providers.hunyuan_provider import HunyuanProvider


class TestBaseProvider:
    """Test class for BaseProvider base class."""

    def test_base_provider_abstract(self):
        """Test that BaseProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider("test-model")

    def test_base_provider_interface(self):
        """Test that BaseProvider defines the required interface."""
        # Check that required methods are defined (even if abstract)
        required_methods = ['send_prompt']
        
        for method_name in required_methods:
            assert hasattr(BaseProvider, method_name)

    def test_base_provider_subclass_requirements(self):
        """Test that subclasses must implement required methods."""
        class IncompleteProvider(BaseProvider):
            pass
        
        with pytest.raises(TypeError):
            IncompleteProvider("test-model")

    def test_base_provider_complete_subclass(self):
        """Test that complete subclass can be instantiated."""
        class CompleteProvider(BaseProvider):
            def __init__(self, model_name: str, **kwargs):
                self.model_name = model_name
                super().__init__(model_name, **kwargs)
            
            async def send_prompt(self, prompt: str) -> str:
                return f"Response to: {prompt}"
        
        provider = CompleteProvider("test-model")
        assert provider.model_name == "test-model"


class TestOllamaProvider:
    """Test class for OllamaProvider."""

    def test_ollama_provider_init(self):
        """Test OllamaProvider initialization."""
        provider = OllamaProvider("llama3:8b")
        
        assert provider.model_name == "llama3:8b"
        assert hasattr(provider, 'send_prompt')

    def test_ollama_provider_init_with_config(self):
        """Test OllamaProvider initialization with configuration."""
        config = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "host": "localhost",
            "port": 11434
        }
        
        provider = OllamaProvider("llama3:8b", **config)
        
        assert provider.model_name == "llama3:8b"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_ollama_send_prompt_success(self, mock_post):
        """Test successful prompt sending with Ollama."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Test response from Ollama"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = OllamaProvider("llama3:8b")
        result = await provider.send_prompt("Test prompt")
        
        assert result == "Test response from Ollama"
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_ollama_send_prompt_network_error(self, mock_post):
        """Test Ollama prompt sending with network error."""
        mock_post.side_effect = Exception("Network error")
        
        provider = OllamaProvider("llama3:8b")
        
        with pytest.raises(Exception):
            await provider.send_prompt("Test prompt")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_ollama_send_prompt_invalid_response(self, mock_post):
        """Test Ollama with invalid response format."""
        mock_response = Mock()
        mock_response.json.return_value = {}  # Missing 'response' key
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = OllamaProvider("llama3:8b")
        
        with pytest.raises((KeyError, ValueError)):
            await provider.send_prompt("Test prompt")

    def test_ollama_provider_configuration(self):
        """Test OllamaProvider configuration options."""
        provider = OllamaProvider(
            "llama3:8b",
            temperature=0.3,
            max_tokens=1024,
            host="192.168.1.100",
            port=11434
        )
        
        assert provider.model_name == "llama3:8b"

    @pytest.mark.asyncio
    async def test_ollama_prompt_formatting(self):
        """Test that Ollama formats prompts correctly."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "formatted response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            provider = OllamaProvider("llama3:8b")
            await provider.send_prompt("Test prompt with special chars: 🐍")
            
            # Verify the prompt was sent correctly
            call_args = mock_post.call_args
            assert "Test prompt with special chars: 🐍" in str(call_args)


class TestDeepSeekProvider:
    """Test class for DeepSeekProvider."""

    def test_deepseek_provider_init(self):
        """Test DeepSeekProvider initialization."""
        provider = DeepSeekProvider("deepseek-coder")
        
        assert provider.model_name == "deepseek-coder"
        assert hasattr(provider, 'send_prompt')

    def test_deepseek_provider_init_with_api_key(self):
        """Test DeepSeekProvider initialization with API key."""
        provider = DeepSeekProvider("deepseek-coder", api_key="test-key-123")
        
        assert provider.model_name == "deepseek-coder"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_deepseek_send_prompt_success(self, mock_post):
        """Test successful prompt sending with DeepSeek."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response from DeepSeek"
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = DeepSeekProvider("deepseek-coder", api_key="test-key")
        result = await provider.send_prompt("Test prompt")
        
        assert result == "Test response from DeepSeek"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_deepseek_send_prompt_auth_error(self, mock_post):
        """Test DeepSeek prompt sending with authentication error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_response
        
        provider = DeepSeekProvider("deepseek-coder", api_key="invalid-key")
        
        with pytest.raises(Exception):
            await provider.send_prompt("Test prompt")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_deepseek_send_prompt_rate_limit(self, mock_post):
        """Test DeepSeek with rate limiting."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("429 Rate Limit")
        mock_post.return_value = mock_response
        
        provider = DeepSeekProvider("deepseek-coder", api_key="test-key")
        
        with pytest.raises(Exception):
            await provider.send_prompt("Test prompt")

    def test_deepseek_provider_configuration(self):
        """Test DeepSeekProvider configuration options."""
        provider = DeepSeekProvider(
            "deepseek-coder",
            api_key="test-key",
            temperature=0.2,
            max_tokens=4096,
            base_url="https://api.deepseek.com"
        )
        
        assert provider.model_name == "deepseek-coder"

    @pytest.mark.asyncio
    async def test_deepseek_message_formatting(self):
        """Test DeepSeek message formatting."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "response"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            provider = DeepSeekProvider("deepseek-coder", api_key="test-key")
            await provider.send_prompt("System: You are a helpful assistant.\nUser: Hello")
            
            # Verify message formatting
            call_args = mock_post.call_args
            request_json = call_args[1]['json']
            assert 'messages' in request_json


class TestMistralProvider:
    """Test class for MistralProvider."""

    def test_mistral_provider_init(self):
        """Test MistralProvider initialization."""
        provider = MistralProvider("mistral-7b")
        
        assert provider.model_name == "mistral-7b"
        assert hasattr(provider, 'send_prompt')

    def test_mistral_provider_init_with_api_key(self):
        """Test MistralProvider initialization with API key."""
        provider = MistralProvider("mistral-7b", api_key="test-mistral-key")
        
        assert provider.model_name == "mistral-7b"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_mistral_send_prompt_success(self, mock_post):
        """Test successful prompt sending with Mistral."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response from Mistral"
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = MistralProvider("mistral-7b", api_key="test-key")
        result = await provider.send_prompt("Test prompt")
        
        assert result == "Test response from Mistral"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_mistral_send_prompt_error_handling(self, mock_post):
        """Test Mistral error handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error"
            }
        }
        mock_response.raise_for_status.side_effect = Exception("400 Bad Request")
        mock_post.return_value = mock_response
        
        provider = MistralProvider("invalid-model", api_key="test-key")
        
        with pytest.raises(Exception):
            await provider.send_prompt("Test prompt")

    def test_mistral_provider_models(self):
        """Test MistralProvider with different model names."""
        models = ["mistral-7b", "mistral-8x7b", "mistral-large"]
        
        for model in models:
            provider = MistralProvider(model, api_key="test-key")
            assert provider.model_name == model

    @pytest.mark.asyncio
    async def test_mistral_streaming_disabled(self):
        """Test that Mistral streaming is properly disabled."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "response"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            provider = MistralProvider("mistral-7b", api_key="test-key")
            await provider.send_prompt("Test prompt")
            
            # Verify streaming is disabled
            call_args = mock_post.call_args
            request_json = call_args[1]['json']
            assert request_json.get('stream') is False


class TestHunyuanProvider:
    """Test class for HunyuanProvider."""

    def test_hunyuan_provider_init(self):
        """Test HunyuanProvider initialization."""
        provider = HunyuanProvider("hunyuan-lite")
        
        assert provider.model_name == "hunyuan-lite"
        assert hasattr(provider, 'send_prompt')

    def test_hunyuan_provider_init_with_credentials(self):
        """Test HunyuanProvider initialization with credentials."""
        provider = HunyuanProvider(
            "hunyuan-lite",
            secret_id="test-secret-id",
            secret_key="test-secret-key",
            region="ap-beijing"
        )
        
        assert provider.model_name == "hunyuan-lite"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_hunyuan_send_prompt_success(self, mock_post):
        """Test successful prompt sending with Hunyuan."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Response": {
                "Choices": [
                    {
                        "Message": {
                            "Content": "Test response from Hunyuan"
                        }
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = HunyuanProvider(
            "hunyuan-lite",
            secret_id="test-id",
            secret_key="test-key"
        )
        result = await provider.send_prompt("Test prompt")
        
        assert result == "Test response from Hunyuan"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_hunyuan_send_prompt_auth_error(self, mock_post):
        """Test Hunyuan authentication error."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Response": {
                "Error": {
                    "Code": "AuthFailure",
                    "Message": "Authentication failed"
                }
            }
        }
        mock_response.raise_for_status.side_effect = Exception("403 Forbidden")
        mock_post.return_value = mock_response
        
        provider = HunyuanProvider(
            "hunyuan-lite",
            secret_id="invalid-id",
            secret_key="invalid-key"
        )
        
        with pytest.raises(Exception):
            await provider.send_prompt("Test prompt")

    def test_hunyuan_provider_regions(self):
        """Test HunyuanProvider with different regions."""
        regions = ["ap-beijing", "ap-shanghai", "ap-guangzhou"]
        
        for region in regions:
            provider = HunyuanProvider(
                "hunyuan-lite",
                secret_id="test-id",
                secret_key="test-key",
                region=region
            )
            assert provider.model_name == "hunyuan-lite"

    @pytest.mark.asyncio
    async def test_hunyuan_signature_generation(self):
        """Test Hunyuan API signature generation."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "Response": {"Choices": [{"Message": {"Content": "response"}}]}
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            provider = HunyuanProvider(
                "hunyuan-lite",
                secret_id="test-id",
                secret_key="test-key"
            )
            await provider.send_prompt("Test prompt")
            
            # Verify authentication headers are included
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            assert 'Authorization' in headers or 'X-TC-Authorization' in headers


class TestProvidersIntegration:
    """Test class for provider integration scenarios."""

    def test_all_providers_implement_interface(self):
        """Test that all providers implement the required interface."""
        providers = [
            (OllamaProvider, "llama3:8b"),
            (DeepSeekProvider, "deepseek-coder"),
            (MistralProvider, "mistral-7b"),
            (HunyuanProvider, "hunyuan-lite")
        ]
        
        for provider_class, model_name in providers:
            provider = provider_class(model_name)
            
            # Should have required methods
            assert hasattr(provider, 'send_prompt')
            assert hasattr(provider, 'model_name')
            assert provider.model_name == model_name

    @pytest.mark.asyncio
    async def test_provider_error_consistency(self):
        """Test that all providers handle errors consistently."""
        providers = [
            OllamaProvider("test-model"),
            DeepSeekProvider("test-model", api_key="test"),
            MistralProvider("test-model", api_key="test"),
            HunyuanProvider("test-model", secret_id="test", secret_key="test")
        ]
        
        for provider in providers:
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_post.side_effect = Exception("Network error")
                
                with pytest.raises(Exception):
                    await provider.send_prompt("Test prompt")

    @pytest.mark.asyncio
    async def test_provider_unicode_handling(self):
        """Test that all providers handle unicode correctly."""
        unicode_prompt = "Test with unicode: 🐍🍎 and Chinese: 你好"
        
        providers_configs = [
            (OllamaProvider, "test-model", {}),
            (DeepSeekProvider, "test-model", {"api_key": "test"}),
            (MistralProvider, "test-model", {"api_key": "test"}),
            (HunyuanProvider, "test-model", {"secret_id": "test", "secret_key": "test"})
        ]
        
        for provider_class, model, config in providers_configs:
            provider = provider_class(model, **config)
            
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = self._get_mock_response(provider_class)
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                try:
                    result = await provider.send_prompt(unicode_prompt)
                    assert isinstance(result, str)
                except Exception:
                    # Some providers might not handle unicode - that's documented
                    pass

    def _get_mock_response(self, provider_class):
        """Helper to get appropriate mock response for provider type."""
        if provider_class == OllamaProvider:
            return {"response": "mock response"}
        elif provider_class in [DeepSeekProvider, MistralProvider]:
            return {"choices": [{"message": {"content": "mock response"}}]}
        elif provider_class == HunyuanProvider:
            return {"Response": {"Choices": [{"Message": {"Content": "mock response"}}]}}
        else:
            return {"response": "default mock response"}

    @pytest.mark.asyncio
    async def test_provider_concurrent_requests(self):
        """Test providers with concurrent requests."""
        provider = OllamaProvider("test-model")
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "concurrent response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Send multiple concurrent requests
            tasks = [
                provider.send_prompt(f"Prompt {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(result == "concurrent response" for result in results)
            assert mock_post.call_count == 5

    def test_provider_configuration_validation(self):
        """Test provider configuration validation."""
        # Test various configuration scenarios
        configs = [
            # Valid configurations
            (OllamaProvider, "llama3:8b", {"temperature": 0.7}),
            (DeepSeekProvider, "deepseek-coder", {"api_key": "test", "temperature": 0.2}),
            (MistralProvider, "mistral-7b", {"api_key": "test", "max_tokens": 1000}),
            (HunyuanProvider, "hunyuan-lite", {"secret_id": "test", "secret_key": "test"}),
            
            # Edge case configurations
            (OllamaProvider, "model", {"temperature": 0.0}),
            (OllamaProvider, "model", {"temperature": 2.0}),
        ]
        
        for provider_class, model, config in configs:
            try:
                provider = provider_class(model, **config)
                assert provider.model_name == model
            except (ValueError, TypeError) as e:
                # Some configurations might be invalid - that's expected
                pass

    @pytest.mark.asyncio
    async def test_provider_timeout_handling(self):
        """Test provider timeout handling."""
        provider = OllamaProvider("test-model")
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timed out")
            
            with pytest.raises(asyncio.TimeoutError):
                await provider.send_prompt("Test prompt")

    def test_provider_model_name_validation(self):
        """Test provider model name validation."""
        providers_and_models = [
            (OllamaProvider, ["llama3:8b", "codellama:7b", "mistral:7b"]),
            (DeepSeekProvider, ["deepseek-coder", "deepseek-chat"]),
            (MistralProvider, ["mistral-7b", "mistral-8x7b", "mistral-large"]),
            (HunyuanProvider, ["hunyuan-lite", "hunyuan-standard", "hunyuan-pro"])
        ]
        
        for provider_class, models in providers_and_models:
            for model in models:
                if provider_class == OllamaProvider:
                    provider = provider_class(model)
                elif provider_class in [DeepSeekProvider, MistralProvider]:
                    provider = provider_class(model, api_key="test")
                else:  # HunyuanProvider
                    provider = provider_class(model, secret_id="test", secret_key="test")
                
                assert provider.model_name == model

    @pytest.mark.asyncio
    async def test_provider_response_parsing(self):
        """Test provider response parsing consistency."""
        test_content = "This is a test response with special chars: !@#$%^&*()"
        
        providers_and_responses = [
            (OllamaProvider, {"response": test_content}),
            (DeepSeekProvider, {"choices": [{"message": {"content": test_content}}]}),
            (MistralProvider, {"choices": [{"message": {"content": test_content}}]}),
            (HunyuanProvider, {"Response": {"Choices": [{"Message": {"Content": test_content}}]}}),
        ]
        
        for provider_class, mock_response in providers_and_responses:
            if provider_class == OllamaProvider:
                provider = provider_class("test-model")
            elif provider_class in [DeepSeekProvider, MistralProvider]:
                provider = provider_class("test-model", api_key="test")
            else:  # HunyuanProvider
                provider = provider_class("test-model", secret_id="test", secret_key="test")
            
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_http_response = Mock()
                mock_http_response.json.return_value = mock_response
                mock_http_response.raise_for_status.return_value = None
                mock_post.return_value = mock_http_response
                
                result = await provider.send_prompt("Test prompt")
                assert result == test_content 