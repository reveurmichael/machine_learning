"""Tests for llm.setup_utils module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from llm.setup_utils import check_env_setup


class TestCheckEnvSetup:
    """Test class for check_env_setup function."""

    @patch('os.environ.get')
    def test_check_env_setup_ollama_default(self, mock_get):
        """Test ollama provider with default settings."""
        mock_get.return_value = None  # No OLLAMA_HOST set
        
        result = check_env_setup("ollama")
        
        assert result is True  # Ollama should work without env vars

    @patch('os.environ.get')
    def test_check_env_setup_ollama_custom_host(self, mock_get):
        """Test ollama provider with custom host."""
        mock_get.return_value = "192.168.1.100:11434"
        
        result = check_env_setup("ollama")
        
        assert result is True

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_no_env_file(self, mock_get, mock_exists):
        """Test behavior when .env file doesn't exist."""
        mock_exists.return_value = False
        mock_get.return_value = None
        
        result = check_env_setup("deepseek")
        
        assert result is False

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_deepseek_success(self, mock_get, mock_exists):
        """Test successful deepseek setup."""
        mock_exists.return_value = True
        mock_get.return_value = "test-deepseek-key"
        
        result = check_env_setup("deepseek")
        
        assert result is True

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_hunyuan_success(self, mock_get, mock_exists):
        """Test successful hunyuan setup."""
        mock_exists.return_value = True
        mock_get.return_value = "test-hunyuan-key"
        
        result = check_env_setup("hunyuan")
        
        assert result is True

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_mistral_success(self, mock_get, mock_exists):
        """Test successful mistral setup."""
        mock_exists.return_value = True
        mock_get.return_value = "test-mistral-key"
        
        result = check_env_setup("mistral")
        
        assert result is True

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_missing_key(self, mock_get, mock_exists):
        """Test behavior when API key is missing."""
        mock_exists.return_value = True
        mock_get.return_value = None  # No API key
        
        result = check_env_setup("deepseek")
        
        assert result is False

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_case_insensitive(self, mock_get, mock_exists):
        """Test that provider names are case insensitive."""
        mock_exists.return_value = True
        mock_get.return_value = "test-key"
        
        providers = ["DEEPSEEK", "deepseek", "DeepSeek"]
        
        for provider in providers:
            result = check_env_setup(provider)
            assert result is True

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_unknown_provider(self, mock_get, mock_exists):
        """Test behavior with unknown provider."""
        mock_exists.return_value = True
        mock_get.return_value = None
        
        result = check_env_setup("unknown_provider")
        
        assert result is False

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_empty_provider(self, mock_get, mock_exists):
        """Test behavior with empty provider string."""
        mock_exists.return_value = True
        mock_get.return_value = None
        
        result = check_env_setup("")
        
        assert result is False

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_none_provider(self, mock_get, mock_exists):
        """Test behavior with None provider."""
        mock_exists.return_value = True
        mock_get.return_value = None
        
        with pytest.raises(AttributeError):
            check_env_setup(None)

    @patch('builtins.print')
    @patch('os.environ.get')
    def test_check_env_setup_ollama_prints_message(self, mock_get, mock_print):
        """Test that ollama setup prints appropriate messages."""
        mock_get.return_value = None
        
        check_env_setup("ollama")
        
        # Should print a warning about default host
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("default Ollama host" in call for call in print_calls)

    @patch('builtins.print')
    @patch('os.environ.get')
    def test_check_env_setup_ollama_custom_host_message(self, mock_get, mock_print):
        """Test that ollama with custom host prints appropriate message."""
        mock_get.return_value = "custom-host:11434"
        
        check_env_setup("ollama")
        
        # Should print message about custom host
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("custom Ollama host" in call for call in print_calls)

    @patch('builtins.print')
    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_missing_env_file_message(self, mock_get, mock_exists, mock_print):
        """Test that missing .env file prints appropriate message."""
        mock_exists.return_value = False
        mock_get.return_value = None
        
        check_env_setup("deepseek")
        
        # Should print error about missing .env file
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any(".env file" in call for call in print_calls)

    @patch('builtins.print')
    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_success_message(self, mock_get, mock_exists, mock_print):
        """Test that successful setup prints success message."""
        mock_exists.return_value = True
        mock_get.return_value = "test-key"
        
        check_env_setup("deepseek")
        
        # Should print success message
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("DEEPSEEK_API_KEY found" in call for call in print_calls)

    @patch('builtins.print')
    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_check_env_setup_missing_key_message(self, mock_get, mock_exists, mock_print):
        """Test that missing API key prints warning message."""
        mock_exists.return_value = True
        mock_get.return_value = None
        
        check_env_setup("mistral")
        
        # Should print warning about missing key
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("No API key found for mistral" in call for call in print_calls)


class TestSetupUtilsIntegration:
    """Test class for integration scenarios."""

    @patch.dict('os.environ', {}, clear=True)
    @patch('os.path.exists')
    def test_realistic_env_setup_scenarios(self, mock_exists):
        """Test realistic environment setup scenarios."""
        # Scenario 1: Fresh installation, no .env file
        mock_exists.return_value = False
        assert check_env_setup("deepseek") is False
        
        # Scenario 2: .env file exists but no keys
        mock_exists.return_value = True
        assert check_env_setup("hunyuan") is False
        
        # Scenario 3: Ollama should always work
        assert check_env_setup("ollama") is True

    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'test-deepseek',
        'HUNYUAN_API_KEY': 'test-hunyuan',
        'MISTRAL_API_KEY': 'test-mistral',
        'OLLAMA_HOST': 'localhost:11434'
    }, clear=True)
    @patch('os.path.exists')
    def test_full_env_setup(self, mock_exists):
        """Test with full environment setup."""
        mock_exists.return_value = True
        
        providers = ["deepseek", "hunyuan", "mistral", "ollama"]
        
        for provider in providers:
            result = check_env_setup(provider)
            assert result is True, f"Provider {provider} should be properly configured"

    def test_provider_name_variations(self):
        """Test various provider name formats."""
        with patch('os.path.exists', return_value=True):
            with patch('os.environ.get') as mock_get:
                mock_get.return_value = "test-key"
                
                # Test case variations
                variations = [
                    ("deepseek", "DEEPSEEK"),
                    ("HUNYUAN", "hunyuan"),
                    ("Mistral", "MISTRAL"),
                    ("ollama", "OLLAMA")
                ]
                
                for lower, upper in variations:
                    if lower != "ollama":  # ollama doesn't need API key
                        result_lower = check_env_setup(lower)
                        result_upper = check_env_setup(upper)
                        assert result_lower == result_upper

    @patch('os.path.abspath')
    @patch('os.path.dirname')
    def test_env_file_path_resolution(self, mock_dirname, mock_abspath):
        """Test that .env file path is resolved correctly."""
        mock_abspath.return_value = "/path/to/project/llm/setup_utils.py"
        mock_dirname.side_effect = [
            "/path/to/project/llm",  # First dirname call
            "/path/to/project"       # Second dirname call
        ]
        
        with patch('os.path.exists') as mock_exists:
            with patch('os.environ.get', return_value=None):
                mock_exists.return_value = False
                
                check_env_setup("deepseek")
                
                # Verify that path was constructed correctly
                expected_path = "/path/to/project/.env"
                mock_exists.assert_called_with(expected_path)

    @patch('builtins.print')
    def test_colorama_usage(self, mock_print):
        """Test that colorama colors are used in output."""
        with patch('os.environ.get', return_value=None):
            # Test ollama (should use YELLOW for warning)
            check_env_setup("ollama")
            
            # Test deepseek without .env file (should use RED for error)
            with patch('os.path.exists', return_value=False):
                check_env_setup("deepseek")
            
            # Verify that colored output was used
            mock_print.assert_called()

    def test_multiple_consecutive_calls(self):
        """Test multiple consecutive calls to check_env_setup."""
        with patch('os.environ.get', return_value="test-key"):
            with patch('os.path.exists', return_value=True):
                # Multiple calls should be consistent
                results = []
                for _ in range(5):
                    result = check_env_setup("deepseek")
                    results.append(result)
                
                # All results should be the same
                assert all(r == results[0] for r in results)
                assert results[0] is True

    def test_edge_case_env_values(self):
        """Test edge cases for environment variable values."""
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace
            "   test-key   ",  # Key with whitespace
            "test-key-with-special-chars!@#$%",  # Special characters
        ]
        
        with patch('os.path.exists', return_value=True):
            for env_value in edge_cases:
                with patch('os.environ.get', return_value=env_value):
                    result = check_env_setup("deepseek")
                    
                    if env_value.strip():  # Non-empty after stripping
                        assert result is True
                    else:  # Empty or whitespace-only
                        assert result is False 