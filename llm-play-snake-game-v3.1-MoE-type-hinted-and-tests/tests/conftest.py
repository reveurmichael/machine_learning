"""
Pytest configuration and shared fixtures for the test suite.
"""

import os
import tempfile
import shutil
import json
from typing import Dict, Any, Generator, Iterator
from unittest.mock import Mock, patch
import pytest
import numpy as np
from numpy.typing import NDArray

# Add the project root to Python path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from llm.client import LLMClient


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_directory: str = tempfile.mkdtemp()
    yield temp_directory
    shutil.rmtree(temp_directory)


@pytest.fixture
def game_controller() -> GameController:
    """Create a GameController instance for testing."""
    return GameController(grid_size=10, use_gui=False)


@pytest.fixture
def game_data() -> GameData:
    """Create a GameData instance for testing."""
    return GameData()


@pytest.fixture
def game_logic() -> GameLogic:
    """Create a GameLogic instance for testing."""
    return GameLogic(grid_size=10)


@pytest.fixture
def mock_llm_client() -> Mock:
    """Create a mock LLM client for testing."""
    mock_client: Mock = Mock(spec=LLMClient)
    mock_client.generate_response.return_value = '{"moves": ["UP", "RIGHT"]}'
    mock_client.last_token_count = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    return mock_client


@pytest.fixture
def sample_game_json() -> Dict[str, Any]:
    """Sample game JSON data for testing."""
    return {
        "score": 5,
        "steps": 25,
        "snake_length": 6,
        "game_over": True,
        "game_end_reason": "collision_wall",
        "round_count": 3,
        "llm_info": {
            "primary_provider": "test_provider",
            "primary_model": "test_model",
            "parser_provider": None,
            "parser_model": None
        },
        "time_stats": {
            "start_time": "2024-01-01 12:00:00",
            "end_time": "2024-01-01 12:05:30",
            "duration": 330.0
        },
        "moves": ["UP", "UP", "RIGHT", "RIGHT", "DOWN"],
        "apple_positions": [
            {"x": 5, "y": 5},
            {"x": 7, "y": 3},
            {"x": 2, "y": 8}
        ]
    }


@pytest.fixture
def mock_env_vars() -> Iterator[None]:
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test_deepseek_key",
        "MISTRAL_API_KEY": "test_mistral_key",
        "HUNYUAN_SECRET_ID": "test_hunyuan_id",
        "HUNYUAN_SECRET_KEY": "test_hunyuan_key",
    }):
        yield


@pytest.fixture
def sample_board() -> NDArray[np.int_]:
    """Create a sample game board for testing."""
    board: NDArray[np.int_] = np.zeros((10, 10), dtype=np.int_)
    # Add snake at positions (5,5), (5,4), (5,3)
    board[5, 5] = 1  # head
    board[5, 4] = 1  # body
    board[5, 3] = 1  # tail
    # Add apple at (7, 7)
    board[7, 7] = 2
    return board


@pytest.fixture
def snake_positions() -> NDArray[np.int_]:
    """Sample snake positions for testing."""
    return np.array([[5, 5], [5, 4], [5, 3]], dtype=np.int_)


@pytest.fixture
def apple_position() -> NDArray[np.int_]:
    """Sample apple position for testing."""
    return np.array([7, 7], dtype=np.int_)


# Patch pygame imports since we don't want to initialize pygame in tests
@pytest.fixture(autouse=True)
def mock_pygame() -> Iterator[None]:
    """Mock pygame to avoid initialization issues in tests."""
    with patch.dict('sys.modules', {
        'pygame': Mock(),
        'pygame.display': Mock(),
        'pygame.font': Mock(),
        'pygame.time': Mock(),
        'pygame.event': Mock(),
        'pygame.key': Mock(),
        'pygame.locals': Mock(),
    }):
        yield 