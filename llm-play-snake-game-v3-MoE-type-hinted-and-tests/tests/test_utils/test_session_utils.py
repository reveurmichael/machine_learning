"""
Tests for utils.session_utils module.

Focuses on testing session management utilities for session lifecycle,
state persistence, session validation, and session recovery.
"""

import pytest
import tempfile
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import uuid

from utils.session_utils import SessionUtils


class TestSessionUtils:
    """Test session utility functions."""

    def test_session_creation_and_initialization(self) -> None:
        """Test creation and initialization of new game sessions."""
        
        session_utils: SessionUtils = SessionUtils()
        
        # Test session creation scenarios
        session_creation_configs: List[Dict[str, Any]] = [
            {
                "session_type": "single_player",
                "player_config": {
                    "player_id": "player_123",
                    "player_name": "TestPlayer",
                    "skill_level": "beginner"
                },
                "game_config": {
                    "grid_size": 10,
                    "max_games": 5,
                    "llm_provider": "deepseek"
                },
                "expected_success": True
            },
            {
                "session_type": "tournament",
                "player_config": {
                    "player_id": "player_456",
                    "player_name": "ProPlayer",
                    "skill_level": "expert"
                },
                "game_config": {
                    "grid_size": 15,
                    "max_games": 10,
                    "llm_provider": "mistral",
                    "tournament_mode": True
                },
                "expected_success": True
            },
            {
                "session_type": "benchmark",
                "player_config": {
                    "player_id": "benchmark_789",
                    "player_name": "BenchmarkBot",
                    "skill_level": "ai"
                },
                "game_config": {
                    "grid_size": 12,
                    "max_games": 100,
                    "llm_provider": "hunyuan",
                    "benchmark_suite": "standard"
                },
                "expected_success": True
            }
        ]
        
        session_creation_results: List[Dict[str, Any]] = []
        
        for config in session_creation_configs:
            session_type = config["session_type"]
            player_config = config["player_config"]
            game_config = config["game_config"]
            expected_success = config["expected_success"]
            
            # Create session
            creation_result = session_utils.create_session(
                session_type=session_type,
                player_config=player_config,
                game_config=game_config
            )
            
            actual_success = creation_result["success"]
            assert actual_success == expected_success, f"Session creation outcome mismatch for {session_type}: expected {expected_success}, got {actual_success}"
            
            if actual_success:
                session_data = creation_result["session"]
                
                # Verify session structure
                assert "session_id" in session_data, f"Session ID missing for {session_type}"
                assert "session_type" in session_data, f"Session type missing for {session_type}"
                assert "created_at" in session_data, f"Creation timestamp missing for {session_type}"
                assert "player_config" in session_data, f"Player config missing for {session_type}"
                assert "game_config" in session_data, f"Game config missing for {session_type}"
                assert "status" in session_data, f"Session status missing for {session_type}"
                
                # Verify session ID format (should be UUID-like)
                session_id = session_data["session_id"]
                assert len(session_id) > 10, f"Session ID too short for {session_type}"
                
                # Verify status is 'active' for new sessions
                assert session_data["status"] == "active", f"New session should be active for {session_type}"
                
                # Verify configuration preservation
                assert session_data["session_type"] == session_type, f"Session type mismatch for {session_type}"
                assert session_data["player_config"]["player_id"] == player_config["player_id"], f"Player ID mismatch for {session_type}"
            
            session_creation_results.append({
                "session_type": session_type,
                "success": actual_success,
                "result": creation_result
            })
        
        # Verify creation results
        successful_creations = [r for r in session_creation_results if r["success"]]
        assert len(successful_creations) == 3, "Should create 3 valid sessions"
