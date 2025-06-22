"""
Tests for utils.continuation_utils module.

Focuses on testing game continuation utilities for session resumption,
state recovery, checkpoint management, and continuation validation.
"""

import pytest
import tempfile
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from numpy.typing import NDArray

from utils.continuation_utils import ContinuationUtils
from core.game_data import GameData


class TestContinuationUtils:
    """Test continuation utility functions."""

    def test_session_state_checkpoint_creation(self) -> None:
        """Test creation and validation of session state checkpoints."""
        
        continuation_utils: ContinuationUtils = ContinuationUtils()
        
        # Mock session state data
        test_session_states: List[Dict[str, Any]] = [
            {
                "session_id": "checkpoint_session_1",
                "current_game": 3,
                "total_games": 10,
                "completed_games": [
                    {
                        "game_id": "game_1",
                        "final_score": 150,
                        "total_steps": 75,
                        "completion_status": "success"
                    },
                    {
                        "game_id": "game_2", 
                        "final_score": 200,
                        "total_steps": 90,
                        "completion_status": "success"
                    }
                ],
                "session_config": {
                    "grid_size": 10,
                    "max_steps": 500,
                    "llm_provider": "deepseek"
                },
                "checkpoint_time": time.time(),
                "session_start_time": time.time() - 300
            },
            {
                "session_id": "checkpoint_session_2",
                "current_game": 1,
                "total_games": 5,
                "completed_games": [],
                "session_config": {
                    "grid_size": 8,
                    "max_steps": 300,
                    "llm_provider": "mistral"
                },
                "checkpoint_time": time.time(),
                "session_start_time": time.time() - 60
            }
        ]
        
        # Test checkpoint creation
        created_checkpoints: List[Dict[str, Any]] = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for session_state in test_session_states:
                checkpoint_result = continuation_utils.create_session_checkpoint(
                    session_state=session_state,
                    checkpoint_dir=temp_dir
                )
                
                assert checkpoint_result["success"] is True, f"Failed to create checkpoint for {session_state['session_id']}"
                
                checkpoint_file = checkpoint_result["checkpoint_file"]
                assert os.path.exists(checkpoint_file), f"Checkpoint file not created: {checkpoint_file}"
                
                # Verify checkpoint file content
                with open(checkpoint_file, 'r') as f:
                    saved_checkpoint = json.load(f)
                
                # Verify checkpoint structure
                assert saved_checkpoint["session_id"] == session_state["session_id"], "Session ID mismatch"
                assert saved_checkpoint["current_game"] == session_state["current_game"], "Current game mismatch"
                assert saved_checkpoint["total_games"] == session_state["total_games"], "Total games mismatch"
                assert len(saved_checkpoint["completed_games"]) == len(session_state["completed_games"]), "Completed games count mismatch"
                
                # Verify checkpoint metadata
                assert "checkpoint_metadata" in saved_checkpoint, "Checkpoint metadata missing"
                metadata = saved_checkpoint["checkpoint_metadata"]
                assert "creation_time" in metadata, "Creation time missing"
                assert "format_version" in metadata, "Format version missing"
                assert "checksum" in metadata, "Checksum missing"
                
                created_checkpoints.append({
                    "session_id": session_state["session_id"],
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_data": saved_checkpoint,
                    "original_state": session_state
                })
        
        assert len(created_checkpoints) == 2, "Should create checkpoints for all sessions"

    def test_session_state_recovery_and_validation(self) -> None:
        """Test recovery and validation of session states from checkpoints."""
        
        continuation_utils: ContinuationUtils = ContinuationUtils()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test checkpoint files
            test_checkpoints: List[Dict[str, Any]] = [
                {
                    "session_id": "recovery_session_1",
                    "current_game": 5,
                    "total_games": 10,
                    "completed_games": [
                        {"game_id": f"game_{i}", "final_score": 100 + i * 25, "total_steps": 50 + i * 10}
                        for i in range(1, 5)
                    ],
                    "session_config": {"grid_size": 10, "llm_provider": "deepseek"},
                    "checkpoint_metadata": {
                        "creation_time": time.time() - 100,
                        "format_version": "1.0",
                        "checksum": "valid_checksum_123"
                    }
                },
                {
                    "session_id": "recovery_session_2",
                    "current_game": 2,
                    "total_games": 3,
                    "completed_games": [
                        {"game_id": "game_1", "final_score": 300, "total_steps": 120}
                    ],
                    "session_config": {"grid_size": 12, "llm_provider": "mistral"},
                    "checkpoint_metadata": {
                        "creation_time": time.time() - 50,
                        "format_version": "1.0",
                        "checksum": "valid_checksum_456"
                    }
                }
            ]
            
            checkpoint_files: List[str] = []
            
            # Save test checkpoints
            for i, checkpoint in enumerate(test_checkpoints):
                checkpoint_file = os.path.join(temp_dir, f"checkpoint_{i + 1}.json")
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                checkpoint_files.append(checkpoint_file)
            
            # Test checkpoint recovery
            recovery_results: List[Dict[str, Any]] = []
            
            for checkpoint_file in checkpoint_files:
                recovery_result = continuation_utils.recover_session_state(
                    checkpoint_file=checkpoint_file,
                    validate_integrity=True
                )
                
                assert recovery_result["success"] is True, f"Failed to recover checkpoint from {checkpoint_file}"
                
                recovered_state = recovery_result["session_state"]
                original_checkpoint = next(c for c in test_checkpoints if c["session_id"] == recovered_state["session_id"])
                
                # Verify recovered state integrity
                assert recovered_state["session_id"] == original_checkpoint["session_id"], "Session ID mismatch"
                assert recovered_state["current_game"] == original_checkpoint["current_game"], "Current game mismatch"
                assert recovered_state["total_games"] == original_checkpoint["total_games"], "Total games mismatch"
                assert len(recovered_state["completed_games"]) == len(original_checkpoint["completed_games"]), "Completed games mismatch"
                
                # Verify recovery metadata
                assert "recovery_metadata" in recovery_result, "Recovery metadata missing"
                recovery_metadata = recovery_result["recovery_metadata"]
                assert "recovery_time" in recovery_metadata, "Recovery time missing"
                assert "validation_passed" in recovery_metadata, "Validation status missing"
                assert recovery_metadata["validation_passed"] is True, "Validation should pass"
                
                recovery_results.append({
                    "checkpoint_file": checkpoint_file,
                    "recovered_state": recovered_state,
                    "recovery_metadata": recovery_metadata
                })
            
            assert len(recovery_results) == 2, "Should recover all checkpoints"

    def test_continuation_point_identification(self) -> None:
        """Test identification of valid continuation points within game sessions."""
        
        continuation_utils: ContinuationUtils = ContinuationUtils()
        
        # Mock session with various continuation scenarios
        test_session_scenarios: List[Dict[str, Any]] = [
            {
                "session_id": "continuation_session_1",
                "scenario": "mid_session_continuation",
                "session_data": {
                    "total_games": 10,
                    "completed_games": 6,
                    "current_game_state": {
                        "game_id": "game_7",
                        "current_step": 45,
                        "max_steps": 500,
                        "snake_alive": True,
                        "game_in_progress": True
                    },
                    "session_config": {"grid_size": 10, "llm_provider": "deepseek"}
                },
                "expected_continuation_type": "resume_current_game"
            },
            {
                "session_id": "continuation_session_2", 
                "scenario": "between_games_continuation",
                "session_data": {
                    "total_games": 5,
                    "completed_games": 3,
                    "current_game_state": {
                        "game_id": "game_3",
                        "final_score": 250,
                        "completion_status": "finished",
                        "game_in_progress": False
                    },
                    "session_config": {"grid_size": 8, "llm_provider": "mistral"}
                },
                "expected_continuation_type": "start_next_game"
            },
            {
                "session_id": "continuation_session_3",
                "scenario": "session_completed",
                "session_data": {
                    "total_games": 3,
                    "completed_games": 3,
                    "current_game_state": {
                        "game_id": "game_3",
                        "final_score": 180,
                        "completion_status": "finished",
                        "game_in_progress": False
                    },
                    "session_config": {"grid_size": 12, "llm_provider": "hunyuan"}
                },
                "expected_continuation_type": "session_complete"
            }
        ]
        
        continuation_analysis_results: List[Dict[str, Any]] = []
        
        for scenario in test_session_scenarios:
            analysis_result = continuation_utils.analyze_continuation_point(
                session_data=scenario["session_data"]
            )
            
            assert analysis_result["success"] is True, f"Failed to analyze continuation for {scenario['session_id']}"
            
            continuation_type = analysis_result["continuation_type"]
            expected_type = scenario["expected_continuation_type"]
            
            assert continuation_type == expected_type, f"Continuation type mismatch for {scenario['session_id']}: expected {expected_type}, got {continuation_type}"
            
            # Verify continuation metadata
            assert "continuation_metadata" in analysis_result, "Continuation metadata missing"
            metadata = analysis_result["continuation_metadata"]
            
            if continuation_type == "resume_current_game":
                assert "current_step" in metadata, "Current step missing for game resumption"
                assert "remaining_steps" in metadata, "Remaining steps missing for game resumption"
                assert metadata["game_resumable"] is True, "Game should be resumable"
                
            elif continuation_type == "start_next_game":
                assert "next_game_number" in metadata, "Next game number missing"
                assert "remaining_games" in metadata, "Remaining games count missing"
                assert metadata["games_remaining"] > 0, "Should have remaining games"
                
            elif continuation_type == "session_complete":
                assert "completion_status" in metadata, "Completion status missing"
                assert metadata["all_games_completed"] is True, "All games should be completed"
            
            continuation_analysis_results.append({
                "session_id": scenario["session_id"],
                "scenario": scenario["scenario"],
                "continuation_type": continuation_type,
                "metadata": metadata,
                "analysis_successful": True
            })
        
        assert len(continuation_analysis_results) == 3, "Should analyze all continuation scenarios"

    def test_session_merging_and_consolidation(self) -> None:
        """Test merging and consolidation of multiple session fragments."""
        
        continuation_utils: ContinuationUtils = ContinuationUtils()
        
        # Mock fragmented session data (e.g., from interrupted sessions)
        session_fragments: List[Dict[str, Any]] = [
            {
                "fragment_id": "fragment_1",
                "session_id": "merge_session_1",
                "games_data": [
                    {"game_id": "game_1", "final_score": 120, "total_steps": 60, "timestamp": time.time() - 300},
                    {"game_id": "game_2", "final_score": 180, "total_steps": 90, "timestamp": time.time() - 250}
                ],
                "fragment_metadata": {
                    "start_time": time.time() - 300,
                    "end_time": time.time() - 200,
                    "fragment_complete": True
                }
            },
            {
                "fragment_id": "fragment_2",
                "session_id": "merge_session_1",
                "games_data": [
                    {"game_id": "game_3", "final_score": 200, "total_steps": 100, "timestamp": time.time() - 150},
                    {"game_id": "game_4", "final_score": 90, "total_steps": 45, "timestamp": time.time() - 100}
                ],
                "fragment_metadata": {
                    "start_time": time.time() - 200,
                    "end_time": time.time() - 50,
                    "fragment_complete": True
                }
            },
            {
                "fragment_id": "fragment_3",
                "session_id": "merge_session_1",
                "games_data": [
                    {"game_id": "game_5", "final_score": 250, "total_steps": 125, "timestamp": time.time() - 30}
                ],
                "fragment_metadata": {
                    "start_time": time.time() - 50,
                    "end_time": time.time() - 10,
                    "fragment_complete": False  # Potentially incomplete
                }
            }
        ]
        
        # Test session merging
        merge_result = continuation_utils.merge_session_fragments(
            fragments=session_fragments,
            session_id="merge_session_1",
            validate_continuity=True
        )
        
        assert merge_result["success"] is True, "Session merging should succeed"
        
        merged_session = merge_result["merged_session"]
        
        # Verify merged session structure
        assert merged_session["session_id"] == "merge_session_1", "Session ID should be preserved"
        assert "games_data" in merged_session, "Games data missing from merged session"
        assert "session_metadata" in merged_session, "Session metadata missing"
        
        merged_games = merged_session["games_data"]
        assert len(merged_games) == 5, "Should merge all games from fragments"
        
        # Verify games are in chronological order
        game_timestamps = [game["timestamp"] for game in merged_games]
        assert game_timestamps == sorted(game_timestamps), "Games should be in chronological order"
        
        # Verify game ID continuity
        game_ids = [game["game_id"] for game in merged_games]
        expected_ids = ["game_1", "game_2", "game_3", "game_4", "game_5"]
        assert game_ids == expected_ids, f"Game IDs should be sequential: expected {expected_ids}, got {game_ids}"
        
        # Verify session metadata
        session_metadata = merged_session["session_metadata"]
        assert "total_games" in session_metadata, "Total games count missing"
        assert "session_duration" in session_metadata, "Session duration missing"
        assert "fragment_count" in session_metadata, "Fragment count missing"
        assert "continuity_verified" in session_metadata, "Continuity verification status missing"
        
        assert session_metadata["total_games"] == 5, "Should count all merged games"
        assert session_metadata["fragment_count"] == 3, "Should count all fragments"
        assert session_metadata["continuity_verified"] is True, "Continuity should be verified"

    def test_continuation_state_validation(self) -> None:
        """Test validation of continuation states and integrity checks."""
        
        continuation_utils: ContinuationUtils = ContinuationUtils()
        
        # Test various validation scenarios
        validation_test_cases: List[Dict[str, Any]] = [
            {
                "case_name": "valid_continuation_state",
                "state_data": {
                    "session_id": "valid_session_1",
                    "current_game": 3,
                    "total_games": 10,
                    "completed_games": [
                        {"game_id": "game_1", "final_score": 150},
                        {"game_id": "game_2", "final_score": 200}
                    ],
                    "session_config": {"grid_size": 10, "llm_provider": "deepseek"},
                    "checkpoint_metadata": {
                        "creation_time": time.time(),
                        "format_version": "1.0",
                        "checksum": "valid_checksum"
                    }
                },
                "should_validate": True,
                "expected_issues": []
            },
            {
                "case_name": "missing_required_fields",
                "state_data": {
                    "session_id": "invalid_session_1",
                    "current_game": 3,
                    # Missing total_games
                    "completed_games": [
                        {"game_id": "game_1", "final_score": 150}
                    ]
                    # Missing session_config and checkpoint_metadata
                },
                "should_validate": False,
                "expected_issues": ["missing_total_games", "missing_session_config", "missing_checkpoint_metadata"]
            },
            {
                "case_name": "inconsistent_game_data",
                "state_data": {
                    "session_id": "invalid_session_2",
                    "current_game": 2,  # Inconsistent with completed games
                    "total_games": 5,
                    "completed_games": [
                        {"game_id": "game_1", "final_score": 150},
                        {"game_id": "game_2", "final_score": 200},
                        {"game_id": "game_3", "final_score": 100}  # 3 completed but current_game is 2
                    ],
                    "session_config": {"grid_size": 10, "llm_provider": "deepseek"},
                    "checkpoint_metadata": {
                        "creation_time": time.time(),
                        "format_version": "1.0",
                        "checksum": "valid_checksum"
                    }
                },
                "should_validate": False,
                "expected_issues": ["inconsistent_game_progression"]
            },
            {
                "case_name": "corrupted_checksum",
                "state_data": {
                    "session_id": "invalid_session_3",
                    "current_game": 2,
                    "total_games": 5,
                    "completed_games": [
                        {"game_id": "game_1", "final_score": 150}
                    ],
                    "session_config": {"grid_size": 10, "llm_provider": "deepseek"},
                    "checkpoint_metadata": {
                        "creation_time": time.time(),
                        "format_version": "1.0",
                        "checksum": "corrupted_checksum_xyz"  # Invalid checksum
                    }
                },
                "should_validate": False,
                "expected_issues": ["checksum_validation_failed"]
            }
        ]
        
        validation_results: List[Dict[str, Any]] = []
        
        for test_case in validation_test_cases:
            validation_result = continuation_utils.validate_continuation_state(
                state_data=test_case["state_data"],
                strict_validation=True
            )
            
            case_name = test_case["case_name"]
            should_validate = test_case["should_validate"]
            expected_issues = test_case["expected_issues"]
            
            # Verify validation outcome
            validation_passed = validation_result["validation_passed"]
            assert validation_passed == should_validate, f"Validation outcome mismatch for {case_name}: expected {should_validate}, got {validation_passed}"
            
            # Verify identified issues
            if not should_validate:
                assert "validation_issues" in validation_result, f"Validation issues missing for failing case {case_name}"
                identified_issues = validation_result["validation_issues"]
                
                for expected_issue in expected_issues:
                    assert any(expected_issue in issue["issue_type"] for issue in identified_issues), f"Expected issue '{expected_issue}' not found in {case_name}"
            
            validation_results.append({
                "case_name": case_name,
                "validation_passed": validation_passed,
                "validation_result": validation_result,
                "test_successful": True
            })
        
        assert len(validation_results) == 4, "Should test all validation cases"
        
        # Verify distribution of test results
        passed_validations = [r for r in validation_results if r["validation_passed"]]
        failed_validations = [r for r in validation_results if not r["validation_passed"]]
        
        assert len(passed_validations) == 1, "Should have 1 valid case"
        assert len(failed_validations) == 3, "Should have 3 invalid cases"
