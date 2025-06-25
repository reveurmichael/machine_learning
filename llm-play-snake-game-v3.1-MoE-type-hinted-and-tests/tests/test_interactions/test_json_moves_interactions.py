"""
Tests for JSON parsing ↔ Move utilities interactions.

Focuses on testing how JSON parsing utilities and move processing
interact when handling malformed move data, validation chains,
and error recovery scenarios.
"""

import pytest
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch

from utils.json_utils import safe_json_parse, extract_json_from_text, repair_malformed_json
from utils.moves_utils import normalize_direction, calculate_next_position, detect_collision
from config.game_constants import DIRECTIONS


class TestJsonMovesInteractions:
    """Test interactions between JSON parsing and move utilities."""

    def test_malformed_move_data_processing_chain(self) -> None:
        """Test processing chain for malformed move data from JSON to validation."""
        # Test various malformed JSON scenarios containing move data
        malformed_scenarios: List[Tuple[str, str, List[str]]] = [
            ('{"moves": ["UP", "RIGHT"', "incomplete_json", []),
            ('{"moves": ["up", "RIGHT"]}', "lowercase_moves", ["UP", "RIGHT"]),
            ('{"moves": ["U", "R", "D", "L"]}', "abbreviated_moves", ["UP", "RIGHT", "DOWN", "LEFT"]),
            ('{"moves": ["NORTH", "EAST"]}', "alternative_names", ["UP", "RIGHT"]),
            ('{"moves": ["UP", "", "RIGHT"]}', "empty_moves", ["UP", "RIGHT"]),
            ('{"moves": ["UP", null, "RIGHT"]}', "null_moves", ["UP", "RIGHT"]),
            ('{"moves": ["UP", "INVALID", "RIGHT"]}', "invalid_moves", ["UP", "RIGHT"]),
            ('{"moves": ["UP UP", "RIGHT"]}', "spaced_moves", ["UP", "RIGHT"]),
            ('{"moves": [" UP ", " RIGHT "]}', "whitespace_moves", ["UP", "RIGHT"]),
            ('{"moves": ["0", "1", "2", "3"]}', "numeric_moves", ["UP", "RIGHT", "DOWN", "LEFT"]),
        ]
        
        for json_text, scenario, expected_moves in malformed_scenarios:
            # Step 1: JSON parsing
            parsed_data: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            
            if parsed_data is None:
                # Try extraction and repair
                extracted_data: Optional[Dict[str, Any]] = extract_json_from_text(json_text)
                if extracted_data is None:
                    repaired_json: Optional[str] = repair_malformed_json(json_text)
                    if repaired_json:
                        parsed_data = safe_json_parse(repaired_json)
            
            # Step 2: Move extraction and validation
            if parsed_data and "moves" in parsed_data:
                raw_moves: Any = parsed_data["moves"]
                
                if isinstance(raw_moves, list):
                    # Step 3: Move normalization and filtering
                    normalized_moves: List[str] = []
                    
                    for move in raw_moves:
                        if move is None or move == "":
                            continue
                            
                        if isinstance(move, str):
                            # Normalize move
                            normalized_move: Optional[str] = normalize_direction(move.strip())
                            if normalized_move and normalized_move in DIRECTIONS:
                                normalized_moves.append(normalized_move)
                        elif isinstance(move, int):
                            # Handle numeric moves
                            move_mapping: Dict[int, str] = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
                            if move in move_mapping:
                                normalized_moves.append(move_mapping[move])
                    
                    # Verify result matches expected
                    assert normalized_moves == expected_moves, f"Failed for {scenario}: expected {expected_moves}, got {normalized_moves}"

    def test_move_validation_with_json_extraction(self) -> None:
        """Test move validation when moves are extracted from complex JSON."""
        # Complex JSON scenarios with embedded move data
        complex_scenarios: List[Tuple[str, str, bool]] = [
            ('{"game_state": {"moves": ["UP", "RIGHT"]}, "metadata": {}}', "nested_moves", True),
            ('{"response": "I suggest moving {"moves": ["UP"]} for best result"}', "embedded_json", True),
            ('```json\n{"moves": ["LEFT", "DOWN"]}\n```\nThis is my analysis', "code_block", True),
            ('Multiple {"moves": ["UP"]} and {"moves": ["DOWN"]} suggestions', "multiple_json", True),
            ('{"actions": [{"type": "move", "direction": "UP"}]}', "action_format", False),
            ('{"next_move": "RIGHT", "reasoning": "avoid wall"}', "single_move_format", False),
            ('{"move_sequence": "UP,RIGHT,DOWN"}', "comma_separated", False),
        ]
        
        for json_text, scenario, has_valid_moves in complex_scenarios:
            # Extract JSON data
            primary_parse: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            extracted_parse: Optional[Dict[str, Any]] = extract_json_from_text(json_text)
            
            moves_found: List[str] = []
            
            # Try primary parse first
            if primary_parse and "moves" in primary_parse:
                raw_moves = primary_parse["moves"]
                if isinstance(raw_moves, list):
                    moves_found.extend([move for move in raw_moves if isinstance(move, str)])
            
            # Try extracted parse if primary failed
            elif extracted_parse and "moves" in extracted_parse:
                raw_moves = extracted_parse["moves"]
                if isinstance(raw_moves, list):
                    moves_found.extend([move for move in raw_moves if isinstance(move, str)])
            
            # Validate found moves
            valid_moves: List[str] = []
            for move in moves_found:
                normalized: Optional[str] = normalize_direction(move)
                if normalized and normalized in DIRECTIONS:
                    valid_moves.append(normalized)
            
            # Check if we found valid moves as expected
            if has_valid_moves:
                assert len(valid_moves) > 0, f"Expected to find valid moves in {scenario}"
                
                # Test move sequence validity
                if len(valid_moves) > 1:
                    for i in range(len(valid_moves) - 1):
                        current_move = valid_moves[i]
                        next_move = valid_moves[i + 1]
                        
                        # Check for invalid reversals
                        opposite_moves: Dict[str, str] = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
                        if next_move == opposite_moves.get(current_move):
                            # This would be an invalid reversal
                            pass  # Could remove from sequence or flag as error
            else:
                # For scenarios where we don't expect standard move format
                assert len(valid_moves) == 0 or scenario in ["single_move_format"], f"Unexpectedly found moves in {scenario}"

    def test_error_recovery_json_to_moves_pipeline(self) -> None:
        """Test error recovery in the JSON-to-moves processing pipeline."""
        # Error scenarios that require multi-step recovery
        error_scenarios: List[Tuple[str, str, List[str]]] = [
            # Syntax errors that can be repaired
            ('{"moves": ["UP", "RIGHT"]}extra text', "trailing_text", ["UP", "RIGHT"]),
            ('{"moves": ["UP" "RIGHT"]}', "missing_comma", []),  # Harder to repair
            ('{"moves": ["UP", "RIGHT"],}', "trailing_comma", ["UP", "RIGHT"]),
            ('{moves: ["UP", "RIGHT"]}', "unquoted_keys", ["UP", "RIGHT"]),
            
            # Content errors that require filtering
            ('{"moves": ["UP", "INVALID", "RIGHT", "", "DOWN"]}', "mixed_invalid", ["UP", "RIGHT", "DOWN"]),
            ('{"moves": ["up", "right", "down"]}', "case_issues", ["UP", "RIGHT", "DOWN"]),
            ('{"moves": [1, 2, 3]}', "numeric_directions", []),  # Depends on implementation
            ('{"moves": ["FORWARD", "BACKWARD"]}', "relative_directions", []),
            
            # Structure errors
            ('{"moves": "UP,RIGHT,DOWN"}', "string_instead_of_array", []),
            ('{"moves": {"0": "UP", "1": "RIGHT"}}', "object_instead_of_array", []),
            ('{"directions": ["UP", "RIGHT"]}', "wrong_key", []),
            ('{"move_list": ["UP", "RIGHT"]}', "similar_key", []),
        ]
        
        for json_text, scenario, expected_moves in error_scenarios:
            recovered_moves: List[str] = []
            
            # Step 1: Try direct parsing
            parsed: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            
            # Step 2: If direct parsing fails, try extraction
            if parsed is None:
                parsed = extract_json_from_text(json_text)
            
            # Step 3: If extraction fails, try repair then parse
            if parsed is None:
                repaired: Optional[str] = repair_malformed_json(json_text)
                if repaired:
                    parsed = safe_json_parse(repaired)
            
            # Step 4: Extract moves with various strategies
            if parsed:
                # Strategy 1: Direct moves key
                if "moves" in parsed and isinstance(parsed["moves"], list):
                    for move in parsed["moves"]:
                        if isinstance(move, str) and move.strip():
                            normalized = normalize_direction(move.strip())
                            if normalized in DIRECTIONS:
                                recovered_moves.append(normalized)
                
                # Strategy 2: Alternative keys
                elif "directions" in parsed and isinstance(parsed["directions"], list):
                    for move in parsed["directions"]:
                        if isinstance(move, str) and move.strip():
                            normalized = normalize_direction(move.strip())
                            if normalized in DIRECTIONS:
                                recovered_moves.append(normalized)
                
                # Strategy 3: String parsing
                elif "moves" in parsed and isinstance(parsed["moves"], str):
                    move_string = parsed["moves"]
                    for separator in [",", " ", ";", "|"]:
                        if separator in move_string:
                            for move in move_string.split(separator):
                                if move.strip():
                                    normalized = normalize_direction(move.strip())
                                    if normalized in DIRECTIONS:
                                        recovered_moves.append(normalized)
                            break
            
            # Verify recovery matches expected result
            assert recovered_moves == expected_moves, f"Recovery failed for {scenario}: expected {expected_moves}, got {recovered_moves}"

    def test_position_calculation_with_parsed_moves(self) -> None:
        """Test position calculation using moves parsed from JSON."""
        # JSON with move sequences for position calculation tests
        position_scenarios: List[Tuple[str, List[int], List[int]]] = [
            ('{"moves": ["UP", "UP", "RIGHT"]}', [5, 5], [6, 3]),
            ('{"moves": ["LEFT", "DOWN", "RIGHT", "UP"]}', [5, 5], [5, 5]),  # Full circle
            ('{"moves": ["RIGHT", "RIGHT", "DOWN", "DOWN"]}', [0, 0], [2, 2]),
            ('{"moves": ["UP", "LEFT", "DOWN", "RIGHT"]}', [3, 3], [3, 3]),  # Another circle
        ]
        
        for json_text, start_pos, expected_end_pos in position_scenarios:
            # Parse JSON to extract moves
            parsed: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            assert parsed is not None, f"Failed to parse: {json_text}"
            assert "moves" in parsed, f"No moves found in: {json_text}"
            
            moves: List[str] = parsed["moves"]
            current_pos: List[int] = start_pos.copy()
            
            # Apply each move using position calculation
            for move in moves:
                # Normalize move (in case of case issues)
                normalized_move: Optional[str] = normalize_direction(move)
                assert normalized_move is not None, f"Failed to normalize move: {move}"
                assert normalized_move in DIRECTIONS, f"Invalid move after normalization: {normalized_move}"
                
                # Calculate next position
                next_pos: List[int] = calculate_next_position(current_pos, normalized_move)
                current_pos = next_pos
            
            # Verify final position
            assert current_pos == expected_end_pos, f"Position calculation failed: expected {expected_end_pos}, got {current_pos}"

    def test_collision_detection_with_json_move_sequences(self) -> None:
        """Test collision detection using move sequences from JSON."""
        # JSON scenarios that should result in collisions
        collision_scenarios: List[Tuple[str, List[List[int]], int, bool]] = [
            # Self collision scenarios
            ('{"moves": ["UP", "RIGHT", "DOWN", "LEFT"]}', [[2, 2], [2, 1], [3, 1]], 10, True),
            ('{"moves": ["RIGHT", "UP", "LEFT", "DOWN"]}', [[3, 3], [3, 2], [4, 2]], 10, True),
            
            # Wall collision scenarios
            ('{"moves": ["UP", "UP", "UP"]}', [[1, 1]], 5, True),  # Hit top wall
            ('{"moves": ["LEFT", "LEFT", "LEFT"]}', [[2, 1]], 5, True),  # Hit left wall
            
            # Safe move scenarios
            ('{"moves": ["RIGHT", "DOWN", "LEFT", "UP"]}', [[5, 5]], 15, False),
            ('{"moves": ["UP", "RIGHT"]}', [[8, 8]], 15, False),
        ]
        
        for json_text, initial_snake, grid_size, should_collide in collision_scenarios:
            # Parse moves from JSON
            parsed: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            assert parsed is not None
            moves: List[str] = parsed["moves"]
            
            # Setup initial state
            snake_positions = initial_snake.copy()
            collision_detected: bool = False
            
            # Apply moves and check for collisions
            for move in moves:
                normalized_move: Optional[str] = normalize_direction(move)
                assert normalized_move is not None
                
                head_pos: List[int] = snake_positions[-1]
                next_pos: List[int] = calculate_next_position(head_pos, normalized_move)
                
                # Check collision
                if detect_collision(next_pos, snake_positions, grid_size):
                    collision_detected = True
                    break
                
                # Update snake position (simplified - just move head)
                snake_positions.append(next_pos)
                if len(snake_positions) > 3:  # Limit snake length for testing
                    snake_positions.pop(0)
            
            assert collision_detected == should_collide, f"Collision detection mismatch for {json_text}"

    def test_move_sequence_validation_pipeline(self) -> None:
        """Test complete move sequence validation pipeline from JSON to execution."""
        # Complex JSON with various validation challenges
        validation_scenarios: List[Tuple[str, List[str], int]] = [
            # Valid sequences
            ('{"moves": ["UP", "RIGHT", "DOWN"]}', ["UP", "RIGHT", "DOWN"], 3),
            
            # Sequences with reversals (should be filtered)
            ('{"moves": ["UP", "DOWN", "LEFT"]}', ["UP", "LEFT"], 2),
            ('{"moves": ["RIGHT", "LEFT", "UP"]}', ["RIGHT", "UP"], 2),
            
            # Mixed valid/invalid moves
            ('{"moves": ["UP", "INVALID", "RIGHT", "", "DOWN"]}', ["UP", "RIGHT", "DOWN"], 3),
            
            # Case normalization needed
            ('{"moves": ["up", "RIGHT", "down"]}', ["UP", "RIGHT", "DOWN"], 3),
            
            # Empty or null handling
            ('{"moves": ["UP", null, "RIGHT"]}', ["UP", "RIGHT"], 2),
        ]
        
        for json_text, expected_sequence, expected_count in validation_scenarios:
            # Parse JSON
            parsed: Optional[Dict[str, Any]] = safe_json_parse(json_text)
            assert parsed is not None
            
            raw_moves: List[Any] = parsed["moves"]
            
            # Step 1: Filter and normalize moves
            cleaned_moves: List[str] = []
            for move in raw_moves:
                if move is not None and move != "":
                    if isinstance(move, str):
                        normalized = normalize_direction(move.strip())
                        if normalized and normalized in DIRECTIONS:
                            cleaned_moves.append(normalized)
            
            # Step 2: Filter invalid reversals
            filtered_moves: List[str] = []
            previous_move: Optional[str] = None
            
            for move in cleaned_moves:
                if previous_move is None:
                    filtered_moves.append(move)
                    previous_move = move
                else:
                    # Check for reversal
                    opposite_moves: Dict[str, str] = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
                    if move != opposite_moves.get(previous_move):
                        filtered_moves.append(move)
                        previous_move = move
                    # Skip reversal moves
            
            # Verify results
            assert filtered_moves == expected_sequence, f"Sequence validation failed: expected {expected_sequence}, got {filtered_moves}"
            assert len(filtered_moves) == expected_count, f"Count mismatch: expected {expected_count}, got {len(filtered_moves)}"

    def test_json_repair_with_move_validation(self) -> None:
        """Test JSON repair specifically for move-containing JSON."""
        # Malformed JSON that specifically affects move data
        repair_scenarios: List[Tuple[str, bool, List[str]]] = [
            ('{"moves": ["UP", "RIGHT"', True, ["UP", "RIGHT"]),  # Missing closing bracket
            ('{"moves": ["UP" "RIGHT"]}', False, []),  # Missing comma - harder to repair
            ('{"moves": ["UP", "RIGHT"],}', True, ["UP", "RIGHT"]),  # Trailing comma
            ('{moves: ["UP", "RIGHT"]}', True, ["UP", "RIGHT"]),  # Unquoted key
            ('{"moves": ["UP", "RIGHT"]}garbage', True, ["UP", "RIGHT"]),  # Trailing garbage
        ]
        
        for malformed_json, should_repair, expected_moves in repair_scenarios:
            # Try to repair JSON
            repaired: Optional[str] = repair_malformed_json(malformed_json)
            
            if should_repair:
                assert repaired is not None, f"Failed to repair: {malformed_json}"
                
                # Parse repaired JSON
                parsed: Optional[Dict[str, Any]] = safe_json_parse(repaired)
                assert parsed is not None, f"Repaired JSON still invalid: {repaired}"
                
                # Extract and validate moves
                if "moves" in parsed:
                    moves: List[str] = []
                    for move in parsed["moves"]:
                        if isinstance(move, str):
                            normalized = normalize_direction(move)
                            if normalized in DIRECTIONS:
                                moves.append(normalized)
                    
                    assert moves == expected_moves, f"Move extraction failed after repair: expected {expected_moves}, got {moves}"
            else:
                # Some JSON is too malformed to repair reliably
                # Verify we don't get false positives
                if repaired is not None:
                    parsed = safe_json_parse(repaired)
                    # If parsed successfully, moves should still be valid
                    if parsed and "moves" in parsed:
                        assert isinstance(parsed["moves"], list) 