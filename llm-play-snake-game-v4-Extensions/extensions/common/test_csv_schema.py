"""
Test script for CSV Schema Utilities
===================================

Tests the CSV schema generation and feature extraction utilities
to ensure they work correctly with different grid sizes.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from csv_schema import (
    generate_csv_schema, 
    TabularFeatureExtractor, 
    create_csv_row,
    get_schema_info,
    validate_game_state
)


def test_schema_generation():
    """Test CSV schema generation for different grid sizes."""
    print("Testing CSV schema generation...")
    
    for grid_size in [8, 10, 12, 16, 20]:
        schema = generate_csv_schema(grid_size)
        info = get_schema_info(grid_size)
        
        print(f"Grid size {grid_size}:")
        print(f"  Total columns: {schema.total_columns}")
        print(f"  Feature columns: {len(schema.feature_columns)}")
        print(f"  Metadata columns: {len(schema.metadata_columns)}")
        print(f"  Target column: {schema.target_column}")
        
        # Verify expected structure
        expected_features = 16  # Fixed number of engineered features
        expected_metadata = 2   # game_id, step_in_game
        expected_total = expected_features + expected_metadata + 1  # +1 for target
        
        assert len(schema.feature_columns) == expected_features, f"Expected {expected_features} features"
        assert len(schema.metadata_columns) == expected_metadata, f"Expected {expected_metadata} metadata columns"
        assert schema.total_columns == expected_total, f"Expected {expected_total} total columns"
        
        print("  ✓ Schema validation passed")
        print()


def test_feature_extraction():
    """Test feature extraction for different grid sizes."""
    print("Testing feature extraction...")
    
    extractor = TabularFeatureExtractor()
    
    for grid_size in [8, 10, 12]:
        # Create a test game state
        test_state = {
            "head_position": [grid_size // 2, grid_size // 2],
            "apple_position": [grid_size - 1, grid_size - 1],
            "snake_positions": [
                [grid_size // 2, grid_size // 2],
                [grid_size // 2, grid_size // 2 + 1],
                [grid_size // 2, grid_size // 2 + 2]
            ],
            "current_direction": "UP",
            "score": 10,
            "steps": 50,
            "snake_length": 3,
            "grid_size": grid_size
        }
        
        # Extract features
        features = extractor.extract_features(test_state, grid_size)
        
        print(f"Grid size {grid_size}:")
        print(f"  Extracted {len(features)} features")
        
        # Verify key features
        assert features["head_x"] == grid_size // 2
        assert features["head_y"] == grid_size // 2
        assert features["apple_x"] == grid_size - 1
        assert features["apple_y"] == grid_size - 1
        assert features["snake_length"] == 3
        
        # Verify apple direction features
        assert features["apple_dir_right"] == 1  # Apple is to the right
        assert features["apple_dir_down"] == 1   # Apple is below
        assert features["apple_dir_left"] == 0   # Apple is not to the left
        assert features["apple_dir_up"] == 0     # Apple is not above
        
        print("  ✓ Feature extraction validation passed")
        print()


def test_csv_row_creation():
    """Test CSV row creation."""
    print("Testing CSV row creation...")
    
    for grid_size in [8, 10, 12]:
        # Create a test game state
        test_state = {
            "head_position": [1, 1],
            "apple_position": [5, 5],
            "snake_positions": [[1, 1], [1, 2]],
            "current_direction": "UP",
            "score": 5,
            "steps": 20,
            "snake_length": 2,
            "grid_size": grid_size
        }
        
        # Create CSV row
        row = create_csv_row(test_state, "RIGHT", game_id=1, step_in_game=5, grid_size=grid_size)
        
        print(f"Grid size {grid_size}:")
        print(f"  Created row with {len(row)} columns")
        
        # Verify row structure
        schema_info = get_schema_info(grid_size)
        expected_columns = schema_info["column_names"]
        
        for col in expected_columns:
            assert col in row, f"Missing column: {col}"
        
        # Verify specific values
        assert row["game_id"] == 1
        assert row["step_in_game"] == 5
        assert row["head_x"] == 1
        assert row["head_y"] == 1
        assert row["apple_x"] == 5
        assert row["apple_y"] == 5
        assert row["snake_length"] == 2
        assert row["target_move"] == "RIGHT"
        
        print("  ✓ CSV row validation passed")
        print()


def test_validation():
    """Test game state validation."""
    print("Testing game state validation...")
    
    # Valid game state
    valid_state = {
        "head_position": [5, 5],
        "apple_position": [7, 3],
        "snake_positions": [[5, 5], [5, 6]],
        "current_direction": "UP",
        "score": 10,
        "steps": 50,
        "snake_length": 2,
        "grid_size": 10
    }
    
    assert validate_game_state(valid_state, 10) == True, "Valid state should pass validation"
    print("  ✓ Valid state validation passed")
    
    # Invalid game state (head out of bounds)
    invalid_state = {
        "head_position": [15, 5],  # Out of bounds for grid size 10
        "apple_position": [7, 3],
        "snake_positions": [[15, 5], [15, 6]],
        "current_direction": "UP",
        "score": 10,
        "steps": 50,
        "snake_length": 2,
        "grid_size": 10
    }
    
    assert validate_game_state(invalid_state, 10) == False, "Invalid state should fail validation"
    print("  ✓ Invalid state validation passed")
    
    # Test with different grid sizes
    for grid_size in [8, 12, 16]:
        test_state = {
            "head_position": [grid_size // 2, grid_size // 2],
            "apple_position": [grid_size - 1, grid_size - 1],
            "snake_positions": [[grid_size // 2, grid_size // 2]],
            "current_direction": "UP",
            "score": 5,
            "steps": 20,
            "snake_length": 1,
            "grid_size": grid_size
        }
        
        assert validate_game_state(test_state, grid_size) == True, f"State should be valid for grid size {grid_size}"
    
    print("  ✓ Multi-grid validation passed")
    print()


def main():
    """Run all tests."""
    print("Running CSV Schema Utilities Tests")
    print("=" * 50)
    
    try:
        test_schema_generation()
        test_feature_extraction()
        test_csv_row_creation()
        test_validation()
        
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 