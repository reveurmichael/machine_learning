#!/usr/bin/env python3
"""
Test script for canonical factory pattern implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def test_canonical_factory():
    """Test the canonical factory pattern implementation."""
    try:
        from agents import create, get_available_algorithms, get_algorithm_info
        
        print("✅ Testing canonical factory pattern...")
        
        # Test available algorithms
        algorithms = get_available_algorithms()
        print(f"📋 Available algorithms: {algorithms}")
        
        # Test creating BFS agent
        bfs_agent = create("BFS")
        print(f"🏭 Created BFS agent: {type(bfs_agent).__name__}")
        
        # Test creating BFS-SAFE-GREEDY agent
        safe_greedy_agent = create("BFS-SAFE-GREEDY")
        print(f"🏭 Created BFS-SAFE-GREEDY agent: {type(safe_greedy_agent).__name__}")
        
        # Test algorithm info
        bfs_info = get_algorithm_info("BFS")
        print(f"ℹ️  BFS info: {bfs_info['description']}")
        
        safe_greedy_info = get_algorithm_info("BFS-SAFE-GREEDY")
        print(f"ℹ️  BFS-SAFE-GREEDY info: {safe_greedy_info['description']}")
        
        print("✅ All tests passed! Canonical factory pattern working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_canonical_factory()
    sys.exit(0 if success else 1) 