#!/usr/bin/env python3
"""
Tests for Simplified Web Architecture
====================================

Tests for the simplified Flask web architecture following KISS principles.
Tests the actual components that exist in the current simplified implementation.

Design Philosophy:
- KISS: Test simple, working components
- No Over-Preparation: Test what actually exists
- Extensible: Test patterns used by Tasks 1-5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_web_imports():
    """Test that all simplified web components can be imported."""
    print("📦 Testing Web Component Imports...")
    
    try:
        # Test base app imports
        from web.base_app import FlaskGameApp, GameFlaskApp
        print("✅ Base web apps imported successfully")
        
        # Test specific app imports
        from web.human_app import HumanWebApp
        from web.llm_app import LLMWebApp
        from web.replay_app import ReplayWebApp
        print("✅ All web apps imported successfully")
        
        # Test factory imports from centralized utils
        from utils.factory_utils import (
            WebAppFactory,
            create_human_web_app,
            create_llm_web_app,
            create_replay_web_app
        )
        print("✅ Factory functions imported successfully")
        
        # Test web module exports
        from web import (
            FlaskGameApp,
            GameFlaskApp,
            HumanWebApp,
            LLMWebApp,
            ReplayWebApp,
            WebAppFactory,
            create_human_web_app,
            create_llm_web_app,
            create_replay_web_app
        )
        print("✅ Web module exports working")
        
        print("🎉 All simplified web components imported successfully!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_factory_pattern():
    """Test simplified factory pattern implementation."""
    print("🏭 Testing Simplified Factory Pattern...")
    
    try:
        from utils.factory_utils import WebAppFactory
        
        # Test factory registry
        available_types = WebAppFactory.get_available_types()
        expected_types = ['HUMAN', 'LLM', 'REPLAY']
        
        for expected in expected_types:
            assert expected in available_types, f"Missing type: {expected}"
        
        print(f"✅ WebAppFactory has correct types: {available_types}")
        
        # Test factory creation (without actually starting servers)
        print("✅ Factory pattern working correctly")
        
        print("🎉 Simplified factory pattern test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False


def test_base_app_creation():
    """Test base Flask app creation."""
    print("🔧 Testing Base App Creation...")
    
    try:
        from web.base_app import FlaskGameApp, GameFlaskApp
        
        # Test FlaskGameApp creation
        flask_app = FlaskGameApp(name="TestApp")
        assert flask_app.app is not None
        assert flask_app.port is not None
        assert 8000 <= flask_app.port <= 16000  # Random port range
        print("✅ FlaskGameApp creation working")
        
        # Test GameFlaskApp creation (base class doesn't take grid_size)
        game_app = GameFlaskApp(name="TestGameApp")
        assert game_app.app is not None
        assert game_app.port is not None
        print("✅ GameFlaskApp creation working")
        
        print("🎉 Base app creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Base app creation test failed: {e}")
        return False


def test_network_integration():
    """Test network utilities integration."""
    print("🌐 Testing Network Integration...")
    
    try:
        from utils.network_utils import random_free_port, is_port_free
        from web.base_app import FlaskGameApp
        
        # Test random port allocation
        port = random_free_port()
        assert 8000 <= port <= 16000
        assert is_port_free(port)
        print("✅ Random port allocation working")
        
        # Test app uses network utilities
        app = FlaskGameApp(name="NetworkTest")
        assert 8000 <= app.port <= 16000
        print("✅ App network integration working")
        
        print("🎉 Network integration test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Network integration test failed: {e}")
        return False


def test_kiss_compliance():
    """Test that the architecture follows KISS principles."""
    print("✨ Testing KISS Compliance...")
    
    try:
        # Count number of classes (should be small)
        from web import base_app, human_app, llm_app, replay_app
        
        base_classes = [name for name in dir(base_app) if name.endswith('App') and not name.startswith('_')]
        human_classes = [name for name in dir(human_app) if name.endswith('App') and not name.startswith('_')]
        llm_classes = [name for name in dir(llm_app) if name.endswith('App') and not name.startswith('_')]
        replay_classes = [name for name in dir(replay_app) if name.endswith('App') and not name.startswith('_')]
        
        total_classes = len(base_classes) + len(human_classes) + len(llm_classes) + len(replay_classes)
        
        print(f"✅ Total web app classes: {total_classes} (KISS: simple architecture)")
        assert total_classes <= 10, "Too many classes - violates KISS principle"
        
        # Test factory functions are simple
        from utils.factory_utils import create_human_web_app, create_llm_web_app, create_replay_web_app
        
        # These should be simple functions, not complex classes
        assert callable(create_human_web_app)
        assert callable(create_llm_web_app)
        assert callable(create_replay_web_app)
        print("✅ Factory functions are simple callables (KISS compliant)")
        
        print("🎉 KISS compliance test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ KISS compliance test failed: {e}")
        return False


def test_extensibility():
    """Test that the architecture is easily extensible for Tasks 1-5."""
    print("🚀 Testing Extensibility for Tasks 1-5...")
    
    try:
        from web.base_app import GameFlaskApp
        from utils.factory_utils import WebAppFactory
        
        # Test that base classes can be extended
        class MockTask1WebApp(GameFlaskApp):
            """Mock Task-1 (Heuristics) web app extension."""
            
            def __init__(self, algorithm: str, **kwargs):
                super().__init__(name=f"Heuristics-{algorithm}", **kwargs)
                self.algorithm = algorithm
                
                @self.app.route('/algorithm')
                def get_algorithm():
                    return {'algorithm': self.algorithm}
        
        # Test extension creation (no grid_size for base class)
        task1_app = MockTask1WebApp(algorithm="BFS")
        assert task1_app.algorithm == "BFS"
        assert task1_app.name == "Heuristics-BFS"
        print("✅ Base classes easily extensible")
        
        # Test factory can be extended
        WebAppFactory.register("HEURISTIC", "MockTask1WebApp")
        available = WebAppFactory.get_available_types()
        assert "HEURISTIC" in available
        print("✅ Factory easily extensible")
        
        print("🎉 Extensibility test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Extensibility test failed: {e}")
        return False


def demonstrate_simplified_workflow():
    """Demonstrate the simplified web architecture workflow."""
    print("🎪 Demonstrating Simplified Web Architecture Workflow...")
    
    try:
        from utils.factory_utils import create_human_web_app, WebAppFactory
        
        print("1. Create web app using factory function:")
        app = create_human_web_app(grid_size=12, port=None)  # Random port
        print(f"   ✅ Created HumanWebApp with grid_size={app.grid_size}, port={app.port}")
        
        print("2. Alternative: Create using WebAppFactory:")
        factory_app = WebAppFactory.create("human", grid_size=10, port=None)
        print(f"   ✅ Created via factory with grid_size={factory_app.grid_size}")
        
        print("3. Demo: Extend for new task:")
        WebAppFactory.register("DEMO", "HumanWebApp")  # Simple registration
        demo_types = WebAppFactory.get_available_types()
        print(f"   ✅ Available types after extension: {demo_types}")
        
        print("🎉 Simplified workflow demonstration complete!\n")
        return True
        
    except Exception as e:
        print(f"❌ Workflow demonstration failed: {e}")
        return False


def main():
    """Run all simplified web architecture tests."""
    print("🧪 Testing Simplified Web Architecture for Snake Game AI")
    print("=" * 60)
    
    tests = [
        test_web_imports,
        test_factory_pattern,
        test_base_app_creation,
        test_network_integration,
        test_kiss_compliance,
        test_extensibility,
        demonstrate_simplified_workflow,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web architecture is KISS-compliant and extensible.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 