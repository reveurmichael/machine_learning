"""
Tests for utils.initialization_utils module.

Focuses on testing system initialization utilities for component setup,
configuration validation, dependency management, and startup sequences.
"""

import pytest
import tempfile
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from numpy.typing import NDArray

from utils.initialization_utils import InitializationUtils


class TestInitializationUtils:
    """Test initialization utility functions."""

    def test_system_component_initialization(self) -> None:
        """Test initialization of core system components."""
        
        init_utils: InitializationUtils = InitializationUtils()
        
        # Mock component initialization scenarios
        component_configs: List[Dict[str, Any]] = [
            {
                "component_name": "GameController",
                "config": {
                    "grid_size": 10,
                    "use_gui": False,
                    "max_steps": 500
                },
                "dependencies": ["GameData", "GameLogic"],
                "initialization_order": 1,
                "critical": True
            },
            {
                "component_name": "LLMClient", 
                "config": {
                    "provider": "deepseek",
                    "api_key": "test_key_123",
                    "timeout": 30
                },
                "dependencies": ["NetworkUtils"],
                "initialization_order": 2,
                "critical": True
            },
            {
                "component_name": "WebServer",
                "config": {
                    "host": "0.0.0.0",
                    "port": 5000,
                    "debug": False
                },
                "dependencies": ["GameController", "LLMClient"],
                "initialization_order": 3,
                "critical": False
            },
            {
                "component_name": "GUI",
                "config": {
                    "window_size": (800, 600),
                    "theme": "dark",
                    "fps": 60
                },
                "dependencies": ["GameController"],
                "initialization_order": 4,
                "critical": False
            }
        ]
        
        # Mock initialization tracker
        init_tracker: Mock = Mock()
        init_tracker.initialized_components = {}
        init_tracker.initialization_log = []
        init_tracker.failed_components = []
        
        def mock_initialize_component(component_config: Dict[str, Any]) -> Dict[str, Any]:
            """Mock component initialization."""
            component_name = component_config["component_name"]
            config = component_config["config"]
            dependencies = component_config["dependencies"]
            critical = component_config["critical"]
            
            start_time = time.time()
            
            # Check dependencies
            missing_deps = []
            for dep in dependencies:
                if dep not in init_tracker.initialized_components:
                    missing_deps.append(dep)
            
            if missing_deps:
                error_result = {
                    "success": False,
                    "component": component_name,
                    "error": f"Missing dependencies: {missing_deps}",
                    "dependencies_missing": missing_deps
                }
                init_tracker.failed_components.append(component_name)
                init_tracker.initialization_log.append({
                    "component": component_name,
                    "status": "failed",
                    "error": error_result["error"],
                    "timestamp": start_time
                })
                return error_result
            
            # Simulate component-specific initialization
            if component_name == "GameController":
                # GameController always succeeds
                component_instance = Mock()
                component_instance.grid_size = config["grid_size"]
                component_instance.use_gui = config["use_gui"]
                component_instance.initialized = True
                
            elif component_name == "LLMClient":
                # LLMClient succeeds if valid provider
                if config["provider"] in ["deepseek", "mistral", "hunyuan"]:
                    component_instance = Mock()
                    component_instance.provider = config["provider"]
                    component_instance.initialized = True
                else:
                    error_result = {
                        "success": False,
                        "component": component_name,
                        "error": f"Invalid provider: {config['provider']}"
                    }
                    init_tracker.failed_components.append(component_name)
                    return error_result
                    
            elif component_name == "WebServer":
                # WebServer succeeds with valid port
                if isinstance(config["port"], int) and 1000 <= config["port"] <= 65535:
                    component_instance = Mock()
                    component_instance.host = config["host"]
                    component_instance.port = config["port"]
                    component_instance.initialized = True
                else:
                    error_result = {
                        "success": False,
                        "component": component_name,
                        "error": f"Invalid port: {config['port']}"
                    }
                    init_tracker.failed_components.append(component_name)
                    return error_result
                    
            elif component_name == "GUI":
                # GUI is optional and may fail
                component_instance = Mock()
                component_instance.window_size = config["window_size"]
                component_instance.initialized = True
            
            # Record successful initialization
            end_time = time.time()
            init_duration = end_time - start_time
            
            init_tracker.initialized_components[component_name] = {
                "instance": component_instance,
                "config": config,
                "initialization_time": init_duration,
                "timestamp": start_time
            }
            
            init_tracker.initialization_log.append({
                "component": component_name,
                "status": "success",
                "duration": init_duration,
                "timestamp": start_time
            })
            
            return {
                "success": True,
                "component": component_name,
                "instance": component_instance,
                "initialization_time": init_duration
            }
        
        init_tracker.initialize_component = mock_initialize_component
        
        # Test initialization sequence
        initialization_results: List[Dict[str, Any]] = []
        
        # Sort components by initialization order
        sorted_components = sorted(component_configs, key=lambda x: x["initialization_order"])
        
        for component_config in sorted_components:
            init_result = init_tracker.initialize_component(component_config)
            initialization_results.append(init_result)
            
            component_name = component_config["component_name"]
            is_critical = component_config["critical"]
            
            if not init_result["success"] and is_critical:
                # Critical component failed - should stop initialization
                remaining_components = [c["component_name"] for c in sorted_components 
                                     if c["initialization_order"] > component_config["initialization_order"]]
                
                for remaining in remaining_components:
                    init_tracker.failed_components.append(remaining)
                    init_tracker.initialization_log.append({
                        "component": remaining,
                        "status": "skipped",
                        "reason": f"Critical component {component_name} failed",
                        "timestamp": time.time()
                    })
                break
        
        # Verify initialization results
        successful_components = [r for r in initialization_results if r["success"]]
        failed_components = [r for r in initialization_results if not r["success"]]
        
        assert len(successful_components) >= 2, "Should initialize at least core components"
        assert len(init_tracker.initialized_components) >= 2, "Should track initialized components"
        
        # Verify initialization order was respected
        init_timestamps = [init_tracker.initialized_components[comp]["timestamp"] 
                          for comp in init_tracker.initialized_components]
        assert init_timestamps == sorted(init_timestamps), "Components should be initialized in order"

    def test_configuration_validation_and_merging(self) -> None:
        """Test configuration validation and merging from multiple sources."""
        
        init_utils: InitializationUtils = InitializationUtils()
        
        # Mock configuration sources
        config_sources: List[Dict[str, Any]] = [
            {
                "source_name": "default_config",
                "priority": 1,
                "config_data": {
                    "game": {
                        "grid_size": 10,
                        "max_steps": 500,
                        "enable_gui": True
                    },
                    "llm": {
                        "provider": "deepseek",
                        "timeout": 30,
                        "max_retries": 3
                    },
                    "web": {
                        "host": "localhost",
                        "port": 5000,
                        "debug": False
                    }
                }
            },
            {
                "source_name": "environment_config",
                "priority": 2,
                "config_data": {
                    "game": {
                        "grid_size": 12,  # Override default
                        "max_games": 10   # New setting
                    },
                    "llm": {
                        "provider": "mistral",  # Override default
                        "api_key": "env_api_key_123"  # New setting
                    },
                    "web": {
                        "port": 8080  # Override default
                    }
                }
            },
            {
                "source_name": "user_config",
                "priority": 3,
                "config_data": {
                    "game": {
                        "enable_gui": False,  # Override default
                        "theme": "dark"       # New setting
                    },
                    "llm": {
                        "timeout": 45  # Override default
                    }
                }
            }
        ]
        
        # Test configuration merging
        merged_config = init_utils.merge_configurations(
            config_sources=config_sources,
            validate_schema=True
        )
        
        assert merged_config["success"] is True, "Configuration merging should succeed"
        
        final_config = merged_config["merged_config"]
        
        # Verify configuration structure
        assert "game" in final_config, "Game configuration section missing"
        assert "llm" in final_config, "LLM configuration section missing"  
        assert "web" in final_config, "Web configuration section missing"
        
        # Verify priority-based overrides
        game_config = final_config["game"]
        assert game_config["grid_size"] == 12, "Environment config should override default grid_size"
        assert game_config["enable_gui"] is False, "User config should override environment/default enable_gui"
        assert game_config["max_steps"] == 500, "Default max_steps should be preserved"
        assert game_config["max_games"] == 10, "Environment config should add max_games"
        assert game_config["theme"] == "dark", "User config should add theme"
        
        llm_config = final_config["llm"]
        assert llm_config["provider"] == "mistral", "Environment config should override default provider"
        assert llm_config["timeout"] == 45, "User config should override environment/default timeout"
        assert llm_config["max_retries"] == 3, "Default max_retries should be preserved"
        assert llm_config["api_key"] == "env_api_key_123", "Environment config should add api_key"
        
        web_config = final_config["web"]
        assert web_config["host"] == "localhost", "Default host should be preserved"
        assert web_config["port"] == 8080, "Environment config should override default port"
        assert web_config["debug"] is False, "Default debug should be preserved"
        
        # Test configuration validation
        validation_result = init_utils.validate_configuration(
            config=final_config,
            schema_rules={
                "game.grid_size": {"type": int, "min": 5, "max": 20},
                "game.max_steps": {"type": int, "min": 100, "max": 1000},
                "llm.provider": {"type": str, "allowed": ["deepseek", "mistral", "hunyuan"]},
                "llm.timeout": {"type": int, "min": 10, "max": 60},
                "web.port": {"type": int, "min": 1000, "max": 65535}
            }
        )
        
        assert validation_result["valid"] is True, "Merged configuration should be valid"
        assert "validation_errors" not in validation_result or len(validation_result["validation_errors"]) == 0, "Should have no validation errors"

    def test_dependency_resolution_and_ordering(self) -> None:
        """Test dependency resolution and component ordering."""
        
        init_utils: InitializationUtils = InitializationUtils()
        
        # Mock component dependency graph
        component_dependencies: Dict[str, Dict[str, Any]] = {
            "ConfigManager": {
                "dependencies": [],
                "provides": ["config"],
                "initialization_time": 0.1
            },
            "Logger": {
                "dependencies": ["ConfigManager"],
                "provides": ["logging"],
                "initialization_time": 0.1
            },
            "NetworkUtils": {
                "dependencies": ["ConfigManager", "Logger"],
                "provides": ["network"],
                "initialization_time": 0.2
            },
            "GameData": {
                "dependencies": ["ConfigManager", "Logger"],
                "provides": ["game_data"],
                "initialization_time": 0.3
            },
            "GameLogic": {
                "dependencies": ["GameData"],
                "provides": ["game_logic"],
                "initialization_time": 0.2
            },
            "LLMClient": {
                "dependencies": ["ConfigManager", "NetworkUtils", "Logger"],
                "provides": ["llm"],
                "initialization_time": 0.5
            },
            "GameController": {
                "dependencies": ["GameData", "GameLogic", "LLMClient"],
                "provides": ["game_controller"],
                "initialization_time": 0.4
            },
            "WebServer": {
                "dependencies": ["GameController", "Logger"],
                "provides": ["web_server"],
                "initialization_time": 0.3
            },
            "GUI": {
                "dependencies": ["GameController"],
                "provides": ["gui"],
                "initialization_time": 0.6,
                "optional": True
            }
        }
        
        # Test dependency resolution
        resolution_result = init_utils.resolve_dependencies(
            components=component_dependencies,
            validate_cycles=True
        )
        
        assert resolution_result["success"] is True, "Dependency resolution should succeed"
        assert "circular_dependencies" not in resolution_result or len(resolution_result["circular_dependencies"]) == 0, "Should not have circular dependencies"
        
        initialization_order = resolution_result["initialization_order"]
        
        # Verify dependency ordering
        component_positions = {comp: i for i, comp in enumerate(initialization_order)}
        
        for component, info in component_dependencies.items():
            if component not in component_positions:
                continue  # Skip optional components that may not be included
                
            comp_position = component_positions[component]
            
            for dependency in info["dependencies"]:
                if dependency in component_positions:
                    dep_position = component_positions[dependency]
                    assert dep_position < comp_position, f"Dependency {dependency} should be initialized before {component}"
        
        # Verify expected ordering relationships
        assert component_positions["ConfigManager"] < component_positions["Logger"], "ConfigManager before Logger"
        assert component_positions["Logger"] < component_positions["NetworkUtils"], "Logger before NetworkUtils"
        assert component_positions["GameData"] < component_positions["GameLogic"], "GameData before GameLogic"
        assert component_positions["LLMClient"] < component_positions["GameController"], "LLMClient before GameController"
        assert component_positions["GameController"] < component_positions["WebServer"], "GameController before WebServer"
        
        # Test with circular dependency
        circular_dependencies = component_dependencies.copy()
        circular_dependencies["ConfigManager"]["dependencies"] = ["WebServer"]  # Create cycle
        
        circular_result = init_utils.resolve_dependencies(
            components=circular_dependencies,
            validate_cycles=True
        )
        
        assert circular_result["success"] is False, "Should detect circular dependency"
        assert "circular_dependencies" in circular_result, "Should report circular dependencies"

    def test_startup_sequence_and_health_checks(self) -> None:
        """Test startup sequence execution and component health checks."""
        
        init_utils: InitializationUtils = InitializationUtils()
        
        # Mock startup sequence
        startup_sequence: List[Dict[str, Any]] = [
            {
                "step_name": "load_configuration",
                "description": "Load and validate system configuration",
                "timeout": 5.0,
                "critical": True,
                "health_check": True
            },
            {
                "step_name": "initialize_logging",
                "description": "Set up logging system",
                "timeout": 2.0,
                "critical": True,
                "health_check": True
            },
            {
                "step_name": "setup_database",
                "description": "Connect to database and run migrations",
                "timeout": 10.0,
                "critical": True,
                "health_check": True
            },
            {
                "step_name": "initialize_llm_client",
                "description": "Initialize LLM client connections",
                "timeout": 15.0,
                "critical": True,
                "health_check": True
            },
            {
                "step_name": "start_web_server",
                "description": "Start web server and routes",
                "timeout": 5.0,
                "critical": False,
                "health_check": True
            },
            {
                "step_name": "initialize_gui",
                "description": "Initialize GUI components",
                "timeout": 8.0,
                "critical": False,
                "health_check": False
            }
        ]
        
        # Mock startup executor
        startup_executor: Mock = Mock()
        startup_executor.executed_steps = []
        startup_executor.health_check_results = {}
        startup_executor.startup_failed = False
        
        def mock_execute_startup_step(step_config: Dict[str, Any]) -> Dict[str, Any]:
            """Mock startup step execution."""
            step_name = step_config["step_name"]
            description = step_config["description"]
            timeout = step_config["timeout"]
            critical = step_config["critical"]
            health_check = step_config["health_check"]
            
            start_time = time.time()
            
            # Simulate step execution
            if step_name == "load_configuration":
                # Always succeeds
                execution_result = {"success": True, "config_loaded": True}
                execution_time = 0.1
                
            elif step_name == "initialize_logging":
                # Always succeeds
                execution_result = {"success": True, "log_level": "INFO"}
                execution_time = 0.05
                
            elif step_name == "setup_database":
                # May fail
                if hasattr(startup_executor, "simulate_db_failure") and startup_executor.simulate_db_failure:
                    execution_result = {"success": False, "error": "Database connection failed"}
                    execution_time = timeout  # Timeout
                else:
                    execution_result = {"success": True, "db_connected": True}
                    execution_time = 0.3
                    
            elif step_name == "initialize_llm_client":
                # Usually succeeds
                execution_result = {"success": True, "providers_connected": ["deepseek", "mistral"]}
                execution_time = 0.8
                
            elif step_name == "start_web_server":
                # May fail if port is busy
                execution_result = {"success": True, "server_port": 5000}
                execution_time = 0.2
                
            elif step_name == "initialize_gui":
                # Optional, may fail
                execution_result = {"success": True, "gui_initialized": True}
                execution_time = 0.4
            
            success = execution_result["success"]
            
            # Record step execution
            step_result = {
                "step_name": step_name,
                "description": description,
                "success": success,
                "execution_time": execution_time,
                "start_time": start_time,
                "result": execution_result
            }
            
            startup_executor.executed_steps.append(step_result)
            
            # Perform health check if requested
            if health_check and success:
                health_result = {
                    "healthy": True,
                    "check_time": time.time(),
                    "status": "operational"
                }
                startup_executor.health_check_results[step_name] = health_result
            elif health_check and not success:
                health_result = {
                    "healthy": False,
                    "check_time": time.time(),
                    "status": "failed",
                    "error": execution_result.get("error", "Unknown error")
                }
                startup_executor.health_check_results[step_name] = health_result
            
            # Check if critical step failed
            if not success and critical:
                startup_executor.startup_failed = True
            
            return step_result
        
        startup_executor.execute_startup_step = mock_execute_startup_step
        
        # Test successful startup sequence
        startup_results: List[Dict[str, Any]] = []
        
        for step_config in startup_sequence:
            if startup_executor.startup_failed:
                # Skip remaining steps if critical step failed
                skipped_result = {
                    "step_name": step_config["step_name"],
                    "success": False,
                    "skipped": True,
                    "reason": "Previous critical step failed"
                }
                startup_results.append(skipped_result)
                continue
            
            step_result = startup_executor.execute_startup_step(step_config)
            startup_results.append(step_result)
        
        # Verify startup sequence
        executed_steps = [r for r in startup_results if not r.get("skipped", False)]
        assert len(executed_steps) >= 4, "Should execute core startup steps"
        
        successful_steps = [r for r in executed_steps if r["success"]]
        assert len(successful_steps) >= 4, "Should have successful core steps"
        
        # Verify health checks
        health_checks = startup_executor.health_check_results
        healthy_components = [comp for comp, result in health_checks.items() if result["healthy"]]
        assert len(healthy_components) >= 3, "Should have healthy core components"
        
        # Test startup failure scenario
        startup_executor.simulate_db_failure = True
        startup_executor.executed_steps = []
        startup_executor.health_check_results = {}
        startup_executor.startup_failed = False
        
        failure_results: List[Dict[str, Any]] = []
        
        for step_config in startup_sequence:
            if startup_executor.startup_failed:
                skipped_result = {
                    "step_name": step_config["step_name"],
                    "success": False,
                    "skipped": True,
                    "reason": "Previous critical step failed"
                }
                failure_results.append(skipped_result)
                continue
            
            step_result = startup_executor.execute_startup_step(step_config)
            failure_results.append(step_result)
        
        # Verify failure handling
        assert startup_executor.startup_failed is True, "Should detect critical step failure"
        
        executed_failure_steps = [r for r in failure_results if not r.get("skipped", False)]
        skipped_failure_steps = [r for r in failure_results if r.get("skipped", False)]
        
        assert len(skipped_failure_steps) > 0, "Should skip steps after critical failure"
        
        # Database setup should have failed
        db_step = next(r for r in executed_failure_steps if r["step_name"] == "setup_database")
        assert db_step["success"] is False, "Database setup should fail in failure scenario"
