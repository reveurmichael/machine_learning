"""
Tests for Main application ↔ Dashboard tabs interactions.

Focuses on testing how the main application coordinates with dashboard tabs
for state sharing, event coordination, and user interface consistency.
"""

import pytest
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

# Mock dashboard components since they may not exist in core
class MockDashboardTab:
    def __init__(self, tab_name: str):
        self.tab_name = tab_name
        self.is_active = False
        self.state = {}
        self.event_handlers = {}
        self.update_count = 0
    
    def activate(self) -> None:
        self.is_active = True
    
    def deactivate(self) -> None:
        self.is_active = False
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        self.state.update(new_state)
        self.update_count += 1
    
    def handle_event(self, event_type: str, event_data: Any) -> None:
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event_data)


class TestMainDashboardInteractions:
    """Test interactions between main application and dashboard tabs."""

    def test_tab_state_synchronization(self) -> None:
        """Test state synchronization between main app and dashboard tabs."""
        
        # Mock main application state
        main_app_state = {
            "current_game": None,
            "score": 0,
            "game_status": "idle",
            "selected_llm": "deepseek",
            "grid_size": 12,
            "games_played": 0,
            "session_active": False
        }
        
        # Create mock dashboard tabs
        tabs = {
            "overview": MockDashboardTab("overview"),
            "human_play": MockDashboardTab("human_play"),
            "continue": MockDashboardTab("continue"),
            "replay": MockDashboardTab("replay")
        }
        
        # Mock dashboard manager
        dashboard_manager: Mock = Mock()
        dashboard_manager.tabs = tabs
        dashboard_manager.active_tab = None
        dashboard_manager.state_sync_log = []
        
        def mock_sync_tab_state(tab_name: str, app_state: Dict[str, Any]) -> bool:
            """Synchronize application state with specific tab."""
            if tab_name not in tabs:
                return False
            
            tab = tabs[tab_name]
            
            # Filter relevant state for each tab
            if tab_name == "overview":
                relevant_state = {
                    "games_played": app_state.get("games_played", 0),
                    "session_active": app_state.get("session_active", False),
                    "selected_llm": app_state.get("selected_llm", "unknown")
                }
            elif tab_name == "human_play":
                relevant_state = {
                    "current_game": app_state.get("current_game"),
                    "score": app_state.get("score", 0),
                    "game_status": app_state.get("game_status", "idle"),
                    "grid_size": app_state.get("grid_size", 10)
                }
            elif tab_name == "continue":
                relevant_state = {
                    "session_active": app_state.get("session_active", False),
                    "games_played": app_state.get("games_played", 0),
                    "selected_llm": app_state.get("selected_llm", "unknown")
                }
            elif tab_name == "replay":
                relevant_state = {
                    "games_played": app_state.get("games_played", 0),
                    "session_active": app_state.get("session_active", False)
                }
            else:
                relevant_state = {}
            
            tab.update_state(relevant_state)
            
            dashboard_manager.state_sync_log.append({
                "tab": tab_name,
                "state": relevant_state.copy(),
                "timestamp": time.time(),
                "sync_successful": True
            })
            
            return True
        
        def mock_sync_all_tabs(app_state: Dict[str, Any]) -> Dict[str, bool]:
            """Synchronize application state with all tabs."""
            sync_results = {}
            
            for tab_name in tabs.keys():
                sync_results[tab_name] = mock_sync_tab_state(tab_name, app_state)
            
            return sync_results
        
        dashboard_manager.sync_tab_state = mock_sync_tab_state
        dashboard_manager.sync_all_tabs = mock_sync_all_tabs
        
        # Test state synchronization scenarios
        state_changes = [
            {
                "change": "start_session",
                "new_state": {**main_app_state, "session_active": True, "games_played": 0}
            },
            {
                "change": "start_game",
                "new_state": {**main_app_state, "session_active": True, "current_game": "game_1", "game_status": "playing", "score": 0}
            },
            {
                "change": "score_update",
                "new_state": {**main_app_state, "session_active": True, "current_game": "game_1", "game_status": "playing", "score": 150}
            },
            {
                "change": "game_complete",
                "new_state": {**main_app_state, "session_active": True, "current_game": None, "game_status": "completed", "score": 150, "games_played": 1}
            },
            {
                "change": "llm_change",
                "new_state": {**main_app_state, "session_active": True, "selected_llm": "mistral", "games_played": 1}
            }
        ]
        
        sync_test_results: List[Dict[str, Any]] = []
        
        for i, change_scenario in enumerate(state_changes):
            change_name = change_scenario["change"]
            new_state = change_scenario["new_state"]
            
            # Update main app state
            main_app_state.update(new_state)
            
            # Synchronize all tabs
            sync_start_time = time.time()
            sync_results = dashboard_manager.sync_all_tabs(main_app_state)
            sync_end_time = time.time()
            
            # Verify synchronization
            all_synced = all(sync_results.values())
            
            sync_test_results.append({
                "change": change_name,
                "iteration": i,
                "sync_results": sync_results,
                "all_synced": all_synced,
                "sync_duration": sync_end_time - sync_start_time,
                "main_state": main_app_state.copy()
            })
            
            # Verify individual tab states
            for tab_name, tab in tabs.items():
                if tab_name == "overview":
                    assert tab.state.get("games_played") == main_app_state["games_played"], \
                        f"Overview tab games_played not synced for {change_name}"
                    assert tab.state.get("session_active") == main_app_state["session_active"], \
                        f"Overview tab session_active not synced for {change_name}"
                
                elif tab_name == "human_play":
                    assert tab.state.get("score") == main_app_state["score"], \
                        f"Human play tab score not synced for {change_name}"
                    assert tab.state.get("game_status") == main_app_state["game_status"], \
                        f"Human play tab game_status not synced for {change_name}"
        
        # Verify synchronization results
        assert len(sync_test_results) == 5, "Should test all state change scenarios"
        assert all(result["all_synced"] for result in sync_test_results), "All synchronizations should succeed"
        
        # Verify synchronization log
        assert len(dashboard_manager.state_sync_log) == 20, "Should log all tab synchronizations (5 changes × 4 tabs)"
        
        # Verify performance
        avg_sync_duration = sum(result["sync_duration"] for result in sync_test_results) / len(sync_test_results)
        assert avg_sync_duration < 0.01, f"Synchronization too slow: {avg_sync_duration}s"

    def test_tab_activation_coordination(self) -> None:
        """Test coordination when switching between dashboard tabs."""
        
        # Create mock tabs with activation behavior
        tabs = {
            "overview": MockDashboardTab("overview"),
            "human_play": MockDashboardTab("human_play"),
            "continue": MockDashboardTab("continue"),
            "replay": MockDashboardTab("replay")
        }
        
        # Mock tab manager with activation coordination
        tab_manager: Mock = Mock()
        tab_manager.active_tab = None
        tab_manager.activation_history = []
        tab_manager.tab_states = {}
        
        def mock_activate_tab(tab_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
            """Activate tab with coordination."""
            if tab_name not in tabs:
                return {"success": False, "error": "Tab not found"}
            
            # Deactivate current tab
            if tab_manager.active_tab:
                old_tab = tabs[tab_manager.active_tab]
                old_tab.deactivate()
                
                # Save old tab state
                tab_manager.tab_states[tab_manager.active_tab] = old_tab.state.copy()
            
            # Activate new tab
            new_tab = tabs[tab_name]
            new_tab.activate()
            
            # Apply context to new tab
            if tab_name in tab_manager.tab_states:
                # Restore previous state
                new_tab.update_state(tab_manager.tab_states[tab_name])
            
            # Apply activation context
            new_tab.update_state(context)
            
            # Update active tab
            tab_manager.active_tab = tab_name
            
            # Log activation
            activation_record = {
                "tab": tab_name,
                "context": context.copy(),
                "timestamp": time.time(),
                "success": True
            }
            tab_manager.activation_history.append(activation_record)
            
            return {
                "success": True,
                "active_tab": tab_name,
                "tab_state": new_tab.state.copy()
            }
        
        tab_manager.activate_tab = mock_activate_tab
        
        # Test tab activation scenarios
        activation_scenarios = [
            {
                "tab": "overview",
                "context": {"session_summary": {"games": 0, "total_score": 0}}
            },
            {
                "tab": "human_play",
                "context": {"game_mode": "human", "grid_size": 12, "difficulty": "medium"}
            },
            {
                "tab": "continue",
                "context": {"session_id": "test_session", "has_saved_games": True}
            },
            {
                "tab": "replay",
                "context": {"replay_file": "game_1.json", "playback_speed": 1.0}
            },
            {
                "tab": "overview",
                "context": {"session_summary": {"games": 3, "total_score": 450}}
            }
        ]
        
        activation_results: List[Dict[str, Any]] = []
        
        for i, scenario in enumerate(activation_scenarios):
            tab_name = scenario["tab"]
            context = scenario["context"]
            
            # Activate tab
            activation_start = time.time()
            result = tab_manager.activate_tab(tab_name, context)
            activation_end = time.time()
            
            # Verify activation
            assert result["success"], f"Failed to activate {tab_name}"
            assert result["active_tab"] == tab_name, f"Active tab mismatch for {tab_name}"
            
            # Verify tab state
            tab = tabs[tab_name]
            assert tab.is_active, f"Tab {tab_name} should be active"
            
            # Verify context applied
            for key, value in context.items():
                assert tab.state.get(key) == value, f"Context {key} not applied to {tab_name}"
            
            # Verify other tabs are inactive
            for other_tab_name, other_tab in tabs.items():
                if other_tab_name != tab_name:
                    assert not other_tab.is_active, f"Tab {other_tab_name} should be inactive"
            
            activation_results.append({
                "tab": tab_name,
                "iteration": i,
                "activation_duration": activation_end - activation_start,
                "context_applied": context,
                "result": result
            })
        
        # Verify activation coordination
        assert len(activation_results) == 5, "Should test all activation scenarios"
        assert tab_manager.active_tab == "overview", "Should end with overview tab active"
        
        # Verify activation history
        assert len(tab_manager.activation_history) == 5, "Should log all activations"
        
        # Verify state preservation across activations
        # Overview was activated twice with different contexts
        overview_activations = [h for h in tab_manager.activation_history if h["tab"] == "overview"]
        assert len(overview_activations) == 2, "Overview should be activated twice"
        
        # Second overview activation should have updated context
        final_overview_state = tabs["overview"].state
        assert final_overview_state.get("session_summary", {}).get("games") == 3, "Overview state should be updated"

    def test_cross_tab_event_propagation(self) -> None:
        """Test event propagation between tabs and main application."""
        
        # Create tabs with event handling
        tabs = {
            "overview": MockDashboardTab("overview"),
            "human_play": MockDashboardTab("human_play"),
            "continue": MockDashboardTab("continue"),
            "replay": MockDashboardTab("replay")
        }
        
        # Set up event handlers for each tab
        event_log: List[Dict[str, Any]] = []
        
        def create_event_handler(tab_name: str, event_type: str):
            def handler(event_data: Any) -> None:
                event_log.append({
                    "tab": tab_name,
                    "event_type": event_type,
                    "event_data": event_data,
                    "timestamp": time.time()
                })
            return handler
        
        # Register event handlers
        for tab_name, tab in tabs.items():
            tab.event_handlers["game_started"] = create_event_handler(tab_name, "game_started")
            tab.event_handlers["game_ended"] = create_event_handler(tab_name, "game_ended")
            tab.event_handlers["score_updated"] = create_event_handler(tab_name, "score_updated")
            tab.event_handlers["llm_changed"] = create_event_handler(tab_name, "llm_changed")
        
        # Mock event broadcaster
        event_broadcaster: Mock = Mock()
        event_broadcaster.event_queue = []
        event_broadcaster.broadcast_log = []
        
        def mock_broadcast_event(
            event_type: str, 
            event_data: Any, 
            target_tabs: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Broadcast event to tabs."""
            
            # Determine target tabs
            if target_tabs is None:
                target_tabs = list(tabs.keys())
            
            broadcast_result = {
                "event_type": event_type,
                "event_data": event_data,
                "target_tabs": target_tabs,
                "delivery_results": {},
                "timestamp": time.time()
            }
            
            # Deliver to target tabs
            for tab_name in target_tabs:
                if tab_name in tabs:
                    try:
                        tab = tabs[tab_name]
                        tab.handle_event(event_type, event_data)
                        broadcast_result["delivery_results"][tab_name] = {"success": True}
                    except Exception as e:
                        broadcast_result["delivery_results"][tab_name] = {"success": False, "error": str(e)}
                else:
                    broadcast_result["delivery_results"][tab_name] = {"success": False, "error": "Tab not found"}
            
            event_broadcaster.broadcast_log.append(broadcast_result)
            return broadcast_result
        
        event_broadcaster.broadcast_event = mock_broadcast_event
        
        # Test event propagation scenarios
        event_scenarios = [
            {
                "event_type": "game_started",
                "event_data": {"game_id": "game_001", "player": "human", "grid_size": 12},
                "target_tabs": ["overview", "human_play"]
            },
            {
                "event_type": "score_updated",
                "event_data": {"new_score": 150, "delta": 50, "game_id": "game_001"},
                "target_tabs": ["overview", "human_play"]
            },
            {
                "event_type": "llm_changed",
                "event_data": {"old_llm": "deepseek", "new_llm": "mistral", "reason": "user_selection"},
                "target_tabs": None  # Broadcast to all
            },
            {
                "event_type": "game_ended",
                "event_data": {"game_id": "game_001", "final_score": 200, "outcome": "completed"},
                "target_tabs": ["overview", "human_play", "replay"]
            },
            {
                "event_type": "score_updated",
                "event_data": {"new_score": 75, "delta": 25, "game_id": "game_002"},
                "target_tabs": ["human_play"]  # Only current game tab
            }
        ]
        
        broadcast_results: List[Dict[str, Any]] = []
        
        for scenario in event_scenarios:
            event_type = scenario["event_type"]
            event_data = scenario["event_data"]
            target_tabs = scenario["target_tabs"]
            
            # Broadcast event
            broadcast_start = time.time()
            result = event_broadcaster.broadcast_event(event_type, event_data, target_tabs)
            broadcast_end = time.time()
            
            result["broadcast_duration"] = broadcast_end - broadcast_start
            broadcast_results.append(result)
        
        # Verify event propagation
        assert len(broadcast_results) == 5, "Should broadcast all events"
        
        # Verify all broadcasts succeeded
        for result in broadcast_results:
            for tab_name, delivery_result in result["delivery_results"].items():
                assert delivery_result["success"], f"Event delivery failed to {tab_name}"
        
        # Verify event log
        assert len(event_log) > 0, "Should log received events"
        
        # Verify specific event deliveries
        game_started_events = [e for e in event_log if e["event_type"] == "game_started"]
        assert len(game_started_events) == 2, "game_started should be delivered to 2 tabs"
        
        llm_changed_events = [e for e in event_log if e["event_type"] == "llm_changed"]
        assert len(llm_changed_events) == 4, "llm_changed should be delivered to all 4 tabs"
        
        score_updated_events = [e for e in event_log if e["event_type"] == "score_updated"]
        assert len(score_updated_events) == 3, "score_updated events should be delivered correctly"
        
        # Verify event ordering
        event_timestamps = [e["timestamp"] for e in event_log]
        assert event_timestamps == sorted(event_timestamps), "Events should be in chronological order"

    def test_dashboard_resource_management(self) -> None:
        """Test resource management coordination between main app and dashboard."""
        
        # Mock resource manager
        resource_manager: Mock = Mock()
        resource_manager.tab_resources = {}
        resource_manager.resource_usage = {"memory": 0, "cpu": 0, "network": 0}
        resource_manager.resource_limits = {"memory": 1000, "cpu": 100, "network": 50}
        
        def mock_allocate_resources(tab_name: str, resource_request: Dict[str, int]) -> Dict[str, Any]:
            """Allocate resources to tab."""
            allocation_result = {
                "tab": tab_name,
                "requested": resource_request.copy(),
                "allocated": {},
                "allocation_successful": True,
                "resource_conflicts": []
            }
            
            # Check if resources are available
            for resource_type, requested_amount in resource_request.items():
                current_usage = resource_manager.resource_usage.get(resource_type, 0)
                limit = resource_manager.resource_limits.get(resource_type, 0)
                
                if current_usage + requested_amount > limit:
                    allocation_result["allocation_successful"] = False
                    allocation_result["resource_conflicts"].append({
                        "resource": resource_type,
                        "requested": requested_amount,
                        "available": limit - current_usage,
                        "limit": limit
                    })
                else:
                    allocation_result["allocated"][resource_type] = requested_amount
                    resource_manager.resource_usage[resource_type] = current_usage + requested_amount
            
            # Store allocation if successful
            if allocation_result["allocation_successful"]:
                resource_manager.tab_resources[tab_name] = allocation_result["allocated"].copy()
            
            return allocation_result
        
        def mock_release_resources(tab_name: str) -> Dict[str, Any]:
            """Release resources from tab."""
            if tab_name not in resource_manager.tab_resources:
                return {"success": False, "error": "No resources allocated to tab"}
            
            released_resources = resource_manager.tab_resources[tab_name]
            
            # Release resources
            for resource_type, amount in released_resources.items():
                current_usage = resource_manager.resource_usage.get(resource_type, 0)
                resource_manager.resource_usage[resource_type] = max(0, current_usage - amount)
            
            # Remove allocation record
            del resource_manager.tab_resources[tab_name]
            
            return {
                "success": True,
                "tab": tab_name,
                "released_resources": released_resources
            }
        
        resource_manager.allocate_resources = mock_allocate_resources
        resource_manager.release_resources = mock_release_resources
        
        # Test resource management scenarios
        resource_scenarios = [
            {
                "tab": "overview",
                "resources": {"memory": 100, "cpu": 10}
            },
            {
                "tab": "human_play",
                "resources": {"memory": 300, "cpu": 30, "network": 20}
            },
            {
                "tab": "continue",
                "resources": {"memory": 150, "cpu": 15, "network": 10}
            },
            {
                "tab": "replay",
                "resources": {"memory": 400, "cpu": 25, "network": 15}
            },
            {
                "tab": "heavy_processing",  # Should fail due to resource limits
                "resources": {"memory": 800, "cpu": 80}
            }
        ]
        
        allocation_results: List[Dict[str, Any]] = []
        
        # Test resource allocation
        for scenario in resource_scenarios:
            tab_name = scenario["tab"]
            resource_request = scenario["resources"]
            
            allocation_result = resource_manager.allocate_resources(tab_name, resource_request)
            allocation_results.append(allocation_result)
        
        # Verify resource allocations
        successful_allocations = [r for r in allocation_results if r["allocation_successful"]]
        failed_allocations = [r for r in allocation_results if not r["allocation_successful"]]
        
        assert len(successful_allocations) == 4, "First 4 allocations should succeed"
        assert len(failed_allocations) == 1, "Last allocation should fail due to limits"
        
        # Verify resource limits respected
        failed_allocation = failed_allocations[0]
        assert len(failed_allocation["resource_conflicts"]) > 0, "Should identify resource conflicts"
        
        # Verify resource usage tracking
        total_memory_used = sum(
            alloc["allocated"].get("memory", 0) 
            for alloc in successful_allocations
        )
        assert resource_manager.resource_usage["memory"] == total_memory_used, "Memory usage should be tracked"
        
        # Test resource release
        release_results: List[Dict[str, Any]] = []
        
        for allocation in successful_allocations:
            tab_name = allocation["tab"]
            release_result = resource_manager.release_resources(tab_name)
            release_results.append(release_result)
        
        # Verify resource release
        assert len(release_results) == 4, "Should release all allocated resources"
        assert all(r["success"] for r in release_results), "All releases should succeed"
        
        # Verify resources are fully released
        assert resource_manager.resource_usage["memory"] == 0, "Memory should be fully released"
        assert resource_manager.resource_usage["cpu"] == 0, "CPU should be fully released"
        assert resource_manager.resource_usage["network"] == 0, "Network should be fully released"
        
        # Verify no dangling allocations
        assert len(resource_manager.tab_resources) == 0, "No tabs should have allocated resources" 