"""
Example: How to Create Web Extensions using the Minimal Flask Architecture
--------------------

This file demonstrates how future extensions (Task 1-5) can easily create
web interfaces by copying and modifying the simple patterns.

Extension Pattern:
1. Copy SimpleFlaskApp pattern
2. Override 3 methods: get_game_data(), get_api_state(), handle_control()
3. Create factory function
4. Create web script following scripts/human_play_web.py pattern

Educational Value:
- Shows copy-paste simplicity
- Demonstrates KISS principles
- Provides concrete examples for different extension types
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.game_flask_app import SimpleFlaskApp


# Example 1: Heuristics Extension (Task-1)
class HeuristicWebApp(SimpleFlaskApp):
    """
    Example heuristics web application.
    
    Extension Pattern: Copy this for any algorithm-based extension
    Educational Value: Shows minimal specialization for pathfinding algorithms
    """
    
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10):
        """Initialize heuristic web app."""
        super().__init__(f"{algorithm} Snake Game")
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.path_length = 0
        self.nodes_explored = 0
        print(f"[HeuristicWeb] {algorithm} initialized for {grid_size}x{grid_size} grid")
    
    def get_game_data(self):
        """Get heuristic game data for template."""
        return {
            'name': self.name,
            'mode': 'heuristic',  # Shows algorithm-section in template
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'features': ['Pathfinding', 'Optimal Routes', 'Step-by-step Visualization'],
            'status': 'ready'
        }
    
    def get_api_state(self):
        """Get heuristic API state."""
        return {
            'mode': 'heuristic',
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'path_length': self.path_length,
            'nodes_explored': self.nodes_explored,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        """Handle heuristic controls."""
        action = data.get('action', '')
        
        if action == 'find_path':
            print(f"[HeuristicWeb] Finding path with {self.algorithm}")
            # Simulate pathfinding
            self.path_length = 15
            self.nodes_explored = 42
            return {
                'action': 'find_path',
                'algorithm': self.algorithm,
                'path_length': self.path_length,
                'nodes_explored': self.nodes_explored,
                'status': 'path_found'
            }
        elif action == 'step':
            print("[HeuristicWeb] Step execution")
            return {'action': 'step', 'status': 'stepped'}
        elif action == 'reset':
            print("[HeuristicWeb] Reset")
            self.path_length = 0
            self.nodes_explored = 0
            return {'action': 'reset', 'status': 'reset'}
        
        return {'error': 'Unknown action'}


# Example 2: RL Extension (Task-2)  
class RLWebApp(SimpleFlaskApp):
    """
    Example RL web application.
    
    Extension Pattern: Copy this for any ML/RL extension
    Educational Value: Shows minimal specialization for training/inference
    """
    
    def __init__(self, agent_type: str = "DQN", grid_size: int = 10):
        """Initialize RL web app."""
        super().__init__(f"{agent_type} RL Snake Game")
        self.agent_type = agent_type
        self.grid_size = grid_size
        self.episode = 0
        self.reward = 0.0
        self.loss = 0.0
        print(f"[RLWeb] {agent_type} initialized for {grid_size}x{grid_size} grid")
    
    def get_game_data(self):
        """Get RL game data for template."""
        return {
            'name': self.name,
            'mode': 'rl',  # Shows training-section in template
            'agent_type': self.agent_type,
            'grid_size': self.grid_size,
            'features': ['Deep Q-Learning', 'Experience Replay', 'Training Metrics'],
            'status': 'ready'
        }
    
    def get_api_state(self):
        """Get RL API state."""
        return {
            'mode': 'rl',
            'agent_type': self.agent_type,
            'grid_size': self.grid_size,
            'episode': self.episode,
            'reward': self.reward,
            'loss': self.loss,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        """Handle RL controls."""
        action = data.get('action', '')
        
        if action == 'train':
            print(f"[RLWeb] Training {self.agent_type}")
            # Simulate training
            self.episode += 1
            self.reward = 12.5
            self.loss = 0.023
            return {
                'action': 'train',
                'agent_type': self.agent_type,
                'episode': self.episode,
                'reward': self.reward,
                'loss': self.loss,
                'status': 'training'
            }
        elif action == 'predict':
            state = data.get('state', [])
            print(f"[RLWeb] Prediction for state length: {len(state)}")
            # Simulate prediction
            prediction = {'move': 'UP', 'confidence': 0.85}
            return {'action': 'predict', 'prediction': prediction, 'status': 'predicted'}
        elif action == 'reset':
            print("[RLWeb] Reset training")
            self.episode = 0
            self.reward = 0.0
            self.loss = 0.0
            return {'action': 'reset', 'status': 'reset'}
        
        return {'error': 'Unknown action'}


# Example 3: Multi-Agent Comparison Extension
class ComparisonWebApp(SimpleFlaskApp):
    """
    Example multi-agent comparison web application.
    
    Extension Pattern: Copy this for comparison/ensemble extensions
    Educational Value: Shows minimal specialization for multi-agent scenarios
    """
    
    def __init__(self, agents: list = None):
        """Initialize comparison web app."""
        super().__init__("Multi-Agent Comparison")
        self.agents = agents or ["BFS", "A*", "DQN", "PPO"]
        self.current_agent = 0
        self.results = {}
        print(f"[ComparisonWeb] Initialized with {len(self.agents)} agents")
    
    def get_game_data(self):
        """Get comparison game data for template."""
        return {
            'name': self.name,
            'mode': 'comparison',
            'agents': self.agents,
            'current_agent': self.current_agent,
            'results': self.results,
            'features': ['Agent Comparison', 'Performance Metrics', 'Head-to-head'],
            'status': 'ready'
        }
    
    def get_api_state(self):
        """Get comparison API state."""
        return {
            'mode': 'comparison',
            'agents': self.agents,
            'current_agent': self.current_agent,
            'current_agent_name': self.agents[self.current_agent],
            'results': self.results,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        """Handle comparison controls."""
        action = data.get('action', '')
        
        if action == 'switch_agent':
            agent_index = data.get('agent_index', 0)
            if 0 <= agent_index < len(self.agents):
                self.current_agent = agent_index
                agent_name = self.agents[agent_index]
                print(f"[ComparisonWeb] Switched to {agent_name}")
                return {
                    'action': 'switch_agent',
                    'current_agent': agent_index,
                    'agent_name': agent_name,
                    'status': 'switched'
                }
        elif action == 'run_comparison':
            print("[ComparisonWeb] Running comparison between all agents")
            # Simulate comparison
            self.results = {
                'BFS': {'score': 8.2, 'time': 0.15},
                'A*': {'score': 9.1, 'time': 0.23},
                'DQN': {'score': 7.8, 'time': 0.05},
                'PPO': {'score': 8.9, 'time': 0.07}
            }
            return {
                'action': 'run_comparison',
                'results': self.results,
                'status': 'completed'
            }
        elif action == 'reset':
            print("[ComparisonWeb] Reset comparison")
            self.results = {}
            self.current_agent = 0
            return {'action': 'reset', 'status': 'reset'}
        
        return {'error': 'Unknown action'}


# Factory functions (KISS pattern)

def create_heuristic_app(algorithm: str = "BFS", **config) -> HeuristicWebApp:
    """Create heuristic web app."""
    return HeuristicWebApp(algorithm=algorithm, **config)


def create_rl_app(agent_type: str = "DQN", **config) -> RLWebApp:
    """Create RL web app."""
    return RLWebApp(agent_type=agent_type, **config)


def create_comparison_app(agents: list = None, **config) -> ComparisonWebApp:
    """Create comparison web app."""
    return ComparisonWebApp(agents=agents, **config)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Extension Examples")
    print()
    
    # Test heuristic app
    print("1. Testing Heuristic App:")
    heuristic_app = create_heuristic_app("A*", grid_size=15)
    print(f"   Game Data: {heuristic_app.get_game_data()}")
    print(f"   API State: {heuristic_app.get_api_state()}")
    print(f"   Control Test: {heuristic_app.handle_control({'action': 'find_path'})}")
    print()
    
    # Test RL app
    print("2. Testing RL App:")
    rl_app = create_rl_app("PPO", grid_size=12)
    print(f"   Game Data: {rl_app.get_game_data()}")
    print(f"   API State: {rl_app.get_api_state()}")
    print(f"   Control Test: {rl_app.handle_control({'action': 'train'})}")
    print()
    
    # Test comparison app
    print("3. Testing Comparison App:")
    comparison_app = create_comparison_app(["BFS", "DQN"])
    print(f"   Game Data: {comparison_app.get_game_data()}")
    print(f"   API State: {comparison_app.get_api_state()}")
    print(f"   Control Test: {comparison_app.handle_control({'action': 'switch_agent', 'agent_index': 1})}")
    print()
    
    print("✅ All extension examples working correctly!")
    print()
    print("🚀 To use these patterns:")
    print("   1. Copy the class you need")
    print("   2. Modify the methods for your specific use case")
    print("   3. Create a factory function")
    print("   4. Create a web script following scripts/*_web.py pattern")
    print("   5. Test with the base.html template") 