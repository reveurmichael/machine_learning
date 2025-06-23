"""
Supervised Learning v0.02 - Game Manager
=======================================

Game manager for supervised learning v0.02, supporting all ML model types.
Extends BaseGameManager from Task-0, demonstrating perfect base class reuse.

Design Pattern: Template Method + Factory Pattern
- Extends BaseGameManager for consistent game loop
- Factory pattern for multi-model agent creation
- Organized structure supporting neural, tree, and graph models
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Type

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from core.game_manager import BaseGameManager
from extensions.common.path_utils import setup_extension_paths
from extensions.supervised_v0_02.game_logic import SupervisedGameLogic
setup_extension_paths()


class SupervisedGameManager(BaseGameManager):
    """
    Game manager for supervised learning v0.02.
    
    Extends BaseGameManager to demonstrate perfect base class reuse.
    Supports all ML model types (neural, tree, graph) for comprehensive evaluation.
    
    Design Pattern: Template Method
    - Inherits game loop and session management from BaseGameManager
    - Implements multi-model-specific initialization and evaluation
    - Demonstrates how extensions can reuse core infrastructure
    """
    
    # Factory pattern: specify which game logic class to use
    GAME_LOGIC_CLS = SupervisedGameLogic
    
    def __init__(self, args):
        """
        Initialize supervised learning game manager.
        
        Args:
            args: Arguments object with configuration parameters
        """
        # Call parent constructor to inherit all base functionality
        super().__init__(args)
        
        # Supervised learning specific attributes
        self.model_type = getattr(args, 'model', 'MLP').upper()
        self.grid_size = getattr(args, 'grid_size', 10)
        
        # Model categories and their agent classes
        self.model_agents = self._initialize_model_agents()
        
        # Current agent instance
        self.current_agent = None
        
        # Extension-specific logging setup
        self.extension_name = "supervised-v0.02"
        self.setup_extension_logging()
    
    def _initialize_model_agents(self) -> Dict[str, Type]:
        """
        Initialize model agent classes for all supported model types.
        
        Factory pattern for model creation - maps model names to agent classes.
        Supports neural networks, tree models, and graph neural networks.
        
        Returns:
            Dictionary mapping model names to agent classes
        """
        agents = {}
        
        # Neural network agents
        try:
            from extensions.supervised_v0_02.models.neural_networks.agent_mlp import MLPAgent
            from extensions.supervised_v0_02.models.neural_networks.agent_cnn import CNNAgent
            from extensions.supervised_v0_02.models.neural_networks.agent_lstm import LSTMAgent
            
            agents.update({
                'MLP': MLPAgent,
                'CNN': CNNAgent,
                'LSTM': LSTMAgent,
            })
        except ImportError as e:
            print(f"Warning: Neural network agents not available: {e}")
        
        # Tree model agents
        try:
            from extensions.supervised_v0_02.models.tree_models.agent_xgboost import XGBoostAgent
            from extensions.supervised_v0_02.models.tree_models.agent_lightgbm import LightGBMAgent
            from extensions.supervised_v0_02.models.tree_models.agent_randomforest import RandomForestAgent
            
            agents.update({
                'XGBOOST': XGBoostAgent,
                'LIGHTGBM': LightGBMAgent,
                'RANDOMFOREST': RandomForestAgent,
            })
        except ImportError as e:
            print(f"Warning: Tree model agents not available: {e}")
        
        # Graph neural network agents
        try:
            from extensions.supervised_v0_02.models.graph_models.agent_gcn import GCNAgent
            
            agents.update({
                'GCN': GCNAgent,
                'GRAPHSAGE': GCNAgent,  # Use GCN as placeholder
                'GAT': GCNAgent,        # Use GCN as placeholder
            })
        except ImportError as e:
            print(f"Warning: Graph model agents not available: {e}")
        
        return agents
    
    def setup_extension_logging(self):
        """
        Setup logging for supervised learning extension.
        
        Logs are stored in ROOT/logs/extensions/supervised-v0.02/
        following the same pattern as heuristics extensions.
        """
        # Create extension-specific log directory
        extension_log_dir = Path("logs/extensions") / self.extension_name
        extension_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update log directory if not specified
        if not self.args.log_dir:
            self.args.log_dir = str(extension_log_dir)
    
    def initialize(self):
        """
        Initialize the supervised learning game manager.
        
        Creates multi-model agent and prepares for evaluation.
        Demonstrates multi-model-specific initialization.
        """
        print(f"Initializing Supervised Learning v0.02 - {self.model_type}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Available models: {list(self.model_agents.keys())}")
        
        # Create model agent
        if self.model_type not in self.model_agents:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        agent_class = self.model_agents[self.model_type]
        self.current_agent = agent_class(grid_size=self.grid_size)
        
        print(f"Created {self.model_type} agent")
        
        # Initialize base game manager
        super().initialize()
    
    def create_game_logic(self) -> SupervisedGameLogic:
        """
        Create supervised learning game logic instance.
        
        Returns:
            SupervisedGameLogic instance configured for multi-model evaluation
        """
        return self.GAME_LOGIC_CLS(
            agent=self.current_agent,
            grid_size=self.grid_size,
            max_steps=self.args.max_steps
        )
    
    def run(self):
        """
        Run the supervised learning evaluation.
        
        Demonstrates multi-model evaluation on the Snake game.
        Follows the same pattern as heuristics extensions.
        """
        print("\n" + "=" * 60)
        print("Starting Supervised Learning v0.02 Evaluation")
        print("=" * 60)
        
        # Initialize if not already done
        if not self.initialized:
            self.initialize()
        
        # Run games using base class functionality
        self.run_games()
        
        # Generate supervised learning specific summary
        self.generate_supervised_summary()
        
        print("\n" + "=" * 60)
        print("Supervised Learning v0.02 Evaluation Complete")
        print("=" * 60)
    
    def generate_supervised_summary(self):
        """
        Generate supervised learning specific summary.
        
        Extends base summary with multi-model specific metrics.
        """
        if not self.game_stats:
            print("No games completed, skipping summary generation")
            return
        
        # Get base summary from parent
        base_summary = self.get_summary_data()
        
        # Add supervised learning specific metrics
        supervised_summary = {
            **base_summary,
            "extension_type": "supervised_learning",
            "version": "v0.02",
            "model_type": self.model_type,
            "grid_size": self.grid_size,
            "model_category": self._get_model_category(),
            "multi_model_metrics": {
                "model_architecture": self.model_type,
                "model_category": self._get_model_category(),
                "input_features": self.current_agent.input_size if hasattr(self.current_agent, 'input_size') else "N/A",
                "is_trained": self.current_agent.is_trained if hasattr(self.current_agent, 'is_trained') else False,
                "framework": self._get_model_framework()
            }
        }
        
        # Save supervised learning specific summary
        summary_filepath = self.get_summary_filepath()
        self.save_summary(summary_filepath, supervised_summary)
        
        print(f"Supervised learning summary saved to: {summary_filepath}")
    
    def _get_model_category(self) -> str:
        """
        Get the category of the current model.
        
        Returns:
            Model category (Neural, Tree, or Graph)
        """
        neural_models = ['MLP', 'CNN', 'LSTM', 'GRU']
        tree_models = ['XGBOOST', 'LIGHTGBM', 'RANDOMFOREST']
        graph_models = ['GCN', 'GRAPHSAGE', 'GAT']
        
        if self.model_type in neural_models:
            return "Neural"
        elif self.model_type in tree_models:
            return "Tree"
        elif self.model_type in graph_models:
            return "Graph"
        else:
            return "Unknown"
    
    def _get_model_framework(self) -> str:
        """
        Get the framework used by the current model.
        
        Returns:
            Framework name
        """
        neural_models = ['MLP', 'CNN', 'LSTM', 'GRU']
        tree_models = ['XGBOOST', 'LIGHTGBM', 'RANDOMFOREST']
        graph_models = ['GCN', 'GRAPHSAGE', 'GAT']
        
        if self.model_type in neural_models:
            return "PyTorch"
        elif self.model_type in tree_models:
            if self.model_type == 'XGBOOST':
                return "XGBoost"
            elif self.model_type == 'LIGHTGBM':
                return "LightGBM"
            else:
                return "Scikit-learn"
        elif self.model_type in graph_models:
            return "PyTorch Geometric"
        else:
            return "Unknown"
    
    def get_summary_filepath(self) -> str:
        """
        Get the filepath for the supervised learning summary.
        
        Returns:
            Path to the summary file
        """
        return os.path.join(self.args.log_dir, "supervised_summary.json")
    
    def save_summary(self, filepath: str, summary_data: Dict[str, Any]):
        """
        Save supervised learning summary to file.
        
        Args:
            filepath: Path to save the summary
            summary_data: Summary data to save
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
    
    def cleanup(self):
        """
        Cleanup resources after evaluation.
        
        Extends base cleanup with multi-model specific cleanup.
        """
        # Multi-model specific cleanup
        if self.current_agent and hasattr(self.current_agent, 'cleanup'):
            self.current_agent.cleanup()
        
        # Call parent cleanup
        super().cleanup() 