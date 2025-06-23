"""
Supervised Learning v0.01 - Game Manager
=======================================

Game manager for supervised learning v0.01, focusing on neural networks only.
Extends BaseGameManager from Task-0, demonstrating perfect base class reuse.

Design Pattern: Template Method + Factory Pattern
- Extends BaseGameManager for consistent game loop
- Factory pattern for neural network agent creation
- Simple structure focused on proof of concept
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from core.game_manager import BaseGameManager
from extensions.common.path_utils import setup_extension_paths
from game_logic import SupervisedGameLogic
from agent_neural import MLPAgent, CNNAgent, LSTMAgent
setup_extension_paths()


class SupervisedGameManager(BaseGameManager):
    """
    Game manager for supervised learning v0.01.
    
    Extends BaseGameManager to demonstrate perfect base class reuse.
    Focuses on neural networks only (MLP, CNN, LSTM) for proof of concept.
    
    Design Pattern: Template Method
    - Inherits game loop and session management from BaseGameManager
    - Implements neural network-specific initialization and evaluation
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
        self.model_type = getattr(args, 'model_type', 'MLP')
        self.training_mode = getattr(args, 'training_mode', False)
        self.grid_size = getattr(args, 'grid_size', 10)
        
        # Neural network agents (v0.01 focuses on neural networks only)
        self.neural_agents = {
            'MLP': MLPAgent,
            'CNN': CNNAgent,
            'LSTM': LSTMAgent
        }
        
        # Current agent instance
        self.current_agent = None
        
        # Extension-specific logging setup
        self.extension_name = "supervised-v0.01"
        self.setup_extension_logging()
    
    def setup_extension_logging(self):
        """
        Setup logging for supervised learning extension.
        
        Logs are stored in ROOT/logs/extensions/supervised-v0.01/
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
        
        Creates neural network agent and prepares for evaluation.
        Demonstrates neural network-specific initialization.
        """
        print(f"Initializing Supervised Learning v0.01 - {self.model_type}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Mode: {'Training' if self.training_mode else 'Evaluation'}")
        
        # Create neural network agent
        if self.model_type not in self.neural_agents:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        agent_class = self.neural_agents[self.model_type]
        self.current_agent = agent_class(grid_size=self.grid_size)
        
        print(f"Created {self.model_type} agent")
        
        # Initialize base game manager
        super().initialize()
    
    def create_game_logic(self) -> SupervisedGameLogic:
        """
        Create supervised learning game logic instance.
        
        Returns:
            SupervisedGameLogic instance configured for neural networks
        """
        return self.GAME_LOGIC_CLS(
            agent=self.current_agent,
            grid_size=self.grid_size,
            max_steps=self.args.max_steps
        )
    
    def run(self):
        """
        Run the supervised learning evaluation.
        
        Demonstrates neural network evaluation on the Snake game.
        Follows the same pattern as heuristics extensions.
        """
        print("\n" + "=" * 60)
        print("Starting Supervised Learning v0.01 Evaluation")
        print("=" * 60)
        
        # Initialize if not already done
        if not self.initialized:
            self.initialize()
        
        # Run games using base class functionality
        self.run_games()
        
        # Generate supervised learning specific summary
        self.generate_supervised_summary()
        
        print("\n" + "=" * 60)
        print("Supervised Learning v0.01 Evaluation Complete")
        print("=" * 60)
    
    def generate_supervised_summary(self):
        """
        Generate supervised learning specific summary.
        
        Extends base summary with neural network specific metrics.
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
            "version": "v0.01",
            "model_type": self.model_type,
            "grid_size": self.grid_size,
            "training_mode": self.training_mode,
            "neural_network_metrics": {
                "model_architecture": self.model_type,
                "input_features": self.current_agent.input_size if hasattr(self.current_agent, 'input_size') else "N/A",
                "is_trained": self.current_agent.is_trained if hasattr(self.current_agent, 'is_trained') else False
            }
        }
        
        # Save supervised learning specific summary
        summary_filepath = self.get_summary_filepath()
        self.save_summary(summary_filepath, supervised_summary)
        
        print(f"Supervised learning summary saved to: {summary_filepath}")
    
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
        
        Extends base cleanup with neural network specific cleanup.
        """
        # Neural network specific cleanup
        if self.current_agent and hasattr(self.current_agent, 'cleanup'):
            self.current_agent.cleanup()
        
        # Call parent cleanup
        super().cleanup() 