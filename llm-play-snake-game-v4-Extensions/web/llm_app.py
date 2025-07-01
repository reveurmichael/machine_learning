"""
LLM Web Game Application with Enhanced Naming
============================================

Web application for LLM-controlled Snake game with GameManager integration.
Provides Flask interface for LLM gameplay with proper Task-0 architecture.
Uses enhanced naming for maximum clarity and consistency.

Design Philosophy:
- Task-0 Integration: Uses existing GameManager and agent infrastructure
- Educational: Shows LLM web interface patterns with enhanced naming
- Lightweight: Simple demo interface (full LLM functionality in scripts/main_web.py)
- Enhanced Naming: Clear, explicit naming that indicates web + game + LLM domain
- Extension Pattern: Template for future LLM extension web interfaces

Educational Value:
- Shows LLM integration patterns in web applications with enhanced naming
- Demonstrates proper GameManager usage for LLM gameplay
- Provides template for extension LLM web interfaces with clear inheritance

Extension Pattern:
Future LLM extensions can copy this enhanced pattern and customize for their specific
LLM implementations while maintaining consistent web interface patterns.

Reference: web/base_app.py for layered web infrastructure
"""

from typing import Dict, Any

# Import Task-0 core components for proper LLM integration
from core.game_manager import GameManager

# Import base application class from layered web infrastructure
from web.base_app import SimpleFlaskApp

# Import utilities following SSOT principles  
from utils.web_utils import build_color_map
from utils.print_utils import create_logger

# Create logger for this module
print_log = create_logger("LLMWebGameApp")


class LLMWebGameApp(SimpleFlaskApp):
    """
    LLM player web game application with enhanced naming.
    
    Uses the existing GameManager infrastructure from Task-0 for proper
    LLM integration following established patterns. Enhanced naming
    clearly indicates this is a web-based game application for LLM players.
    
    Design Pattern: Adapter Pattern (Web + Game + LLM Integration)
    Purpose: Adapts GameManager interface for web API consumption with clear naming
    Educational Value: Shows LLM integration in web applications with enhanced clarity
    Extension Pattern: Copy this enhanced pattern for LLM extension web interfaces
    
    Enhanced Features:
    - Clear naming: LLMWebGameApp indicates web + game + LLM focus
    - Layered inheritance: Inherits from SimpleFlaskApp for game-specific features
    - GameManager integration: Proper Task-0 architecture patterns
    - Educational clarity: Enhanced naming for better understanding
    
    Note: For full LLM functionality with complete session management,
    use scripts/main_web.py. This provides a lightweight demo interface.
    """
    
    def __init__(self, provider: str = "hunyuan", model: str = "hunyuan-turbos-latest", 
                 grid_size: int = 10, **config):
        """Initialize LLM web game app with GameManager integration setup.
        
        Args:
            provider: LLM provider name (default: 'hunyuan')
            model: LLM model name (default: 'hunyuan-turbos-latest')
            grid_size: Size of the game grid (default: 10)
            **config: Additional configuration options
            
        Educational Value: Shows LLM component initialization patterns with enhanced naming
        Extension Pattern: Extensions can copy this setup for their LLM apps
        """
        super().__init__("LLM Snake Web Game")
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        self.config = config
        
        # Initialize game components when needed (lazy initialization)
        # For full implementation, these would be properly set up with GameManager
        self.game_manager = None
        self.agent = None
        
        print_log(f"LLM web game mode: {provider}/{model}, {grid_size}x{grid_size} grid")
        print_log("Demo mode - use scripts/main_web.py for full LLM functionality")
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get application data for template rendering.
        
        Returns:
            Dictionary with application data for HTML template
            
        Educational Value: Shows LLM-specific template data preparation with enhanced naming
        Extension Pattern: Extensions can copy this for their LLM template data
        """
        return {
            'name': self.name,
            'mode': 'llm_web',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'status': 'demo',
            'type': 'llm_web_game_app'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get LLM game state for API consumption.
        
        Returns:
            Dictionary with current LLM game state
            
        Educational Value: Shows LLM-specific state representation with enhanced structure
        Extension Pattern: Extensions can copy this state format for their LLM apps
        
        Note: Full implementation would integrate with GameManager.get_state()
        and include actual LLM reasoning, session management, and game progress.
        """
        # Demo state - for full LLM functionality, integrate with GameManager
        center = self.grid_size // 2
        return {
            'mode': 'llm_web',
            'app_type': 'llm_web_game_app',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'snake_positions': [[center, center]],  # Snake starts in center
            'apple_position': [center + 2, center + 2],  # Apple nearby
            'score': 0,
            'steps': 0,
            'running': True,
            'game_active': True,
            'game_over': False,
            'llm_response': (
                'Ready to start LLM-powered Snake Web Game...\n\n'
                'The AI will analyze the game state and make strategic moves '
                'to maximize score while avoiding collisions.\n\n'
                'For full LLM functionality with complete reasoning and session '
                'management, use scripts/main_web.py.'
            ),
            'planned_moves': [],
            'thinking_process': 'Initializing AI reasoning for web interface...',
            'status': 'demo',
            'colors': build_color_map(as_list=True)
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM web game controls.
        
        Args:
            data: Control data from client
            
        Returns:
            Response dictionary with operation result
            
        Educational Value: Shows LLM-specific control handling with enhanced context
        Extension Pattern: Extensions can copy this for their LLM control logic
        
        Note: Full implementation would delegate to GameManager methods
        for actual LLM gameplay, session management, and statistics tracking.
        """
        action = data.get('action', '')
        
        if action == 'start':
            print_log("LLM web game start (demo mode)")
            return {
                'action': 'start', 
                'status': 'started',
                'app_type': 'llm_web_game_app',
                'message': 'LLM web demo mode started. Use scripts/main_web.py for full functionality.'
            }
        elif action == 'pause':
            print_log("LLM web game pause (demo mode)")
            return {
                'action': 'pause', 
                'status': 'paused',
                'app_type': 'llm_web_game_app',
                'message': 'LLM web demo mode paused.'
            }
        elif action == 'reset':
            print_log("LLM web game reset (demo mode)")
            return {
                'action': 'reset', 
                'status': 'reset',
                'app_type': 'llm_web_game_app',
                'message': 'LLM web demo mode reset.'
            }
        
        return {
            'status': 'error', 
            'app_type': 'llm_web_game_app',
            'message': f'Unknown action: {action}. Demo mode supports: start, pause, reset.'
        }
    
    def get_template_name(self) -> str:
        """Get template name for LLM web game interface.
        
        Returns:
            Template filename for LLM web game interface
            
        Educational Value: Shows LLM template selection patterns with enhanced naming
        Extension Pattern: Extensions can override for custom LLM templates
        """
        return 'main.html'

