"""
MVC Architecture Integration Example
--------------------

Example demonstrating how to use the new MVC web architecture
for different Snake game modes.

This example shows:
- Factory pattern usage for creating components
- Controller registration and routing
- Observer pattern for real-time updates
- Template rendering and view customization

Usage Examples:
    # Human game mode
    python -m web.examples.mvc_integration_example --mode human
    
    # LLM game mode  
    python -m web.examples.mvc_integration_example --mode llm
    
    # Replay mode
    python -m web.examples.mvc_integration_example --mode replay
"""

import sys
from pathlib import Path
import logging
import argparse
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MVC components
from web.factories import create_web_application
from web.models import Observer, LoggingObserver

# Import core game components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExampleStatsObserver(Observer):
    """
    Example observer that tracks game statistics in real-time.
    
    Demonstrates how observers can be used to collect metrics
    and provide live updates to connected clients.
    """
    
    def __init__(self):
        """Initialize statistics observer."""
        self.stats = {
            'total_events': 0,
            'moves_count': 0,
            'apples_eaten': 0,
            'games_completed': 0,
            'average_score': 0.0,
            'last_event_time': None
        }
        self.game_scores = []
    
    def on_game_event(self, event):
        """Handle game events and update statistics."""
        from web.models.events import EventType
        
        self.stats['total_events'] += 1
        self.stats['last_event_time'] = event.timestamp.isoformat()
        
        if event.event_type == EventType.MOVE:
            self.stats['moves_count'] += 1
            
        elif event.event_type == EventType.APPLE_EATEN:
            self.stats['apples_eaten'] += 1
            
        elif event.event_type == EventType.GAME_OVER:
            self.stats['games_completed'] += 1
            if hasattr(event, 'final_score'):
                self.game_scores.append(event.final_score)
                self.stats['average_score'] = sum(self.game_scores) / len(self.game_scores)
        
        logger.info(f"Stats updated: {self.stats}")
    
    def get_statistics(self):
        """Get current statistics."""
        return self.stats.copy()


def create_human_game_example():
    """
    Create human game web application example.
    
    Demonstrates:
    - Factory pattern for creating MVC components
    - Observer registration for real-time stats
    - Template customization for human interface
    """
    logger.info("Creating human game MVC example...")
    
    # Create mock game controller for demonstration
    class MockHumanGameController:
        def __init__(self):
            self.grid_size = 15
            self.score = 0
            self.steps = 0
            self.game_over = False
            self.snake_positions = [(7, 7), (7, 6), (7, 5)]
            self.apple_position = (10, 10)
            self.current_direction = "UP"
            self.start_time = time.time()
        
        def reset(self):
            self.__init__()
        
        def make_move(self, direction):
            # Mock move implementation
            self.steps += 1
            if self.steps % 5 == 0:  # Mock apple eating
                self.score += 1
                return True, True  # game_active, apple_eaten
            return True, False
    
    # Create Flask app using factory
    game_controller = MockHumanGameController()
    app, controller = create_web_application(
        game_controller=game_controller,
        game_mode="human",
        template_folder=str(project_root / "web" / "templates"),
        static_folder=str(project_root / "web" / "static")
    )
    
    # Add custom observers
    stats_observer = ExampleStatsObserver()
    logging_observer = LoggingObserver(logging.INFO)
    
    controller.model_manager.add_observer(stats_observer)
    controller.model_manager.add_observer(logging_observer)
    
    # Add custom route for statistics
    @app.route('/api/stats')
    def get_stats():
        return {
            "status": "success",
            "statistics": stats_observer.get_statistics(),
            "controller_stats": controller.get_performance_stats()
        }
    
    logger.info("Human game MVC application created successfully")
    return app, controller, stats_observer


def create_llm_game_example():
    """
    Create LLM game web application example.
    
    Demonstrates:
    - LLM-specific controller configuration
    - Different template and styling
    - Custom observer for LLM metrics
    """
    logger.info("Creating LLM game MVC example...")
    
    # Create mock LLM game controller
    class MockLLMGameController:
        def __init__(self):
            self.grid_size = 20
            self.score = 0
            self.steps = 0
            self.game_over = False
            self.snake_positions = [(10, 10), (10, 9), (10, 8)]
            self.apple_position = (15, 15)
            self.current_direction = "RIGHT"
            self.start_time = time.time()
            self.llm_response_time = 0.5  # Mock LLM response time
        
        def reset(self):
            self.__init__()
        
        def get_llm_decision(self, game_state):
            # Mock LLM decision making
            time.sleep(0.1)  # Simulate LLM processing
            return "UP"  # Mock decision
    
    # Create Flask app for LLM mode
    game_controller = MockLLMGameController()
    app, controller = create_web_application(
        game_controller=game_controller,
        game_mode="llm",
        template_folder=str(project_root / "web" / "templates"),
        static_folder=str(project_root / "web" / "static")
    )
    
    # Add LLM-specific observers
    class LLMMetricsObserver(Observer):
        def __init__(self):
            self.llm_metrics = {
                'decisions_made': 0,
                'avg_decision_time': 0.0,
                'successful_moves': 0,
                'decision_times': []
            }
        
        def on_game_event(self, event):
            from web.models.events import EventType
            if event.event_type == EventType.MOVE:
                self.llm_metrics['decisions_made'] += 1
                # Mock decision time tracking
                decision_time = 0.3  # Mock time
                self.llm_metrics['decision_times'].append(decision_time)
                self.llm_metrics['avg_decision_time'] = (
                    sum(self.llm_metrics['decision_times']) / 
                    len(self.llm_metrics['decision_times'])
                )
    
    llm_observer = LLMMetricsObserver()
    controller.model_manager.add_observer(llm_observer)
    
    # Add LLM-specific routes
    @app.route('/api/llm-metrics')
    def get_llm_metrics():
        return {
            "status": "success",
            "llm_metrics": llm_observer.llm_metrics,
            "model_performance": controller.get_performance_stats()
        }
    
    logger.info("LLM game MVC application created successfully")
    return app, controller, llm_observer


def create_replay_example():
    """
    Create replay web application example.
    
    Demonstrates:
    - Replay-specific controller and model
    - Navigation controls
    - Timeline visualization
    """
    logger.info("Creating replay MVC example...")
    
    # Create mock replay engine
    class MockReplayEngine:
        def __init__(self):
            self.game_number = 1
            self.current_step = 0
            self.total_steps = 100
            self.paused = False
            self.grid_size = 15
            self.current_score = 25
            self.is_finished = False
        
        def restart_game(self):
            self.current_step = 0
            self.is_finished = False
        
        def get_current_snake_positions(self):
            return [(8, 8), (8, 7), (8, 6), (8, 5)]
        
        def get_current_apple_position(self):
            return (12, 12)
    
    # Create Flask app for replay mode
    replay_engine = MockReplayEngine()
    app, controller = create_web_application(
        replay_engine=replay_engine,
        template_folder=str(project_root / "web" / "templates"),
        static_folder=str(project_root / "web" / "static")
    )
    
    # Add replay-specific routes
    @app.route('/api/replay/navigate', methods=['POST'])
    def navigate_replay():
        from flask import request
        data = request.get_json() or {}
        action = data.get('action')
        
        if action == 'next_game':
            replay_engine.game_number += 1
            return {"status": "success", "game_number": replay_engine.game_number}
        elif action == 'prev_game' and replay_engine.game_number > 1:
            replay_engine.game_number -= 1
            return {"status": "success", "game_number": replay_engine.game_number}
        elif action == 'toggle_pause':
            replay_engine.paused = not replay_engine.paused
            return {"status": "success", "paused": replay_engine.paused}
        
        return {"status": "error", "message": "Invalid action"}
    
    logger.info("Replay MVC application created successfully")
    return app, controller


def run_example_server(app, controller, host='localhost', port=5000):
    """
    Run the example Flask application.
    
    Args:
        app: Flask application instance
        controller: Web controller instance
        host: Server host
        port: Server port
    """
    logger.info(f"Starting example server at http://{host}:{port}")
    
    # Print MVC architecture summary
    print("\n" + "="*60)
    print("MVC ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Controller: {controller.__class__.__name__}")
    print(f"Model: {controller.model_manager.__class__.__name__}")
    print(f"View Renderer: {controller.view_renderer.__class__.__name__}")
    print(f"Observers: {controller.model_manager.get_observer_count()}")
    print(f"Request Filters: {len(controller.request_filters)}")
    print("="*60)
    print()
    
    # Start server
    try:
        app.run(host=host, port=port, debug=True, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


def main():
    """Main example runner."""
    parser = argparse.ArgumentParser(description="MVC Architecture Examples")
    parser.add_argument(
        '--mode', 
        choices=['human', 'llm', 'replay'],
        default='human',
        help='Game mode to demonstrate'
    )
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    
    args = parser.parse_args()
    
    print("\n🐍 Snake Game MVC Architecture Example")
    print(f"Mode: {args.mode.upper()}")
    print(f"URL: http://{args.host}:{args.port}")
    print()
    
    try:
        if args.mode == 'human':
            app, controller, stats_observer = create_human_game_example()
            print("Features demonstrated:")
            print("  ✓ Human input controller")
            print("  ✓ Real-time statistics observer")
            print("  ✓ Input validation and rate limiting")
            print("  ✓ Template Method pattern in action")
            
        elif args.mode == 'llm':
            app, controller, llm_observer = create_llm_game_example()
            print("Features demonstrated:")
            print("  ✓ LLM decision controller")
            print("  ✓ LLM metrics observer")
            print("  ✓ Asynchronous AI processing")
            print("  ✓ Strategy pattern for AI modes")
            
        elif args.mode == 'replay':
            app, controller = create_replay_example()
            print("Features demonstrated:")
            print("  ✓ Replay navigation controller")
            print("  ✓ Timeline state provider")
            print("  ✓ Historical data viewing")
            print("  ✓ Observer pattern for state changes")
        
        print("\nAPI Endpoints available:")
        print("  GET  /                 - Main game interface")
        print("  GET  /api/state        - Current game state") 
        print("  POST /api/control      - Game commands")
        print("  POST /api/reset        - Reset game")
        print("  GET  /api/health       - System health")
        
        if args.mode in ['human', 'llm']:
            print("  GET  /api/stats        - Game statistics")
        if args.mode == 'llm':
            print("  GET  /api/llm-metrics  - LLM performance")
        if args.mode == 'replay':
            print("  POST /api/replay/navigate - Replay controls")
        
        print()
        
        # Run the server
        run_example_server(app, controller, args.host, args.port)
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 