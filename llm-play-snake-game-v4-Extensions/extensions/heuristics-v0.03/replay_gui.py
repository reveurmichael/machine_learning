#!/usr/bin/env python3
"""
Heuristic Replay GUI
===================

Web-based replay interface for heuristic algorithms.
Extends Task-0's web replay infrastructure with heuristic-specific features.

Features:
- Algorithm-aware display
- Performance metrics visualization
- Pathfinding progress tracking
- Support for all 7 heuristic algorithms
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from typing import Any, Dict
from flask import Flask, jsonify

# Import Task-0 web infrastructure
from web.views.template_renderer import TemplateRenderer
from utils.web_utils import build_state_dict

# Import heuristic replay engine
from replay_engine import HeuristicReplayEngine


class HeuristicReplayGUI:
    """
    Web-based GUI for heuristic algorithm replay.
    
    Extends Task-0's web replay infrastructure with heuristic-specific features:
    - Algorithm identification and display
    - Performance metrics visualization
    - Pathfinding progress tracking
    - Enhanced user interface for algorithm comparison
    
    Design Pattern: Adapter Pattern
    - Adapts Task-0 web infrastructure for heuristic replay needs
    - Maintains compatibility with existing web replay framework
    
    Extensive Reuse:
    - Leverages Task-0 Flask application structure
    - Reuses web controllers and template rendering
    - Extends existing game state model
    - Uses Task-0 web utilities and constants
    """
    
    def __init__(self, log_dir: str, game_number: int, host: str = "127.0.0.1", port: int = 5000):
        """Initialize heuristic replay GUI."""
        self.log_dir = log_dir
        self.game_number = game_number
        self.host = host
        self.port = port
        
        # Initialize heuristic replay engine
        self.replay_engine = HeuristicReplayEngine(
            log_dir=log_dir,
            pause_between_moves=1.0,
            auto_advance=False,
            use_gui=False  # Disable PyGame GUI for web mode
        )
        
        # Load game data
        self.game_data = self.replay_engine.load_game_data(game_number)
        
        # Initialize Flask app using Task-0 patterns
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Algorithm display information
        self.algorithm_info = self.replay_engine.get_algorithm_info()
    
    def _setup_routes(self) -> None:
        """Setup Flask routes for heuristic replay."""
        
        @self.app.route('/')
        def index():
            """Main replay page."""
            return self._render_replay_page()
        
        @self.app.route('/api/game_state')
        def get_game_state():
            """Get current game state as JSON."""
            return jsonify(self._build_game_state())
        
        @self.app.route('/api/algorithm_info')
        def get_algorithm_info():
            """Get algorithm information."""
            return jsonify(self.algorithm_info)
        
        @self.app.route('/api/performance_metrics')
        def get_performance_metrics():
            """Get performance metrics."""
            return jsonify(self.replay_engine.performance_metrics)
        
        @self.app.route('/api/control/<action>')
        def control_replay(action: str):
            """Control replay playback."""
            if action == 'play':
                self.replay_engine.paused = False
            elif action == 'pause':
                self.replay_engine.paused = True
            elif action == 'reset':
                self.replay_engine.reset()
            elif action == 'next':
                self.replay_engine.update()
            
            return jsonify({'status': 'success', 'action': action})
    
    def _render_replay_page(self) -> str:
        """
        Render the main replay page.
        
        Uses Task-0 template rendering infrastructure with heuristic-specific data.
        """
        try:
            # Use Task-0 template renderer
            renderer = TemplateRenderer()
            
            # Prepare template context with heuristic-specific data
            context = {
                'game_title': f"Heuristic Replay - {self.algorithm_info['algorithm_display_name']}",
                'algorithm_name': self.algorithm_info['algorithm_name'],
                'algorithm_display_name': self.algorithm_info['algorithm_display_name'],
                'game_number': self.game_number,
                'log_dir': Path(self.log_dir).name,
                'performance_metrics': self.replay_engine.performance_metrics,
                'pathfinding_info': self.replay_engine.pathfinding_info,
                'game_state': self._build_game_state(),
                'controls_enabled': True,
                'heuristic_mode': True
            }
            
            # Render using Task-0 template (with fallback)
            try:
                return renderer.render_template('heuristic_replay.html', context)
            except:
                # Fallback to basic HTML if heuristic template doesn't exist
                return self._render_basic_replay_page(context)
                
        except Exception as e:
            return f"<h1>Error rendering replay page: {e}</h1>"
    
    def _render_basic_replay_page(self, context: Dict[str, Any]) -> str:
        """Render basic replay page as fallback."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{context['game_title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .metric {{ background: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center; }}
                .controls {{ margin: 20px 0; }}
                .control-btn {{ margin: 5px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                .game-area {{ border: 2px solid #333; display: inline-block; }}
                .status {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 {context['algorithm_display_name']} Replay</h1>
                <p><strong>Game:</strong> {context['game_number']} | <strong>Log:</strong> {context['log_dir']}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Score</h3>
                    <p>{context['performance_metrics']['score']}</p>
                </div>
                <div class="metric">
                    <h3>Steps</h3>
                    <p>{context['performance_metrics']['steps']}</p>
                </div>
                <div class="metric">
                    <h3>Rounds</h3>
                    <p>{context['performance_metrics']['round_count']}</p>
                </div>
                <div class="metric">
                    <h3>Score/Step</h3>
                    <p>{context['performance_metrics']['score_per_step']:.3f}</p>
                </div>
            </div>
            
            <div class="controls">
                <button class="control-btn" onclick="controlReplay('play')">▶️ Play</button>
                <button class="control-btn" onclick="controlReplay('pause')">⏸️ Pause</button>
                <button class="control-btn" onclick="controlReplay('reset')">🔄 Reset</button>
                <button class="control-btn" onclick="controlReplay('next')">⏭️ Next</button>
            </div>
            
            <div class="game-area">
                <canvas id="gameCanvas" width="400" height="400"></canvas>
            </div>
            
            <div class="status">
                <h3>Algorithm Information</h3>
                <p><strong>Type:</strong> {context['algorithm_display_name']}</p>
                <p><strong>Total Moves:</strong> {context['pathfinding_info']['total_moves']}</p>
                <p><strong>Apples Collected:</strong> {len(context['pathfinding_info']['apple_positions'])}</p>
            </div>
            
            <script>
                function controlReplay(action) {{
                    fetch(`/api/control/${{action}}`)
                        .then(response => response.json())
                        .then(data => console.log('Control:', data));
                }}
                
                function updateGameState() {{
                    fetch('/api/game_state')
                        .then(response => response.json())
                        .then(data => {{
                            // Update game visualization
                            drawGame(data);
                        }});
                }}
                
                function drawGame(gameState) {{
                    const canvas = document.getElementById('gameCanvas');
                    const ctx = canvas.getContext('2d');
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw grid
                    const cellSize = canvas.width / gameState.grid_size;
                    
                    // Draw snake
                    ctx.fillStyle = '#4CAF50';
                    gameState.snake_positions.forEach(pos => {{
                        ctx.fillRect(pos[0] * cellSize, (gameState.grid_size - 1 - pos[1]) * cellSize, cellSize, cellSize);
                    }});
                    
                    // Draw apple
                    ctx.fillStyle = '#F44336';
                    const apple = gameState.apple_position;
                    ctx.fillRect(apple[0] * cellSize, (gameState.grid_size - 1 - apple[1]) * cellSize, cellSize, cellSize);
                    
                    // Draw grid lines
                    ctx.strokeStyle = '#ddd';
                    for (let i = 0; i <= gameState.grid_size; i++) {{
                        ctx.beginPath();
                        ctx.moveTo(i * cellSize, 0);
                        ctx.lineTo(i * cellSize, canvas.height);
                        ctx.stroke();
                        
                        ctx.beginPath();
                        ctx.moveTo(0, i * cellSize);
                        ctx.lineTo(canvas.width, i * cellSize);
                        ctx.stroke();
                    }}
                }}
                
                // Update game state periodically
                setInterval(updateGameState, 1000);
                updateGameState();
            </script>
        </body>
        </html>
        """
    
    def _build_game_state(self) -> Dict[str, Any]:
        """
        Build current game state for web interface.
        
        Uses Task-0 web utilities with heuristic-specific extensions.
        """
        try:
            # Use Task-0 web utility for base state
            base_state = build_state_dict(
                snake_positions=self.replay_engine.snake_positions.tolist(),
                apple_position=self.replay_engine.apple_position.tolist(),
                score=self.replay_engine.performance_metrics['score'],
                steps=self.replay_engine.performance_metrics['steps'],
                grid_size=self.replay_engine.grid_size
            )
            
            # Add heuristic-specific extensions
            heuristic_extensions = {
                'algorithm_name': self.algorithm_info['algorithm_name'],
                'algorithm_display_name': self.algorithm_info['algorithm_display_name'],
                'performance_metrics': self.replay_engine.performance_metrics,
                'pathfinding_info': self.replay_engine.pathfinding_info,
                'replay_progress': {
                    'current_move': self.replay_engine.move_index,
                    'total_moves': len(self.replay_engine.moves),
                    'percentage': (self.replay_engine.move_index / max(len(self.replay_engine.moves), 1)) * 100
                },
                'game_active': self.replay_engine.running,
                'paused': self.replay_engine.paused
            }
            
            # Merge base state with heuristic extensions
            base_state.update(heuristic_extensions)
            return base_state
            
        except Exception as e:
            # Fallback minimal state
            return {
                'error': str(e),
                'algorithm_name': self.algorithm_info.get('algorithm_name', 'Unknown'),
                'snake_positions': [],
                'apple_position': [0, 0],
                'score': 0,
                'steps': 0,
                'grid_size': 10
            }
    
    def run(self) -> None:
        """
        Start the web replay server.
        
        Uses Flask development server with heuristic-specific configuration.
        """
        print("🌐 Starting Heuristic Web Replay")
        print(f"🧠 Algorithm: {self.algorithm_info['algorithm_display_name']}")
        print(f"🎮 Game: {self.game_number}")
        print(f"📁 Log Directory: {self.log_dir}")
        print(f"🌍 Server: http://{self.host}:{self.port}")
        print("⚠️  Press Ctrl+C to stop")
        
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Web replay server stopped")
        except Exception as e:
            print(f"❌ Server error: {e}")


def create_heuristic_replay_gui(
    log_dir: str,
    game_number: int,
    host: str = "127.0.0.1",
    port: int = 5000
) -> HeuristicReplayGUI:
    """
    Factory function for creating heuristic replay GUIs.
    
    Design Pattern: Factory Pattern
    - Provides consistent interface for creating web replay interfaces
    - Encapsulates configuration and initialization logic
    
    Args:
        log_dir: Directory containing heuristic game logs
        game_number: Game number to replay
        host: Web server host address
        port: Web server port number
        
    Returns:
        Configured HeuristicReplayGUI instance
    """
    return HeuristicReplayGUI(
        log_dir=log_dir,
        game_number=game_number,
        host=host,
        port=port
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Heuristic Web Replay")
    parser.add_argument("--log-dir", required=True, help="Log directory path")
    parser.add_argument("--game-number", type=int, default=1, help="Game number to replay")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run web replay GUI
    gui = create_heuristic_replay_gui(
        log_dir=args.log_dir,
        game_number=args.game_number,
        host=args.host,
        port=args.port
    )
    
    gui.run() 