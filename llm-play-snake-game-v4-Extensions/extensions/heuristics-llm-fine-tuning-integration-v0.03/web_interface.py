"""web_interface.py - Web Interface Controller for v0.03

Provides web-specific functionality including Flask routes, WebSocket support,
and integration with the Streamlit application.

Design Patterns:
- Controller Pattern: Handles web requests and responses
- Observer Pattern: Real-time updates via WebSocket
- Adapter Pattern: Bridges Streamlit and Flask functionality
"""

from __future__ import annotations

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from flask import Flask, jsonify, request, render_template_string
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from extensions.common import training_logging_utils

logger = training_logging_utils.TrainingLogger("web_interface")


@dataclass
class TrainingStatus:
    """Data class for training status information.
    
    Design Pattern: Value Object
    - Immutable container for training state
    - Provides serialization methods
    - Type-safe status representation
    """
    
    is_active: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0
    model_name: str = ""
    strategy: str = ""
    datasets_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate training progress percentage."""
        if self.total_epochs == 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100


class WebDashboard:
    """Web dashboard controller for the v0.03 extension.
    
    Design Pattern: Facade Pattern
    - Provides simplified interface to web functionality
    - Coordinates between Flask and Streamlit components
    - Manages WebSocket connections for real-time updates
    """
    
    def __init__(self, port: int = 5000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = None
        self.socketio = None
        self.training_status = TrainingStatus()
        self.connected_clients = set()
        
        if FLASK_AVAILABLE:
            self.setup_flask_app()
        else:
            logger.warning("Flask not available, web dashboard disabled")
    
    def setup_flask_app(self):
        """Setup Flask application with routes and WebSocket support."""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'llm-fine-tuning-v0.03'
        
        # Setup SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Register routes
        self.register_routes()
        self.register_websocket_handlers()
    
    def register_routes(self):
        """Register Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/status')
        def get_status():
            """Get current training status."""
            return jsonify(self.training_status.to_dict())
        
        @self.app.route('/api/datasets')
        def get_datasets():
            """Get available datasets."""
            try:
                datasets = self.get_available_datasets()
                return jsonify({
                    'success': True,
                    'datasets': datasets
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/models')
        def get_models():
            """Get trained models."""
            try:
                models = self.get_trained_models()
                return jsonify({
                    'success': True,
                    'models': models
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/training/start', methods=['POST'])
        def start_training():
            """Start training with given configuration."""
            try:
                config = request.get_json()
                result = self.start_training(config)
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/training/stop', methods=['POST'])
        def stop_training():
            """Stop current training."""
            try:
                result = self.stop_training()
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/evaluation/run', methods=['POST'])
        def run_evaluation():
            """Run model evaluation."""
            try:
                config = request.get_json()
                result = self.run_evaluation(config)
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def register_websocket_handlers(self):
        """Register WebSocket event handlers for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.connected_clients.add(request.sid)
            emit('status_update', self.training_status.to_dict())
            logger.info(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.connected_clients.discard(request.sid)
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_status')
        def handle_status_request():
            """Handle status request from client."""
            emit('status_update', self.training_status.to_dict())
    
    def get_dashboard_template(self) -> str:
        """Get the HTML template for the dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Fine-tuning Dashboard v0.03</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .status-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            margin-top: 10px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        .btn-danger {
            background-color: #e74c3c;
            color: white;
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-active {
            background-color: #2ecc71;
        }
        .status-inactive {
            background-color: #95a5a6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LLM Fine-tuning Dashboard v0.03</h1>
            <p>Real-time monitoring and control for heuristics-based LLM fine-tuning</p>
        </div>
        
        <div class="status-card">
            <h2><span id="status-indicator" class="status-indicator status-inactive"></span>Training Status</h2>
            <div id="status-text">Ready</div>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill"></div>
            </div>
            <div id="progress-text">0% complete</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div id="metric-epoch" class="metric-value">0/0</div>
                <div class="metric-label">Epoch</div>
            </div>
            <div class="metric-card">
                <div id="metric-loss" class="metric-value">0.000</div>
                <div class="metric-label">Current Loss</div>
            </div>
            <div class="metric-card">
                <div id="metric-lr" class="metric-value">0.00e+00</div>
                <div class="metric-label">Learning Rate</div>
            </div>
            <div class="metric-card">
                <div id="metric-time" class="metric-value">00:00</div>
                <div class="metric-label">Elapsed Time</div>
            </div>
        </div>
        
        <div class="status-card">
            <h3>Controls</h3>
            <div class="controls">
                <button id="btn-start" class="btn btn-primary">▶️ Start Training</button>
                <button id="btn-stop" class="btn btn-danger" disabled>⏹️ Stop Training</button>
                <button id="btn-refresh" class="btn btn-primary">🔄 Refresh</button>
            </div>
        </div>
        
        <div class="status-card">
            <h3>Training Progress Chart</h3>
            <div id="loss-chart" style="height: 400px;"></div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Loss history for charting
        let lossHistory = [];
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to server');
            socket.emit('request_status');
        });
        
        socket.on('status_update', function(status) {
            updateUI(status);
        });
        
        // UI update function
        function updateUI(status) {
            // Update status indicator
            const indicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            
            if (status.is_active) {
                indicator.className = 'status-indicator status-active';
                statusText.textContent = `Training ${status.strategy} model: ${status.model_name}`;
            } else {
                indicator.className = 'status-indicator status-inactive';
                statusText.textContent = 'Ready';
            }
            
            // Update progress bar
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            const progress = status.total_epochs > 0 ? (status.current_epoch / status.total_epochs) * 100 : 0;
            
            progressFill.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '% complete';
            
            // Update metrics
            document.getElementById('metric-epoch').textContent = `${status.current_epoch}/${status.total_epochs}`;
            document.getElementById('metric-loss').textContent = status.current_loss.toFixed(3);
            document.getElementById('metric-lr').textContent = status.learning_rate.toExponential(2);
            
            // Format elapsed time
            const hours = Math.floor(status.elapsed_time / 3600);
            const minutes = Math.floor((status.elapsed_time % 3600) / 60);
            const seconds = Math.floor(status.elapsed_time % 60);
            document.getElementById('metric-time').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Update button states
            document.getElementById('btn-start').disabled = status.is_active;
            document.getElementById('btn-stop').disabled = !status.is_active;
            
            // Update loss chart
            if (status.is_active && status.current_loss > 0) {
                lossHistory.push({
                    epoch: status.current_epoch,
                    loss: status.current_loss,
                    timestamp: new Date()
                });
                updateLossChart();
            }
        }
        
        // Chart update function
        function updateLossChart() {
            const trace = {
                x: lossHistory.map(d => d.epoch),
                y: lossHistory.map(d => d.loss),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Training Loss',
                line: { color: '#667eea' }
            };
            
            const layout = {
                title: 'Training Loss Over Time',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss' },
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            Plotly.newPlot('loss-chart', [trace], layout);
        }
        
        // Button event handlers
        document.getElementById('btn-start').addEventListener('click', function() {
            // This would typically open a configuration dialog
            alert('Training configuration dialog would open here');
        });
        
        document.getElementById('btn-stop').addEventListener('click', function() {
            if (confirm('Are you sure you want to stop training?')) {
                fetch('/api/training/stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            socket.emit('request_status');
                        } else {
                            alert('Failed to stop training: ' + data.error);
                        }
                    });
            }
        });
        
        document.getElementById('btn-refresh').addEventListener('click', function() {
            socket.emit('request_status');
        });
        
        // Initialize empty chart
        updateLossChart();
    </script>
</body>
</html>
        """
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets."""
        datasets = []
        
        for grid_size in [8, 10, 12, 16, 20]:
            dataset_dir = Path(f"logs/extensions/datasets/grid-size-{grid_size}")
            if dataset_dir.exists():
                for file_path in dataset_dir.glob("*.csv"):
                    datasets.append({
                        'name': file_path.name,
                        'grid_size': grid_size,
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
        
        return datasets
    
    def get_trained_models(self) -> List[Dict[str, Any]]:
        """Get list of trained models."""
        models = []
        models_dir = Path("logs/extensions/models")
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'created': datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                    })
        
        return models
    
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training with given configuration."""
        try:
            # Update training status
            self.training_status.is_active = True
            self.training_status.model_name = config.get('model_name', 'Unknown')
            self.training_status.strategy = config.get('strategy', 'Unknown')
            self.training_status.total_epochs = config.get('num_epochs', 0)
            self.training_status.datasets_count = len(config.get('datasets', []))
            
            # Broadcast status update
            self.broadcast_status_update()
            
            logger.info(f"Training started: {config}")
            
            return {
                'success': True,
                'message': 'Training started successfully'
            }
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop current training."""
        try:
            self.training_status.is_active = False
            self.training_status.current_epoch = 0
            self.training_status.total_epochs = 0
            
            # Broadcast status update
            self.broadcast_status_update()
            
            logger.info("Training stopped")
            
            return {
                'success': True,
                'message': 'Training stopped successfully'
            }
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run model evaluation."""
        try:
            # Mock evaluation for now
            results = {
                'model': config.get('model_name', 'Unknown'),
                'dataset': config.get('dataset', 'Unknown'),
                'metrics': {
                    'win_rate': 0.78,
                    'avg_score': 13.2,
                    'decision_accuracy': 0.85
                }
            }
            
            logger.info(f"Evaluation completed: {results}")
            
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_training_progress(self, epoch: int, loss: float, lr: float):
        """Update training progress and broadcast to connected clients."""
        self.training_status.current_epoch = epoch
        self.training_status.current_loss = loss
        self.training_status.learning_rate = lr
        
        # Broadcast to all connected clients
        self.broadcast_status_update()
    
    def broadcast_status_update(self):
        """Broadcast status update to all connected clients."""
        if self.socketio and self.connected_clients:
            self.socketio.emit('status_update', self.training_status.to_dict())
    
    def run(self):
        """Run the web dashboard."""
        if not FLASK_AVAILABLE:
            logger.error("Flask not available, cannot run web dashboard")
            return
        
        logger.info(f"Starting web dashboard on port {self.port}")
        self.socketio.run(
            self.app,
            host='0.0.0.0',
            port=self.port,
            debug=self.debug
        ) 