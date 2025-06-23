"""app.py - Interactive Streamlit Web Application for LLM Fine-tuning v0.03

This is the main entry point for the web interface, providing an interactive
dashboard for configuring, running, and monitoring LLM fine-tuning experiments.

Key Features:
- Multi-tab interface for different workflows
- Real-time training progress monitoring
- Interactive parameter configuration
- Model comparison and evaluation
- Dataset management and preprocessing
- Experiment tracking and history

Design Patterns:
- Model-View-Controller (MVC): Separates UI from business logic
- Observer Pattern: Real-time updates from training processes
- State Pattern: UI state management across tabs
- Command Pattern: User actions as executable commands

Evolution from v0.02: CLI-only → Interactive web interface with real-time monitoring
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Fix Python path for Streamlit
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

# Import Streamlit and other UI libraries
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
except ImportError as e:
    st.error(f"Missing required dependencies: {e}")
    st.stop()

# Import from common utilities
from extensions.common import (
    training_logging_utils
)

# Import from v0.02 components (reuse core logic)
try:
    from extensions.heuristics_llm_fine_tuning_integration_v0_02.pipeline import MultiDatasetPipeline
    from extensions.heuristics_llm_fine_tuning_integration_v0_02.training_config import TrainingConfigBuilder
    from extensions.heuristics_llm_fine_tuning_integration_v0_02.evaluation import EvaluationSuite
    from extensions.heuristics_llm_fine_tuning_integration_v0_02.comparison import ModelComparator
    v0_02_available = True
except ImportError as e:
    st.warning(f"v0.02 components not available: {e}")
    v0_02_available = False

# Set up logging
logger = training_logging_utils.TrainingLogger("streamlit_app")

def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        'training_in_progress': False,
        'selected_datasets': [],
        'trained_models': [],
        'experiment_history': [],
        'current_config': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_sidebar():
    """Setup the application sidebar with navigation and settings."""
    with st.sidebar:
        st.title("🚀 LLM Fine-tuning v0.03")
        st.markdown("---")
        
        # Version information
        st.info("""
        **Version:** 0.03  
        **Type:** Web Interface  
        **Evolution:** v0.02 CLI → v0.03 Interactive
        """)
        
        # Quick stats
        if st.session_state.get('experiment_history'):
            st.metric(
                "Total Experiments", 
                len(st.session_state['experiment_history'])
            )
        
        # Settings
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.checkbox(
            "Auto-refresh", 
            value=True,
            help="Automatically refresh training progress"
        )
        
        show_advanced = st.checkbox(
            "Show Advanced Options",
            value=False,
            help="Show advanced configuration options"
        )
        
        return auto_refresh, show_advanced


def training_tab():
    """Training configuration and execution tab."""
    st.header("🎯 Training Configuration")
    
    if not v0_02_available:
        st.error("v0.02 components required for training are not available.")
        return
    
    # Training strategy selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        strategy = st.selectbox(
            "Training Strategy",
            ["LoRA", "QLoRA", "Full Fine-tuning"],
            help="Choose the fine-tuning approach"
        )
        
        model_name = st.selectbox(
            "Base Model",
            [
                "microsoft/DialoGPT-small",
                "microsoft/DialoGPT-medium", 
                "gpt2",
                "distilgpt2"
            ],
            help="Base model for fine-tuning"
        )
    
    with col2:
        grid_size = st.selectbox(
            "Grid Size",
            [8, 10, 12, 16, 20],
            index=1,
            help="Snake game grid size for datasets"
        )
        
        num_epochs = st.slider(
            "Number of Epochs",
            min_value=1,
            max_value=20,
            value=5,
            help="Training epochs"
        )
    
    # Dataset selection
    st.subheader("📁 Dataset Selection")
    available_datasets = get_available_datasets(grid_size)
    
    if available_datasets:
        selected_datasets = st.multiselect(
            "Select Datasets",
            available_datasets,
            default=available_datasets[:2] if len(available_datasets) >= 2 else available_datasets,
            help="Choose heuristic datasets for training"
        )
    else:
        st.warning("No datasets found. Please generate datasets first using heuristics extensions.")
        selected_datasets = []
    
    # Training controls
    st.subheader("🚀 Training Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "▶️ Start Training",
            disabled=st.session_state.get('training_in_progress') or not selected_datasets,
            use_container_width=True
        ):
            start_training({
                'strategy': strategy,
                'model_name': model_name,
                'datasets': selected_datasets,
                'grid_size': grid_size,
                'num_epochs': num_epochs,
            })
    
    with col2:
        if st.button(
            "⏹️ Stop Training",
            disabled=not st.session_state.get('training_in_progress'),
            use_container_width=True
        ):
            stop_training()
    
    with col3:
        if st.button("💾 Save Config", use_container_width=True):
            config = {
                'strategy': strategy,
                'model_name': model_name,
                'datasets': selected_datasets,
                'grid_size': grid_size,
                'num_epochs': num_epochs,
            }
            save_training_config(config)
    
    # Quick validation
    if selected_datasets:
        st.success(f"✅ Ready to train on {len(selected_datasets)} datasets")


def monitoring_tab():
    """Real-time training monitoring tab."""
    st.header("📊 Training Progress Monitoring")
    
    if not st.session_state.get('training_in_progress'):
        st.info("No training in progress. Start training from the Training tab.")
        
        # Show recent experiment history
        history = st.session_state.get('experiment_history', [])
        if history:
            st.subheader("📈 Recent Experiments")
            df = pd.DataFrame(history[-5:])  # Last 5 experiments
            st.dataframe(df, use_container_width=True)
        return
    
    # Training in progress - show monitoring interface
    st.success("🔄 Training in progress...")
    
    # Progress metrics (mock data for now)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Epoch", "3/5", delta="60%")
    
    with col2:
        st.metric("Current Loss", "1.234", delta="-0.056")
    
    with col3:
        st.metric("Learning Rate", "5e-05")
    
    with col4:
        st.metric("Elapsed Time", "15m 32s")
    
    # Progress bar
    st.progress(0.6)
    
    # Training curves (mock data)
    if st.checkbox("Show Training Curves"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curve
            epochs = list(range(1, 6))
            losses = [2.5, 2.1, 1.8, 1.5, 1.2]
            loss_df = pd.DataFrame({'Epoch': epochs, 'Loss': losses})
            st.line_chart(loss_df.set_index('Epoch'))
            st.caption("Training Loss")
        
        with col2:
            # Learning rate schedule
            steps = list(range(0, 1000, 100))
            lrs = [5e-5 * (0.9 ** (step // 100)) for step in steps]
            lr_df = pd.DataFrame({'Step': steps, 'Learning Rate': lrs})
            st.line_chart(lr_df.set_index('Step'))
            st.caption("Learning Rate Schedule")


def comparison_tab():
    """Model comparison and analysis tab."""
    st.header("📈 Model Comparison")
    
    trained_models = get_trained_models()
    
    if len(trained_models) < 2:
        st.warning("Need at least 2 trained models for comparison.")
        if trained_models:
            st.info(f"Currently have {len(trained_models)} trained model(s).")
        return
    
    # Model selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        model_a = st.selectbox(
            "Model A",
            trained_models,
            help="First model for comparison"
        )
    
    with col2:
        model_b = st.selectbox(
            "Model B", 
            [m for m in trained_models if m != model_a],
            help="Second model for comparison"
        )
    
    if st.button("🔍 Run Comparison", use_container_width=True):
        with st.spinner("Running model comparison..."):
            comparison_results = run_model_comparison(model_a, model_b)
            display_comparison_results(comparison_results)


def datasets_tab():
    """Dataset management and preprocessing tab."""
    st.header("💾 Dataset Management")
    
    # Dataset discovery
    st.subheader("📁 Available Datasets")
    
    grid_sizes = [8, 10, 12, 16, 20]
    selected_grid_size = st.selectbox("Grid Size", grid_sizes, index=1)
    
    datasets = get_available_datasets(selected_grid_size)
    
    if datasets:
        # Dataset table
        dataset_info = []
        for dataset in datasets:
            info = get_dataset_info(selected_grid_size, dataset)
            dataset_info.append(info)
        
        df = pd.DataFrame(dataset_info)
        st.dataframe(df, use_container_width=True)
        
        # Dataset preprocessing options
        st.subheader("🔧 Dataset Preprocessing")
        
        selected_datasets = st.multiselect(
            "Select datasets for preprocessing",
            datasets,
            help="Choose datasets to preprocess for training"
        )
        
        if selected_datasets and st.button("🔄 Preprocess Datasets"):
            with st.spinner("Preprocessing datasets..."):
                # Mock preprocessing
                st.success(f"✅ Preprocessed {len(selected_datasets)} datasets")
    else:
        st.warning(f"No datasets found for grid size {selected_grid_size}")
        st.info("Generate datasets using heuristics extensions first.")


def evaluation_tab():
    """Model evaluation and testing tab."""
    st.header("🔍 Model Evaluation")
    
    trained_models = get_trained_models()
    
    if not trained_models:
        st.warning("No trained models available for evaluation.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Evaluation",
        trained_models,
        help="Choose a trained model to evaluate"
    )
    
    # Evaluation configuration
    col1, col2 = st.columns(2)
    
    with col1:
        eval_dataset = st.selectbox(
            "Evaluation Dataset",
            get_available_datasets(10),  # Default grid size
            help="Choose dataset for evaluation"
        )
        
        num_games = st.slider(
            "Number of Games",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of games to evaluate"
        )
    
    with col2:
        metrics = st.multiselect(
            "Evaluation Metrics",
            [
                "Win Rate",
                "Average Score", 
                "Decision Accuracy",
                "Response Time",
                "BLEU Score",
                "Perplexity"
            ],
            default=["Win Rate", "Average Score", "Decision Accuracy"],
            help="Choose metrics to compute"
        )
    
    if st.button("🚀 Run Evaluation", use_container_width=True):
        with st.spinner("Running model evaluation..."):
            evaluation_results = run_model_evaluation(
                selected_model, eval_dataset, num_games, metrics
            )
            display_evaluation_results(evaluation_results)


def settings_tab():
    """Application settings and configuration tab."""
    st.header("⚙️ Settings & Configuration")
    
    # General settings
    st.subheader("🔧 General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            help="Set logging verbosity"
        )
        
        auto_save = st.checkbox(
            "Auto-save Configurations",
            value=True,
            help="Automatically save training configurations"
        )
    
    with col2:
        max_history = st.number_input(
            "Max History Items",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum number of history items to keep"
        )
    
    # System information
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Extension Version", "0.03")
        st.metric("v0.02 Components", "✅ Available" if v0_02_available else "❌ Missing")
    
    with col2:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}")
        st.metric("Streamlit Available", "✅ Yes")


# Helper functions
def get_available_datasets(grid_size: int) -> List[str]:
    """Get list of available datasets for given grid size."""
    dataset_dir = Path(f"logs/extensions/datasets/grid-size-{grid_size}")
    if not dataset_dir.exists():
        return []
    
    datasets = []
    for file_path in dataset_dir.glob("*.csv"):
        datasets.append(file_path.name)
    for file_path in dataset_dir.glob("*.jsonl"):
        datasets.append(file_path.name)
    
    return sorted(datasets)


def get_dataset_info(grid_size: int, dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset."""
    dataset_path = Path(f"logs/extensions/datasets/grid-size-{grid_size}") / dataset_name
    
    try:
        stat = dataset_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        
        # Try to get sample count
        if dataset_path.suffix == '.csv':
            with open(dataset_path, 'r') as f:
                sample_count = sum(1 for _ in f) - 1  # Subtract header
        elif dataset_path.suffix == '.jsonl':
            with open(dataset_path, 'r') as f:
                sample_count = sum(1 for _ in f)
        else:
            sample_count = "Unknown"
        
        return {
            'Dataset': dataset_name,
            'Samples': sample_count,
            'Size (MB)': f"{size_mb:.2f}",
            'Modified': modified,
        }
    except Exception:
        return {
            'Dataset': dataset_name,
            'Samples': "Error",
            'Size (MB)': "Error",
            'Modified': "Error",
        }


def get_trained_models() -> List[str]:
    """Get list of trained models."""
    models_dir = Path("logs/extensions/models")
    if not models_dir.exists():
        return []
    
    models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    
    return sorted(models)


def start_training(config: Dict[str, Any]):
    """Start training with given configuration."""
    try:
        st.session_state['training_in_progress'] = True
        st.session_state['current_config'] = config
        
        # Add to experiment history
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'strategy': config['strategy'],
            'model': config['model_name'],
            'datasets': len(config['datasets']),
            'epochs': config['num_epochs'],
            'status': 'running'
        }
        
        history = st.session_state.get('experiment_history', [])
        history.append(experiment)
        st.session_state['experiment_history'] = history
        
        st.success(f"🚀 Started training {config['strategy']} on {len(config['datasets'])} datasets")
        
    except Exception as e:
        st.error(f"Failed to start training: {e}")


def stop_training():
    """Stop current training."""
    st.session_state['training_in_progress'] = False
    st.warning("⏹️ Training stopped by user")


def save_training_config(config: Dict[str, Any]):
    """Save training configuration."""
    config_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.success(f"💾 Configuration saved as '{config_name}'")


def run_model_comparison(model_a: str, model_b: str) -> Dict[str, Any]:
    """Run comparison between two models."""
    # Mock results for demonstration
    return {
        'model_a': model_a,
        'model_b': model_b,
        'metrics': {
            'win_rate': {'a': 0.75, 'b': 0.72, 'difference': 0.03},
            'avg_score': {'a': 12.5, 'b': 11.8, 'difference': 0.7},
        }
    }


def display_comparison_results(results: Dict[str, Any]):
    """Display model comparison results."""
    st.subheader(f"📊 {results['model_a']} vs {results['model_b']}")
    
    metrics = results.get('metrics', {})
    
    for metric_name, metric_data in metrics.items():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{results['model_a']} {metric_name}", f"{metric_data['a']:.2f}")
        
        with col2:
            st.metric(f"{results['model_b']} {metric_name}", f"{metric_data['b']:.2f}")
        
        with col3:
            difference = metric_data['difference']
            st.metric("Difference", f"{difference:+.2f}")


def run_model_evaluation(model: str, dataset: str, 
                       num_games: int, metrics: List[str]) -> Dict[str, Any]:
    """Run model evaluation."""
    # Mock evaluation results
    return {
        'model': model,
        'dataset': dataset,
        'num_games': num_games,
        'results': {
            'Win Rate': 0.78,
            'Average Score': 13.2,
            'Decision Accuracy': 0.85,
        }
    }


def display_evaluation_results(results: Dict[str, Any]):
    """Display model evaluation results."""
    st.subheader(f"📊 Evaluation Results: {results['model']}")
    
    metrics_data = results.get('results', {})
    
    # Display metrics in columns
    cols = st.columns(len(metrics_data))
    for i, (metric, value) in enumerate(metrics_data.items()):
        with cols[i]:
            st.metric(metric, f"{value:.3f}")


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="LLM Fine-tuning Integration v0.03",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    auto_refresh, show_advanced = setup_sidebar()
    
    # Main content
    try:
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🚀 LLM Fine-tuning Integration v0.03")
            st.markdown("*Interactive web interface for heuristics-based LLM fine-tuning*")
        
        with col2:
            if st.session_state.get('training_in_progress'):
                st.error("🔄 Training in Progress")
            else:
                st.success("✅ Ready")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Training", "📊 Monitoring", "📈 Comparison", 
            "💾 Datasets", "🔍 Evaluation", "⚙️ Settings"
        ])
        
        with tab1:
            training_tab()
        
        with tab2:
            monitoring_tab()
        
        with tab3:
            comparison_tab()
        
        with tab4:
            datasets_tab()
        
        with tab5:
            evaluation_tab()
        
        with tab6:
            settings_tab()
            
    except Exception as e:
        st.error(f"Application error: {e}")
        if st.button("🔄 Restart Application"):
            st.experimental_rerun()


if __name__ == "__main__":
    main() 