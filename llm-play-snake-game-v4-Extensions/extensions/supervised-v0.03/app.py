#!/usr/bin/env python3
"""
Supervised Learning v0.03 - Streamlit Web Interface
==================================================

Modern Streamlit web interface for interactive supervised learning training and evaluation.
Provides tabs for different model types and comprehensive parameter control.

Design Pattern: Template Method
- Modular tab-based interface
- Interactive parameter adjustment
- Real-time training progress visualization
- Model comparison and evaluation
"""

import sys
import os
from pathlib import Path

# Fix Python path for Streamlit
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

import streamlit as st
import subprocess
import json
from pathlib import Path

# Import extension components
from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


def main():
    """Main Streamlit application for supervised learning v0.03."""
    
    st.set_page_config(
        page_title="Supervised Learning v0.03",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Supervised Learning v0.03 - Snake Game AI")
    st.markdown("Interactive training and evaluation for multiple model types")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("Global Settings")
        grid_size = st.selectbox("Grid Size", [8, 10, 12, 15, 20], index=1)
        max_games = st.slider("Max Games", 1, 1000, 100)
        verbose = st.checkbox("Verbose Output", value=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Neural Networks", "🌳 Tree Models", "🕸️ Graph Models", "📊 Evaluation"
    ])
    
    with tab1:
        neural_networks_interface(grid_size, max_games, verbose)
    
    with tab2:
        tree_models_interface(grid_size, max_games, verbose)
    
    with tab3:
        graph_models_interface(grid_size, max_games, verbose)
    
    with tab4:
        evaluation_interface(grid_size)


def neural_networks_interface(grid_size: int, max_games: int, verbose: bool):
    """Interface for neural network models."""
    st.header("Neural Networks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Model Type",
            ["MLP", "CNN", "LSTM", "GRU"],
            key="neural_model"
        )
        
        # Model-specific parameters
        if model_type in ["MLP", "LSTM", "GRU"]:
            hidden_size = st.slider("Hidden Size", 64, 512, 256, step=64)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.001, 0.01, 0.1],
                value=0.001
            )
        
        epochs = st.slider("Epochs", 10, 1000, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col2:
        st.subheader("Training Control")
        
        if st.button(f"🚀 Train {model_type}", key="train_neural"):
            with st.spinner(f"Training {model_type}..."):
                result = train_neural_model(
                    model_type, grid_size, max_games, epochs, 
                    batch_size, verbose, hidden_size, learning_rate
                )
                st.success(f"{model_type} training completed!")
                st.json(result)
        
        if st.button(f"🎯 Evaluate {model_type}", key="eval_neural"):
            with st.spinner(f"Evaluating {model_type}..."):
                result = evaluate_model(model_type, grid_size)
                st.success(f"{model_type} evaluation completed!")
                st.json(result)


def tree_models_interface(grid_size: int, max_games: int, verbose: bool):
    """Interface for tree-based models."""
    st.header("Tree Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Model Type",
            ["XGBoost", "LightGBM", "RandomForest"],
            key="tree_model"
        )
        
        # Tree-specific parameters
        if model_type == "XGBoost":
            max_depth = st.slider("Max Depth", 3, 10, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        elif model_type == "LightGBM":
            num_leaves = st.slider("Num Leaves", 10, 100, 31)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    
    with col2:
        st.subheader("Training Control")
        
        if st.button(f"🚀 Train {model_type}", key="train_tree"):
            with st.spinner(f"Training {model_type}..."):
                result = train_tree_model(
                    model_type, grid_size, max_games, verbose,
                    max_depth, learning_rate, num_leaves
                )
                st.success(f"{model_type} training completed!")
                st.json(result)
        
        if st.button(f"🎯 Evaluate {model_type}", key="eval_tree"):
            with st.spinner(f"Evaluating {model_type}..."):
                result = evaluate_model(model_type, grid_size)
                st.success(f"{model_type} evaluation completed!")
                st.json(result)


def graph_models_interface(grid_size: int, max_games: int, verbose: bool):
    """Interface for graph neural network models."""
    st.header("Graph Neural Networks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Model Type",
            ["GCN", "GraphSAGE", "GAT"],
            key="graph_model"
        )
        
        # GNN-specific parameters
        hidden_channels = st.slider("Hidden Channels", 16, 128, 64, step=16)
        num_layers = st.slider("Num Layers", 2, 5, 3)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01],
            value=0.001
        )
    
    with col2:
        st.subheader("Training Control")
        
        if st.button(f"🚀 Train {model_type}", key="train_graph"):
            with st.spinner(f"Training {model_type}..."):
                result = train_graph_model(
                    model_type, grid_size, max_games, verbose,
                    hidden_channels, num_layers, learning_rate
                )
                st.success(f"{model_type} training completed!")
                st.json(result)
        
        if st.button(f"🎯 Evaluate {model_type}", key="eval_graph"):
            with st.spinner(f"Evaluating {model_type}..."):
                result = evaluate_model(model_type, grid_size)
                st.success(f"{model_type} evaluation completed!")
                st.json(result)


def evaluation_interface(grid_size: int):
    """Interface for model evaluation and comparison."""
    st.header("Model Evaluation & Comparison")
    
    # Get available models
    model_dir = Path(f"logs/extensions/models/grid-size-{grid_size}")
    available_models = []
    
    if model_dir.exists():
        for framework_dir in model_dir.iterdir():
            if framework_dir.is_dir():
                for model_file in framework_dir.glob("*.pth"):
                    available_models.append(str(model_file))
                for model_file in framework_dir.glob("*.json"):
                    available_models.append(str(model_file))
                for model_file in framework_dir.glob("*.txt"):
                    available_models.append(str(model_file))
    
    if available_models:
        st.subheader("Available Models")
        selected_models = st.multiselect(
            "Select models to compare",
            available_models,
            key="model_comparison"
        )
        
        if st.button("📊 Compare Models"):
            with st.spinner("Comparing models..."):
                results = compare_models(selected_models, grid_size)
                st.success("Model comparison completed!")
                
                # Display results in a nice format
                for model_path, metrics in results.items():
                    with st.expander(f"📈 {Path(model_path).name}"):
                        st.json(metrics)
    else:
        st.info(f"No trained models found for grid size {grid_size}")


def train_neural_model(model_type: str, grid_size: int, max_games: int, 
                      epochs: int, batch_size: int, verbose: bool,
                      hidden_size: int = 256, learning_rate: float = 0.001) -> dict:
    """Train a neural network model using the training script."""
    cmd = [
        "python", "scripts/train.py",
        "--model", model_type,
        "--grid-size", str(grid_size),
        "--max-games", str(max_games),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--hidden-size", str(hidden_size),
        "--learning-rate", str(learning_rate)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="extensions/supervised-v0.03")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_tree_model(model_type: str, grid_size: int, max_games: int, verbose: bool,
                    max_depth: int = 6, learning_rate: float = 0.1, num_leaves: int = 31) -> dict:
    """Train a tree-based model using the training script."""
    cmd = [
        "python", "scripts/train.py",
        "--model", model_type,
        "--grid-size", str(grid_size),
        "--max-games", str(max_games),
        "--max-depth", str(max_depth),
        "--learning-rate", str(learning_rate),
        "--num-leaves", str(num_leaves)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="extensions/supervised-v0.03")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_graph_model(model_type: str, grid_size: int, max_games: int, verbose: bool,
                     hidden_channels: int = 64, num_layers: int = 3, learning_rate: float = 0.001) -> dict:
    """Train a graph neural network model using the training script."""
    cmd = [
        "python", "scripts/train.py",
        "--model", model_type,
        "--grid-size", str(grid_size),
        "--max-games", str(max_games),
        "--hidden-channels", str(hidden_channels),
        "--num-layers", str(num_layers),
        "--learning-rate", str(learning_rate)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="extensions/supervised-v0.03")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate_model(model_type: str, grid_size: int) -> dict:
    """Evaluate a trained model."""
    cmd = [
        "python", "scripts/evaluate.py",
        "--model", model_type,
        "--grid-size", str(grid_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="extensions/supervised-v0.03")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_models(model_paths: list, grid_size: int) -> dict:
    """Compare multiple trained models."""
    results = {}
    
    for model_path in model_paths:
        try:
            cmd = [
                "python", "scripts/evaluate.py",
                "--model-path", model_path,
                "--grid-size", str(grid_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="extensions/supervised-v0.03")
            
            if result.returncode == 0:
                # Try to parse JSON output
                try:
                    metrics = json.loads(result.stdout)
                    results[model_path] = metrics
                except json.JSONDecodeError:
                    results[model_path] = {"raw_output": result.stdout}
            else:
                results[model_path] = {"error": result.stderr}
                
        except Exception as e:
            results[model_path] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    main() 