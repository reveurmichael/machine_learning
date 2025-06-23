"""
Training dashboard component for supervised learning v0.03.

Design Pattern: Component Pattern
- Focused, single-responsibility component
- Clean interface for Streamlit integration
- Modular dashboard architecture
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from pathlib import Path
import json


class TrainingDashboard:
    """Dashboard component for training visualization."""
    
    def __init__(self):
        """Initialize the training dashboard."""
        self.metrics_data = []
        self.config_data = {}
    
    def render_header(self):
        """Render dashboard header."""
        st.title("🎯 Supervised Learning v0.03 - Training Dashboard")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Models", len(self.metrics_data))
        with col2:
            st.metric("Total Experiments", len(self.config_data))
        with col3:
            st.metric("Best Accuracy", self._get_best_accuracy())
    
    def render_configuration(self, config: Dict[str, Any]):
        """Render configuration section."""
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(config.get("model", {}))
        
        with col2:
            st.json(config.get("training", {}))
    
    def render_training_metrics(self, metrics: List[Dict[str, Any]]):
        """Render training metrics visualization."""
        if not metrics:
            st.warning("No training metrics available")
            return
        
        st.subheader("📊 Training Metrics")
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        
        # Training curves
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = px.line(df, x="epoch", y=["train_loss", "val_loss"],
                              title="Training & Validation Loss")
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            fig_acc = px.line(df, x="epoch", y=["train_acc", "val_acc"],
                             title="Training & Validation Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Metrics summary
        st.subheader("📈 Metrics Summary")
        
        if len(df) > 0:
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Train Loss", f"{latest['train_loss']:.4f}")
            with col2:
                st.metric("Final Val Loss", f"{latest['val_loss']:.4f}")
            with col3:
                st.metric("Final Train Acc", f"{latest['train_acc']:.4f}")
            with col4:
                st.metric("Final Val Acc", f"{latest['val_acc']:.4f}")
    
    def render_model_comparison(self, models_data: List[Dict[str, Any]]):
        """Render model comparison section."""
        if not models_data:
            return
        
        st.subheader("🔍 Model Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_data in models_data:
            comparison_data.append({
                "Model": model_data.get("model_type", "Unknown"),
                "Accuracy": model_data.get("final_accuracy", 0),
                "Training Time": model_data.get("training_time", 0),
                "Parameters": model_data.get("num_parameters", 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Comparison chart
        fig = px.bar(df, x="Model", y="Accuracy", 
                    title="Model Accuracy Comparison",
                    color="Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.dataframe(df, use_container_width=True)
    
    def render_experiment_selector(self, experiments: List[str]) -> Optional[str]:
        """Render experiment selector."""
        st.subheader("🧪 Experiment Selection")
        
        if not experiments:
            st.warning("No experiments found")
            return None
        
        selected = st.selectbox("Choose Experiment", experiments)
        return selected
    
    def render_model_selector(self, models: List[str]) -> Optional[str]:
        """Render model type selector."""
        st.subheader("🤖 Model Selection")
        
        model_options = ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"]
        selected = st.selectbox("Choose Model Type", model_options)
        return selected
    
    def render_training_controls(self) -> Dict[str, Any]:
        """Render training control parameters."""
        st.subheader("🎮 Training Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        
        with col2:
            grid_size = st.selectbox("Grid Size", [5, 10, 15, 20], index=1)
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, format="%.1f")
            max_games = st.number_input("Max Games", 100, 10000, 1000, step=100)
        
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "grid_size": grid_size,
            "validation_split": validation_split,
            "max_games": max_games
        }
    
    def render_status(self, status: str, message: str = ""):
        """Render status information."""
        if status == "success":
            st.success(f"✅ {message}")
        elif status == "error":
            st.error(f"❌ {message}")
        elif status == "warning":
            st.warning(f"⚠️ {message}")
        elif status == "info":
            st.info(f"ℹ️ {message}")
    
    def _get_best_accuracy(self) -> str:
        """Get the best accuracy from metrics data."""
        if not self.metrics_data:
            return "N/A"
        
        best_acc = max(self.metrics_data, key=lambda x: x.get("val_acc", 0))
        return f"{best_acc.get('val_acc', 0):.4f}"
    
    def load_metrics_from_file(self, filepath: Path):
        """Load metrics from JSON file."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.metrics_data = json.load(f)
    
    def load_config_from_file(self, filepath: Path):
        """Load configuration from JSON file."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.config_data = json.load(f)


def create_training_dashboard() -> TrainingDashboard:
    """Factory function to create training dashboard."""
    return TrainingDashboard() 