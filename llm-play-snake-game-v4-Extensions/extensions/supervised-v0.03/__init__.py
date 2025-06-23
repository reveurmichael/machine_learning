"""
Supervised Learning v0.03 - Web Interface & Multi-Model Framework
================================================================

This extension implements a comprehensive supervised learning framework with web interface,
supporting multiple model types and interactive training/evaluation.

Design Philosophy:
- Web interface with Streamlit for interactive training and evaluation
- Multi-model support (Neural Networks, Tree Models, Graph Models)
- Standardized model saving/loading with ONNX export
- Grid size flexibility and proper directory structure
- No backward compatibility - only modern, future-proof code

Key Components:
- Streamlit web interface (app.py)
- Multi-model agents (MLP, CNN, LSTM, XGBoost, LightGBM, GCN)
- Interactive training and evaluation scripts
- Standardized model utilities with metadata
- Replay and visualization capabilities

Usage:
    streamlit run app.py
    python scripts/train.py --model MLP --grid-size 15
    python scripts/evaluate.py --model-path logs/extensions/models/grid-size-15/pytorch/
"""

__version__ = "0.03"
__author__ = "Snake Game Extensions"
__description__ = "Supervised Learning with Web Interface for Snake Game" 