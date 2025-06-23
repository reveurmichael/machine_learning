"""
Tree Model Training Script for Supervised Learning v0.02
--------------------

Trains tree-based models (XGBoost, LightGBM, RandomForest) on tabular datasets.
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train tree-based models (XGBoost, LightGBM, RandomForest)")
    parser.add_argument('--model', type=str, required=True, help='Model type (XGBOOST, LIGHTGBM, RANDOMFOREST)')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset file')
    args = parser.parse_args()
    # TODO: Load dataset, initialize agent, and train
    print(f"Training {args.model} on {args.dataset_path}")

if __name__ == "__main__":
    main() 