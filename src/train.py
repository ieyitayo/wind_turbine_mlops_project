"""
src/train.py
------------
Main training script for the wind turbine power output prediction model.

Usage:
    python src/train.py                          # uses default config
    python src/train.py --config configs/train_config.yaml
    python src/train.py --n-estimators 300 --max-depth 20   # override params

All hyperparameters, metrics, and the trained model are logged to MLflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import (
    build_preprocessor,
    handle_missing_values,
    load_dataset,
    split_features_target,
    validate_dataframe,
)
from src.evaluate import check_minimum_threshold, compute_metrics, residual_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "Ridge": Ridge,
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return config


def build_model(config: dict, overrides: dict | None = None) -> object:
    """
    Instantiate the model from config, applying any CLI overrides.

    Args:
        config: Full config dict.
        overrides: Optional dict of hyperparameter overrides from CLI.

    Returns:
        Unfitted sklearn estimator.
    """
    model_type = config["model"]["type"]
    params = dict(config["model"]["hyperparameters"])
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_type]
    # Ridge does not accept tree-specific params — filter them out
    if model_type == "Ridge":
        ridge_params = {k: v for k, v in params.items()
                        if k in ("alpha", "random_state")}
        return model_cls(**ridge_params)

    return model_cls(**params)


def run_training(config: dict, overrides: dict | None = None) -> dict:
    """
    End-to-end training run: load → preprocess → train → evaluate → log.

    Args:
        config: Configuration dictionary.
        overrides: Optional hyperparameter overrides.

    Returns:
        Metrics dictionary from the training run.
    """
    data_path = config["paths"]["data"]
    target_col = config["data"]["target_column"]
    numeric_features = config["data"]["numeric_features"]
    categorical_features = config["data"]["categorical_features"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    min_r2 = config["evaluation"]["minimum_r2_threshold"]
    tracking_uri = config["paths"]["mlflow_tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["project"]["name"])

    # --- Data loading & validation ---
    df = load_dataset(data_path)
    validate_dataframe(df)
    df = handle_missing_values(df)

    X, y = split_features_target(df, target_col, numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Build sklearn pipeline ---
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(config, overrides)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # --- MLflow run ---
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run started: %s", run_id)

        # Log config hyperparameters
        model_params = dict(config["model"]["hyperparameters"])
        if overrides:
            model_params.update({k: v for k, v in overrides.items() if v is not None})
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("data_version", _get_dvc_hash(data_path))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))

        # Train
        logger.info("Training %s on %d samples...", config["model"]["type"], len(X_train))
        full_pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = full_pipeline.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        residuals = residual_summary(y_test.values, y_pred)

        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(residuals)

        # Log model artifact
        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path="wind_turbine_model",
            registered_model_name="WindTurbinePowerPredictor",
        )

        logger.info("Run %s complete — R²: %.4f | RMSE: %.2f kW | MAE: %.2f kW",
                    run_id, metrics["r2"], metrics["rmse"], metrics["mae"])

    # --- Threshold gate ---
    if not check_minimum_threshold(metrics, min_r2):
        logger.error("Training run did not meet minimum R² threshold of %.2f. Exiting.", min_r2)
        sys.exit(1)

    return metrics


def _get_dvc_hash(data_path: str) -> str:
    """Return DVC MD5 hash if .dvc file exists, else 'untracked'."""
    dvc_file = data_path + ".dvc"
    if os.path.exists(dvc_file):
        with open(dvc_file) as f:
            for line in f:
                if "md5" in line:
                    return line.split(":")[-1].strip()
    return "untracked"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wind turbine power model")
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model-type", default=None,
                        help="Override model type (RandomForestRegressor / GradientBoostingRegressor / Ridge)")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.model_type:
        config["model"]["type"] = args.model_type

    overrides = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "random_state": config["model"]["hyperparameters"].get("random_state", 42),
    }

    metrics = run_training(config, overrides)
    print(f"\n✅ Training complete — R²: {metrics['r2']:.4f} | "
          f"RMSE: {metrics['rmse']:.2f} kW | MAE: {metrics['mae']:.2f} kW")
