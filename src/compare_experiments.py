"""
src/compare_experiments.py
--------------------------
Query MLflow tracking to compare all experiment runs and identify
the best model based on the primary R² metric.

Usage:
    python src/compare_experiments.py
    python src/compare_experiments.py --metric rmse --ascending
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compare_runs(
    experiment_name: str,
    tracking_uri: str,
    sort_metric: str = "metrics.r2",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Fetch all runs for the given experiment and return a sorted summary DataFrame.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Path or URI to MLflow tracking store.
        sort_metric: Column name to sort by (e.g. 'metrics.r2').
        ascending: Sort ascending if True (for error metrics like RMSE).

    Returns:
        DataFrame with one row per run, sorted by sort_metric.
    """
    mlflow.set_tracking_uri(tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"{sort_metric} {'ASC' if ascending else 'DESC'}"],
    )

    if runs.empty:
        print("No runs found. Run src/train.py at least once first.")
        return runs

    # Select readable columns
    cols_of_interest = [
        "run_id",
        "params.model_type",
        "params.n_estimators",
        "params.max_depth",
        "metrics.r2",
        "metrics.rmse",
        "metrics.mae",
        "start_time",
        "status",
    ]
    available = [c for c in cols_of_interest if c in runs.columns]
    return runs[available].reset_index(drop=True)


def print_summary(df: pd.DataFrame, sort_metric: str) -> None:
    """Print a formatted comparison table and highlight the best run."""
    print("\n" + "=" * 80)
    print("  MLFLOW EXPERIMENT COMPARISON — Wind Turbine Power Prediction")
    print("=" * 80)
    print(df.to_string(index=True))
    print("\n" + "-" * 80)

    best_idx = 0  # Already sorted
    best = df.iloc[best_idx]
    print(f"\n🏆  BEST RUN (by {sort_metric}):")
    print(f"    Run ID      : {best.get('run_id', 'N/A')}")
    print(f"    Model       : {best.get('params.model_type', 'N/A')}")
    print(f"    n_estimators: {best.get('params.n_estimators', 'N/A')}")
    print(f"    max_depth   : {best.get('params.max_depth', 'N/A')}")
    print(f"    R²          : {best.get('metrics.r2', 'N/A'):.4f}" if isinstance(best.get("metrics.r2"), float) else f"    R²          : {best.get('metrics.r2', 'N/A')}")
    print(f"    RMSE (kW)   : {best.get('metrics.rmse', 'N/A'):.2f}" if isinstance(best.get("metrics.rmse"), float) else f"    RMSE (kW)   : {best.get('metrics.rmse', 'N/A')}")
    print(f"    MAE  (kW)   : {best.get('metrics.mae', 'N/A'):.2f}" if isinstance(best.get("metrics.mae"), float) else f"    MAE  (kW)   : {best.get('metrics.mae', 'N/A')}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MLflow experiment runs")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--metric", default="r2",
                        choices=["r2", "rmse", "mae"],
                        help="Primary metric to rank runs by")
    parser.add_argument("--ascending", action="store_true",
                        help="Sort ascending (use for RMSE/MAE)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    sort_metric = f"metrics.{args.metric}"
    ascending = args.ascending or args.metric in ("rmse", "mae")

    df = compare_runs(
        experiment_name=config["project"]["name"],
        tracking_uri=config["paths"]["mlflow_tracking_uri"],
        sort_metric=sort_metric,
        ascending=ascending,
    )

    if not df.empty:
        print_summary(df, args.metric)
