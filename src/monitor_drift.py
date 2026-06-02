"""
src/monitor_drift.py
--------------------
Feature drift detection using Evidently (compatible with v0.7+).

Compares a reference distribution (Summer/Autumn training data) against
a simulated winter production dataset, capturing realistic seasonal drift.

Usage:
    python src/monitor_drift.py
    python src/monitor_drift.py --threshold 0.25 --report reports/drift_report.html

Exit codes:
    0 — Drift within acceptable threshold
    1 — Drift exceeds threshold (alert: investigate or retrain)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import handle_missing_values, load_dataset, validate_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_reference_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample reference rows from Summer/Autumn — the model's training distribution."""
    summer_autumn = df[df["season"].isin(["Summer", "Autumn"])].copy()
    n = min(n, len(summer_autumn))
    return summer_autumn.sample(n=n, random_state=42).reset_index(drop=True)


def generate_production_data(reference: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Simulate a winter production dataset with realistic drift:
      - Wind speed shifts higher (+3 m/s, stronger winter winds)
      - Temperature drops ~15 °C (winter cold)
      - Air density increases (denser cold air)
      - Nacelle temperature changes with ambient
      - Weather conditions shift toward Stormy/Foggy
      - Season becomes exclusively Winter
    """
    np.random.seed(99)
    prod = reference.sample(n=n, replace=True, random_state=99).copy().reset_index(drop=True)

    prod["wind_speed_ms"] = np.clip(
        prod["wind_speed_ms"] + np.random.normal(3.0, 1.5, n), 0, 25
    ).round(2)
    prod["temperature_c"] = (
        prod["temperature_c"] - 14 + np.random.normal(0, 2, n)
    ).round(2)
    prod["air_density_kgm3"] = (
        1.293 * (273.15 / (273.15 + prod["temperature_c"])) + np.random.normal(0, 0.01, n)
    ).round(4)
    prod["nacelle_temp_c"] = (
        prod["temperature_c"] + prod["rotor_speed_rpm"] * 0.8 + np.random.normal(0, 2, n)
    ).round(2)
    prod["day_of_year"] = np.random.randint(355, 366, n)
    prod["season"] = "Winter"
    prod["weather_condition"] = np.random.choice(
        ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"],
        n, p=[0.10, 0.30, 0.25, 0.20, 0.15]
    )

    logger.info("Production data generated: %d rows with winter drift applied.", n)
    return prod


def run_drift_detection(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    numeric_features: list[str],
    report_path: str,
    drift_pvalue_threshold: float = 0.05,
) -> tuple[float, list[str], list[str]]:
    """
    Run Evidently drift detection (Evidently v0.7+ API).

    Uses K-S test p-values: a feature is flagged as drifted if p < drift_pvalue_threshold.

    Returns:
        Tuple of (drift_share, drifted_feature_names, stable_feature_names)
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        logger.error("Evidently not installed. Run: pip install evidently")
        sys.exit(1)

    # Filter to only numeric features present in both datasets
    features = [f for f in numeric_features
                if f in reference.columns and f in production.columns]

    ref_data = reference[features].copy()
    prod_data = production[features].copy()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_data, current_data=prod_data)

    # Save HTML report
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    snapshot.save_html(report_path)
    logger.info("Drift report saved to %s", report_path)

    # Parse results: value field = K-S p-value for numeric columns
    result_dict = snapshot.dump_dict()
    metric_results = result_dict.get("metric_results", {})

    drifted_features: list[str] = []
    stable_features: list[str] = []

    for metric_id, metric_data in metric_results.items():
        display_name = metric_data.get("display_name", "")
        if not display_name.startswith("Value drift for "):
            continue

        col_name = display_name.replace("Value drift for ", "")
        if col_name not in features:
            continue

        p_value = metric_data.get("value")
        if p_value is not None and float(p_value) < drift_pvalue_threshold:
            drifted_features.append(col_name)
        else:
            stable_features.append(col_name)

    n_drifted = len(drifted_features)
    n_total = len(features)
    drift_share = n_drifted / n_total if n_total > 0 else 0.0

    return drift_share, drifted_features, stable_features


def print_drift_summary(
    drift_share: float,
    drifted_features: list[str],
    stable_features: list[str],
    threshold: float,
) -> None:
    n_drifted = len(drifted_features)
    n_stable = len(stable_features)
    n_total = n_drifted + n_stable
    status = "🚨 DRIFT ALERT" if drift_share > threshold else "✅ Within threshold"

    print("\n" + "=" * 70)
    print("  WIND TURBINE MODEL — FEATURE DRIFT MONITORING REPORT")
    print("=" * 70)
    print(f"  Reference : Training data (Summer/Autumn distribution)")
    print(f"  Production: Simulated Winter conditions")
    print(f"  Features monitored : {n_total}")
    print(f"  Features drifted   : {n_drifted} ({drift_share:.1%})")
    print(f"  Drift threshold    : {threshold:.1%}")
    print(f"  Status             : {status}")

    if drifted_features:
        print(f"\n  Drifted features ({n_drifted}):")
        for f in drifted_features:
            print(f"    ✗ {f}")

    if stable_features:
        print(f"\n  Stable features ({n_stable}):")
        for f in stable_features:
            print(f"    ✓ {f}")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evidently drift monitoring")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--report", default="reports/drift_report.html")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override drift threshold from config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    numeric_features = config["data"]["numeric_features"]
    ref_size = config["drift"]["reference_size"]
    prod_size = config["drift"]["production_size"]
    threshold = args.threshold or config["drift"]["threshold"]

    df = load_dataset(config["paths"]["data"])
    validate_dataframe(df)
    df = handle_missing_values(df)

    reference = get_reference_data(df, ref_size)
    production = generate_production_data(reference, prod_size)

    logger.info("Reference: %d rows | Production: %d rows", len(reference), len(production))

    drift_share, drifted_features, stable_features = run_drift_detection(
        reference=reference,
        production=production,
        numeric_features=numeric_features,
        report_path=args.report,
    )

    print_drift_summary(drift_share, drifted_features, stable_features, threshold)

    if drift_share > threshold:
        logger.warning(
            "Drift share %.2f exceeds threshold %.2f — recommend investigation or retraining.",
            drift_share, threshold
        )
        sys.exit(1)
    else:
        logger.info("Drift within acceptable limits. Continue monitoring.")
        sys.exit(0)
