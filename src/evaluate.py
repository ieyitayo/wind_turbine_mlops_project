"""
src/evaluate.py
---------------
Model evaluation utilities for the wind turbine power output regression task.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Metrics returned:
        - r2   : Coefficient of determination (primary metric)
        - rmse : Root Mean Squared Error (kW)
        - mae  : Mean Absolute Error (kW)

    Args:
        y_true: Ground truth power output values.
        y_pred: Predicted power output values.

    Returns:
        Dictionary of metric name → float value.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)} elements."
        )

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    metrics = {"r2": round(r2, 6), "rmse": round(rmse, 4), "mae": round(mae, 4)}
    logger.info("Evaluation metrics — R²: %.4f | RMSE: %.2f kW | MAE: %.2f kW",
                r2, rmse, mae)
    return metrics


def check_minimum_threshold(metrics: Dict[str, float], threshold: float) -> bool:
    """
    Verify that the primary R² metric meets the minimum deployment threshold.

    Args:
        metrics: Output of compute_metrics().
        threshold: Minimum acceptable R² value.

    Returns:
        True if R² >= threshold, False otherwise.
    """
    passes = metrics["r2"] >= threshold
    if passes:
        logger.info("Model passed minimum R² threshold: %.4f >= %.4f",
                    metrics["r2"], threshold)
    else:
        logger.warning("Model FAILED minimum R² threshold: %.4f < %.4f",
                       metrics["r2"], threshold)
    return passes


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics of residuals for diagnostic purposes.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with mean, std, p5, p95 of residuals.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_p5": float(np.percentile(residuals, 5)),
        "residual_p95": float(np.percentile(residuals, 95)),
    }
