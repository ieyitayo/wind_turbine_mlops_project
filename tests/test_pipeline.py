"""
tests/test_pipeline.py
----------------------
pytest test suite for the Wind Turbine Power Prediction MLOps pipeline.

Covers:
  - Unit tests for preprocessing functions   (6 tests)
  - Data validation tests                    (3 tests)
  - Model validation tests                   (2 tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import (
    build_preprocessor,
    encode_categoricals,
    handle_missing_values,
    load_dataset,
    split_features_target,
    validate_dataframe,
)
from src.evaluate import check_minimum_threshold, compute_metrics


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal valid DataFrame that mimics the wind turbine dataset schema."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame({
        "turbine_id": np.random.choice(["WTG-01", "WTG-02"], n),
        "wind_speed_ms": np.random.uniform(3, 15, n),
        "wind_direction_deg": np.random.uniform(0, 360, n),
        "temperature_c": np.random.uniform(-5, 30, n),
        "air_density_kgm3": np.random.uniform(1.15, 1.30, n),
        "rotor_speed_rpm": np.random.uniform(5, 18, n),
        "pitch_angle_deg": np.random.uniform(0, 25, n),
        "nacelle_temp_c": np.random.uniform(10, 50, n),
        "turbine_age_years": np.random.randint(1, 15, n),
        "day_of_year": np.random.randint(1, 366, n),
        "weather_condition": np.random.choice(["Clear", "Cloudy", "Rainy"], n),
        "season": np.random.choice(["Winter", "Spring", "Summer", "Autumn"], n),
        "availability_status": np.random.choice(["Available", "Derated"], n),
        "power_output_kw": np.random.uniform(0, 2000, n),
    })


@pytest.fixture
def df_with_nulls(sample_df) -> pd.DataFrame:
    """Sample DataFrame with deliberate NaN values in numeric columns."""
    df = sample_df.copy()
    df.loc[0, "wind_speed_ms"] = np.nan
    df.loc[5, "temperature_c"] = np.nan
    df.loc[10, "rotor_speed_rpm"] = np.nan
    return df


# ============================================================
# UNIT TESTS — Preprocessing functions (6 tests)
# ============================================================

class TestHandleMissingValues:

    def test_nulls_are_filled(self, df_with_nulls):
        """After imputation, no numeric NaN values should remain."""
        result = handle_missing_values(df_with_nulls)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_fills_with_median(self, df_with_nulls):
        """Imputed value should equal the column median of non-null rows."""
        expected_median = df_with_nulls["wind_speed_ms"].median()
        result = handle_missing_values(df_with_nulls)
        # Row 0 was set to NaN and should now be the median
        assert result.loc[0, "wind_speed_ms"] == pytest.approx(expected_median, rel=1e-6)

    def test_does_not_modify_original(self, df_with_nulls):
        """The original DataFrame must not be mutated (returns a copy)."""
        original_null_count = df_with_nulls["wind_speed_ms"].isnull().sum()
        _ = handle_missing_values(df_with_nulls)
        assert df_with_nulls["wind_speed_ms"].isnull().sum() == original_null_count


class TestEncodeCategoricals:

    def test_categorical_columns_removed(self, sample_df):
        """Original categorical columns should not appear in the encoded output."""
        result = encode_categoricals(sample_df)
        assert "weather_condition" not in result.columns
        assert "season" not in result.columns
        assert "availability_status" not in result.columns

    def test_ohe_columns_created(self, sample_df):
        """OHE should create binary indicator columns for each category level."""
        result = encode_categoricals(sample_df)
        ohe_cols = [c for c in result.columns if c.startswith("weather_condition_")]
        assert len(ohe_cols) > 0, "Expected one-hot encoded weather_condition columns"

    def test_raises_on_unknown_category(self, sample_df):
        """encode_categoricals must raise ValueError for unknown category values."""
        bad_df = sample_df.copy()
        bad_df.loc[0, "weather_condition"] = "Hurricane"
        with pytest.raises(ValueError, match="Unknown weather_condition values"):
            encode_categoricals(bad_df)


# ============================================================
# DATA VALIDATION TESTS (3 tests)
# ============================================================

class TestDataValidation:

    def test_required_columns_present(self):
        """The actual dataset file must contain all required schema columns."""
        df = load_dataset("data/wind_turbine_production.csv")
        from src.preprocess import REQUIRED_COLUMNS
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"

    def test_target_within_physical_bounds(self):
        """Power output must be between 0 and 2200 kW (physical constraint)."""
        df = load_dataset("data/wind_turbine_production.csv")
        assert (df["power_output_kw"] >= 0).all(), "Negative power output detected"
        assert (df["power_output_kw"] <= 2200).all(), "Power output exceeds 2200 kW"

    def test_numeric_features_in_expected_ranges(self):
        """Key numeric features must fall within physically plausible ranges."""
        df = load_dataset("data/wind_turbine_production.csv")
        assert df["wind_speed_ms"].dropna().between(0, 30).all(), \
            "wind_speed_ms out of range [0, 30]"
        assert df["air_density_kgm3"].dropna().between(1.0, 1.5).all(), \
            "air_density_kgm3 out of range [1.0, 1.5]"
        assert df["turbine_age_years"].between(0, 30).all(), \
            "turbine_age_years out of range [0, 30]"


# ============================================================
# MODEL VALIDATION TESTS (2 tests)
# ============================================================

class TestModelValidation:

    @pytest.fixture
    def trained_pipeline(self, sample_df):
        """Train a minimal pipeline on the sample fixture for model tests."""
        numeric_features = [
            "wind_speed_ms", "wind_direction_deg", "temperature_c",
            "air_density_kgm3", "rotor_speed_rpm", "pitch_angle_deg",
            "nacelle_temp_c", "turbine_age_years", "day_of_year",
        ]
        categorical_features = [
            "turbine_id", "weather_condition", "season", "availability_status"
        ]
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        X, y = split_features_target(
            sample_df, "power_output_kw", numeric_features, categorical_features
        )
        pipe.fit(X, y)
        return pipe, X, y

    def test_predictions_correct_shape_and_type(self, trained_pipeline):
        """Predictions must be a 1-D numeric array matching input row count."""
        pipe, X, _ = trained_pipeline
        preds = pipe.predict(X)
        assert preds.ndim == 1, "Predictions should be 1-dimensional"
        assert len(preds) == len(X), "Prediction count must match input row count"
        assert np.issubdtype(preds.dtype, np.floating), "Predictions must be float dtype"

    def test_model_meets_minimum_r2_on_sample(self, trained_pipeline):
        """A model trained and evaluated on the sample fixture must achieve R² >= 0.5."""
        pipe, X, y = trained_pipeline
        preds = pipe.predict(X)
        metrics = compute_metrics(y.values, preds)
        assert check_minimum_threshold(metrics, threshold=0.50), \
            f"Model R² {metrics['r2']:.4f} is below minimum 0.50 on training sample"
