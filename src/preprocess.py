"""
src/preprocess.py
-----------------
Data loading, cleaning, and feature engineering for the wind turbine
power output prediction pipeline.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the wind turbine CSV dataset from disk.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty or cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV at {filepath}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    logger.info("Loaded dataset: %d rows × %d columns from %s", *df.shape, filepath)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "turbine_id", "wind_speed_ms", "wind_direction_deg", "temperature_c",
    "air_density_kgm3", "rotor_speed_rpm", "pitch_angle_deg", "nacelle_temp_c",
    "turbine_age_years", "day_of_year", "weather_condition", "season",
    "availability_status", "power_output_kw",
]


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Assert that the DataFrame contains all expected columns and basic
    sanity constraints on the target variable.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or target values are invalid.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if (df["power_output_kw"] < 0).any():
        raise ValueError("power_output_kw contains negative values — data integrity issue.")

    if (df["power_output_kw"] > 2200).any():
        raise ValueError("power_output_kw exceeds 2200 kW — physically implausible for a 2 MW turbine.")

    logger.info("Dataframe validation passed.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values with the column median.
    Does NOT modify the input DataFrame (returns a copy).

    Args:
        df: Raw DataFrame, potentially containing NaN values.

    Returns:
        DataFrame with numeric NaN values filled with column medians.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed %s with median %.4f", col, median_val)

    logger.info("Missing value imputation complete.")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed derived features.
    Does NOT modify the input DataFrame (returns a copy).

    New features:
        - wind_power_density  : 0.5 * air_density * wind_speed^3  (W/m²)
        - temp_air_density_interaction : temperature × air_density
        - is_high_wind        : 1 if wind_speed > 12 m/s (above-rated region)

    Args:
        df: DataFrame after missing value handling.

    Returns:
        DataFrame with additional feature columns appended.
    """
    df = df.copy()

    df["wind_power_density"] = (
        0.5 * df["air_density_kgm3"] * df["wind_speed_ms"] ** 3
    ).round(4)

    df["temp_air_density_interaction"] = (
        df["temperature_c"] * df["air_density_kgm3"]
    ).round(4)

    df["is_high_wind"] = (df["wind_speed_ms"] > 12).astype(int)

    logger.info("Feature engineering complete. New columns: wind_power_density, "
                "temp_air_density_interaction, is_high_wind")
    return df


# ---------------------------------------------------------------------------
# Encode categoricals
# ---------------------------------------------------------------------------

VALID_WEATHER_CONDITIONS = {"Clear", "Cloudy", "Rainy", "Foggy", "Stormy"}
VALID_SEASONS = {"Winter", "Spring", "Summer", "Autumn"}
VALID_AVAILABILITY = {"Available", "Derated", "Maintenance"}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and one-hot encode categorical columns in-place style (returns copy).
    Raises ValueError if unexpected category values are found.

    Args:
        df: DataFrame with raw categorical columns.

    Returns:
        DataFrame with categorical columns replaced by OHE binary columns.

    Raises:
        ValueError: If unknown category values are detected.
    """
    df = df.copy()

    unknown_weather = set(df["weather_condition"].unique()) - VALID_WEATHER_CONDITIONS
    if unknown_weather:
        raise ValueError(f"Unknown weather_condition values: {unknown_weather}")

    unknown_season = set(df["season"].unique()) - VALID_SEASONS
    if unknown_season:
        raise ValueError(f"Unknown season values: {unknown_season}")

    unknown_avail = set(df["availability_status"].unique()) - VALID_AVAILABILITY
    if unknown_avail:
        raise ValueError(f"Unknown availability_status values: {unknown_avail}")

    df = pd.get_dummies(
        df,
        columns=["weather_condition", "season", "availability_status"],
        drop_first=False,
        dtype=int,
    )
    logger.info("Categorical encoding complete.")
    return df


# ---------------------------------------------------------------------------
# sklearn preprocessor (used inside the training pipeline)
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that imputes and scales numerics,
    and imputes + OHE encodes categoricals.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Configured (unfitted) ColumnTransformer.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target (y).

    Args:
        df: Full DataFrame.
        target_column: Name of the target column.
        numeric_features: Numeric feature names to retain.
        categorical_features: Categorical feature names to retain.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df[target_column].copy()
    return X, y
