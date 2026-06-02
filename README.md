# Wind Turbine Power Output Prediction (MLOps Pipeline)

A production-grade MLOps pipeline that predicts **wind turbine power output (kW)**
from meteorological and operational sensor data. The project demonstrates end-to-end
ML engineering practices: version control with DVC, experiment tracking with MLflow,
automated testing with pytest, CI/CD with GitHub Actions, and drift monitoring with Evidently.

---

## Project Structure

```
wind_turbine_mlops_project/
├── src/
│   ├── preprocess.py          # Data loading, cleaning, feature engineering
│   ├── train.py               # Training script (reads from config, logs to MLflow)
│   ├── evaluate.py            # Metric computation and threshold checks
│   ├── compare_experiments.py # Query and rank MLflow runs
│   └── monitor_drift.py       # Evidently feature drift detection
├── configs/
│   └── train_config.yaml      # All hyperparameters, paths, and settings
├── tests/
│   └── test_pipeline.py       # 11 pytest tests (unit test, data and model validation)
├── data/
│   └── wind_turbine_production.csv.dvc  # DVC pointer which is the actual CSV tracked by DVC
├── reports/                   # Generated drift HTML reports (gitignored)
├── mlruns/                    # MLflow tracking store (gitignored)
├── .github/
│   └── workflows/
│       └── pipeline.yml       # GitHub Actions: test → train CI/CD pipeline
├── requirements.txt
├── .gitignore
├── MONITORING.md
└── README.md
```

---

## Dataset & Prediction Task

**Dataset:** Synthetic wind turbine SCADA data (3,000 rows × 14 columns)  
**Task:** Using regression to predict `power_output_kw` (0–2,050 kW per turbine)  
**Features:** 9 numeric + 4 categorical (turbine ID, weather condition, season, availability status)

The dataset simulates a 12-turbine wind farm with:
- Physics-inspired power generation (Betz limit, swept area, Cp efficiency)
- Realistic Weibull-distributed wind speeds
- Seasonal temperature variation
- Cut-in (3 m/s) and cut-out (24 m/s) wind speed limits
- ~4% missing values in 4 sensor columns

Key engineered features: `wind_power_density` (0.5 × ρ × v³), `is_high_wind` (above-rated flag).

---

## Setup

### Prerequisites
- Python 3.11+
- Git
- pip

### Installation

```bash
git clone https://github.com/<ieyitayo>/wind-turbine-mlops.git
cd wind-turbine-mlops
pip install -r requirements.txt
```

### Pull Data with DVC

```bash
pip install dvc
dvc pull
```

> **Note:** If you don't have access to the configured DVC remote, you can generate
> the dataset locally:
> ```bash
> python3 -c "exec(open('generate_data.py').read())"
> ```

---

## Running Training

### Basic run (uses defaults from config):
```bash
python src/train.py
```

### Override hyperparameters via CLI:
```bash
python src/train.py --model-type GradientBoostingRegressor --n-estimators 150 --max-depth 5
python src/train.py --model-type Ridge
python src/train.py --n-estimators 300 --max-depth 20
```

### All parameters are defined in `configs/train_config.yaml` — no hardcoded values.

---

##  Running Tests

```bash
pytest tests/ -v
```

Expected output: **11 passed** covering:
- Unit tests: missing value imputation, categorical encoding, input validation (6 tests)
- Data validation: schema checks, target bounds, feature ranges (3 tests)
- Model validation: prediction shape/type, minimum R² threshold (2 tests)

---

## MLflow Experiment Comparison

After running multiple training experiments:

```bash
python src/compare_experiments.py             # rank by R² (default)
python src/compare_experiments.py --metric rmse --ascending  # rank by RMSE
```

**Five experiments were run:**

| Model | n_estimators | max_depth | R² | RMSE (kW) |
|---|---|---|---|---|
| RandomForestRegressor | 300 | 20 | 0.9928 | 63.4 |
| RandomForestRegressor | 200 | 15 | 0.9928 | 63.5 |
| RandomForestRegressor | 50 | 8 | 0.9922 | 65.9 |
| GradientBoostingRegressor | 150 | 5 | 0.9916 | 68.3 |
| Ridge | — | — | 0.8790 | 259.6 |

**Best model:** RandomForestRegressor with 300 estimators and max_depth=20 (R²=0.9928).  
The tree-based models dramatically outperform Ridge regression because power output
is a non-linear function of wind speed (P ∝ v³ in the below-rated region).

---

## Drift Monitoring

```bash
python src/monitor_drift.py
python src/monitor_drift.py --threshold 0.25 --report reports/custom_report.html
```

The script compares Summer/Autumn training data against simulated Winter production data.
An HTML report is saved to `reports/drift_report.html`.

See [MONITORING.md] for the full drift analysis.

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main` and every pull request:

1. **Test job**: installs dependencies, runs `pytest tests/ -v`
2. **Train job** *(depends on test passing)*: runs training script, verifies R² ≥ 0.85

Check the **Actions** tab in your GitHub repository for pipeline run history.

---

## Configuration

All settings live in `configs/train_config.yaml`. The training script reads this file
at runtime- nothing is hardcoded in source code.

Key configurable sections:
- `model.type`: switch between RandomForestRegressor, GradientBoostingRegressor, Ridge
- `model.hyperparameters`: all sklearn estimator parameters
- `evaluation.minimum_r2_threshold`: training will exit with code 1 if not met
- `drift.threshold`: fraction of drifted features triggering exit code 1
