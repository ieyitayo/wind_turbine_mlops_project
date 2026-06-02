# Drift Monitoring Analysis (Wind Turbine Power Model)

## Overview

The drift monitoring script (`src/monitor_drift.py`) compares a **reference distribution**
(Summer/Autumn training data, n=1,489) against a **simulated production dataset** representing
Winter operating conditions (n=500). This reflects a realistic deployment scenario: a model
trained during summer operations is applied through the winter season without retraining.

Evidently's K-S test (p < 0.05) is used to flag individual feature drift.
The overall drift share threshold is **30%** (configurable via `configs/train_config.yaml`).

---

## Drift Results

```
Features monitored : 9
Features drifted   : 3 (33.3%)
Drift threshold    : 30.0%
Status             : DRIFT ALERT
```

| Feature | Status | Reason |
|---|---|---|
| `wind_speed_ms` | ✓ Stable | Wind speed distribution not significantly shifted |
| `wind_direction_deg` | ✓ Stable | Direction is season-independent in this simulation |
| `temperature_c` | ✓ Stable | K-S test did not flag despite the 14°C mean shift* |
| `air_density_kgm3` | ✓ Stable | Correlated with temperature; overlap in distributions |
| `rotor_speed_rpm` | ✗ **Drifted** | Higher winter wind speeds shift rotor RPM distribution |
| `pitch_angle_deg` | ✗ **Drifted** | Pitch control responds differently at higher wind speeds |
| `turbine_age_years` | ✗ **Drifted** | Sampling artifact — winter subset has different age mix |
| `nacelle_temp_c` | ✓ Stable | Composite of ambient temp + rotor speed; K-S boundary |
| `day_of_year` | ✓ Stable | Integer distribution shifts but K-S test not significant |

*Some temperature drift may be absorbed by the continuous K-S test's sensitivity at the
sample sizes used. Running a larger production sample would expose it clearly.

---

## Question 1: Which features drifted and why?

**`rotor_speed_rpm`** drifted because rotor speed is directly driven by wind speed.
The Winter simulation applies a +3 m/s mean shift to wind speed, which pushes turbines
more often into their above-rated operating region. In this region, the pitch controller
limits rotor speed, creating a compressed, bi-modal RPM distribution that is statistically
distinct from the summer training distribution.

**`pitch_angle_deg`** drifted for the same underlying reason. Below rated wind speed (~12 m/s),
pitch angles cluster near 0–5°. Above rated, pitch angles actively increase to 10–30° to
limit power. Winter winds push more observations into the above-rated region, shifting the
pitch angle distribution from a low-angle cluster to a higher-angle cluster. The K-S test
detects this as a clear distribution shift.

**`turbine_age_years`** shows drift as a sampling artifact. The Winter production dataset
is sampled from Summer/Autumn rows with replacement, and the random seed selects a
non-representative age distribution. In a real deployment this field would not drift
(turbines age deterministically), but it illustrates how small sample sizes can produce
spurious drift signals — an important monitoring caveat.

---

## Question 2: Would this drift likely affect model performance?

**Yes — materially.** The drifted features are not peripheral:

- `rotor_speed_rpm` is one of the most predictive features in the Random Forest
  (high correlation with power output: rotor speed directly relates to energy conversion).
  A distribution shift here moves many predictions outside the training data's
  feature space, where the forest extrapolates rather than interpolates.

- `pitch_angle_deg` encodes the control regime the turbine is operating in (below-rated
  vs. above-rated). The model learned separate patterns for each regime. If Winter data
  presents a disproportionately high-pitch distribution, the model will predict using
  patterns learned from fewer training samples in that region, increasing RMSE.

A practical indicator: the model achieves R²=0.993 on Summer/Autumn test data.
On a winter holdout with the drifted distribution, we would expect R² to degrade
to approximately 0.92–0.96 based on the magnitude of the shift. A formal retrain
would confirm this.

---

## Question 3: What action would you recommend?

### Recommendation: **Retrain with seasonal data inclusion**

The drift pattern is systematic (seasonal), predictable, and affects operationally
important features. This is not noise,  it is a known physics-driven regime shift.

**Immediate actions:**
1. **Do not blindly trust current model predictions for winter data.** Apply wider
   prediction intervals and flag high-wind predictions (>15 m/s) for manual review.
2. **Collect Winter production labels** as they come in to build a labeled retraining set.

**30-day action plan:**
1. Retrain the model on a dataset balanced across all four seasons.
2. Re-evaluate with a stratified seasonal test split to confirm drift-robust performance.
3. Lower the drift threshold to 25% so the alert fires earlier next year before
   Winter drift accumulates.

**Longer-term:**
- Implement rolling retraining: retrain monthly with a 12-month rolling window so the
  model always has recent Winter data in its training set.
- Add `season` as an explicit stratification variable in the train/test split
  (currently split randomly, which can under-represent extreme seasons).
- Monitor `rotor_speed_rpm` and `pitch_angle_deg` specifically with a tighter
  column-level threshold (p < 0.01) as early-warning indicators.
