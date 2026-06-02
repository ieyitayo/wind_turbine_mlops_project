"""
generate_data.py
----------------
Generates the wind turbine dataset locally.
Run this if DVC pull is unavailable (e.g., CI environments without a remote).

Usage:
    python generate_data.py
"""
import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

np.random.seed(42)
N = 3000

turbine_ids = [f"WTG-{str(i).zfill(2)}" for i in range(1, 13)]
turbine_id = np.random.choice(turbine_ids, N)
wind_speed = np.clip(np.random.weibull(2.2, N) * 9.5, 0, 25)
wind_direction = np.random.uniform(0, 360, N)
day_of_year = np.random.randint(1, 366, N)
temperature = 12 + 10 * \
    np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 3, N)
air_density = 1.293 * (273.15 / (273.15 + temperature)
                       ) + np.random.normal(0, 0.01, N)
rotor_speed = np.clip(wind_speed * 1.8 + np.random.normal(0, 1.5, N), 3, 20)
pitch_angle = np.where(wind_speed < 12, np.random.uniform(
    0, 5, N), np.random.uniform(10, 30, N))
weather_conditions = np.random.choice(
    ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"], N, p=[0.40, 0.30, 0.15, 0.10, 0.05])


def season_map(d): return ("Winter" if d < 80 or d >=
                           355 else "Spring" if d < 172 else "Summer" if d < 264 else "Autumn")


season = np.array([season_map(d) for d in day_of_year])
turbine_age = np.random.choice(range(1, 16), N)
nacelle_temp = temperature + rotor_speed * 0.8 + np.random.normal(0, 2, N)
swept_area = 5027
Cp = np.clip(0.38 - 0.003 * pitch_angle +
             np.random.normal(0, 0.02, N), 0.05, 0.45)
power_raw = 0.5 * air_density * swept_area * (wind_speed ** 3) * Cp / 1000
power_output = np.where(wind_speed < 3, 0, np.where(wind_speed > 24, 0, np.where(
    wind_speed > 12, np.minimum(power_raw, 2000), power_raw)))
power_output = np.clip(power_output + np.random.normal(0, 20, N), 0, 2050)
availability = np.random.choice(
    ["Available", "Derated", "Maintenance"], N, p=[0.88, 0.09, 0.03])
power_output = np.where(availability == "Maintenance", 0, power_output)

df = pd.DataFrame({
    "turbine_id": turbine_id,
    "wind_speed_ms": wind_speed.round(2),
    "wind_direction_deg": wind_direction.round(1),
    "temperature_c": temperature.round(2),
    "air_density_kgm3": air_density.round(4),
    "rotor_speed_rpm": rotor_speed.round(2),
    "pitch_angle_deg": pitch_angle.round(2),
    "nacelle_temp_c": nacelle_temp.round(2),
    "turbine_age_years": turbine_age,
    "day_of_year": day_of_year,
    "weather_condition": weather_conditions,
    "season": season,
    "availability_status": availability,
    "power_output_kw": power_output.round(2),
})

for col in ["wind_speed_ms", "temperature_c", "rotor_speed_rpm", "nacelle_temp_c"]:
    mask = np.random.random(N) < 0.04
    df.loc[mask, col] = np.nan

df.to_csv("data/wind_turbine_production.csv", index=False)
print(
    f"Dataset generated: {df.shape[0]} rows x {df.shape[1]} columns -> data/wind_turbine_production.csv")
