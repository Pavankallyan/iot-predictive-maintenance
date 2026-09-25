"""Windowing + feature engineering for sensor anomaly detection.

For each device, slide a 24h trailing window (step 6h) over the readings and
compute per-sensor statistics plus short-term trend slopes. Windows falling
within 72h before a known failure are labeled 1 (prefailure), everything else 0.

Features per window (3 sensors x [mean, std, min, max, slope] = 15):
    temp_mean/std/min/max/slope, vib_mean/std/min/max/slope, power_mean/std/...
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

SENSORS = ["temperature_c", "vibration_mm_s", "power_w"]
WINDOW_H = 24
STEP_H = 6
PREFAIL_H = 72

FEATURE_COLS = [f"{s}_{stat}" for s in ["temp", "vib", "power"]
                for stat in ["mean", "std", "min", "max", "slope"]]


def _slope(y):
    x = np.arange(len(y), dtype=float)
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_windows(readings: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    readings = readings.copy()
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    faults = faults.copy()
    faults["failure_time"] = pd.to_datetime(faults["failure_time"])
    fail_by_device = dict(zip(faults["device_id"], faults["failure_time"]))

    rows = []
    for device_id, grp in readings.groupby("device_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        failure_time = fail_by_device.get(device_id)
        for start in range(0, len(grp) - WINDOW_H + 1, STEP_H):
            win = grp.iloc[start:start + WINDOW_H]
            end_time = win["timestamp"].iloc[-1]
            feat = {"device_id": device_id, "window_end": end_time}
            for sensor, prefix in zip(SENSORS, ["temp", "vib", "power"]):
                vals = win[sensor].to_numpy()
                feat[f"{prefix}_mean"] = float(vals.mean())
                feat[f"{prefix}_std"] = float(vals.std())
                feat[f"{prefix}_min"] = float(vals.min())
                feat[f"{prefix}_max"] = float(vals.max())
                feat[f"{prefix}_slope"] = _slope(vals)
            if failure_time is not None:
                hours_to_fail = (failure_time - end_time).total_seconds() / 3600.0
                feat["label"] = int(0 < hours_to_fail <= PREFAIL_H)
            else:
                feat["label"] = 0
            rows.append(feat)
    return pd.DataFrame(rows)


def load_windows() -> pd.DataFrame:
    readings = pd.read_csv(os.path.join(DATA_DIR, "sensor_readings.csv"))
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv"))
    return build_windows(readings, faults)


if __name__ == "__main__":
    df = load_windows()
    print(df.head().to_string())
    print(f"\n{len(df)} windows, {int(df['label'].sum())} prefailure "
          f"({df['label'].mean():.2%})")
