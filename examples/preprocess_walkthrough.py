"""Walkthrough of the windowing + feature-engineering pipeline.

Builds a tiny synthetic fleet in memory (no data files needed), slides the
24h trailing windows over it, and prints the resulting feature rows and
prefailure labels — a quick look at exactly what the model trains on.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from preprocess import build_windows, PREFAIL_H, WINDOW_H, STEP_H

rng = np.random.default_rng(7)


def make_device(device_id, n_hours, failure_hour=None, drift=False):
    """Hourly sensor readings; optionally ramp vibration + temp before failure."""
    ts = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    temp = 55 + 2 * rng.standard_normal(n_hours)
    vib = 3 + 0.4 * rng.standard_normal(n_hours)
    power = 120 + 5 * rng.standard_normal(n_hours)
    if drift:
        ramp = np.linspace(0, 1, n_hours) ** 2
        vib = vib + 8 * ramp      # bearing-wear: vibration drifts up
        temp = temp + 12 * ramp   # and the device runs hot before dying
    readings = pd.DataFrame({
        "device_id": device_id,
        "timestamp": ts,
        "temperature_c": temp,
        "vibration_mm_s": vib,
        "power_w": power,
    })
    fault = None
    if failure_hour is not None:
        fault = pd.DataFrame({
            "device_id": [device_id],
            "failure_time": [ts[failure_hour]],
        })
    return readings, fault


def main():
    healthy, healthy_fault = make_device("hub-01", n_hours=24 * 10)
    failing, failing_fault = make_device(
        "hub-02", n_hours=24 * 10, failure_hour=24 * 10 - 1, drift=True)

    readings = pd.concat([healthy, failing], ignore_index=True)
    faults = pd.concat(
        [f for f in (healthy_fault, failing_fault) if f is not None],
        ignore_index=True)
    print(f"Synthetic fleet: {readings['device_id'].nunique()} devices, "
          f"{len(readings)} hourly readings")
    print(f"Known faults: {len(faults)} "
          f"(hub-02 fails at {faults['failure_time'].iloc[0]})")

    windows = build_windows(readings, faults)
    print(f"\n{len(windows)} windows ({WINDOW_H}h trailing, {STEP_H}h step); "
          f"{int(windows['label'].sum())} labeled prefailure "
          f"(window end within {PREFAIL_H}h before failure)")

    print("\nHealthy device, last window:")
    h = windows[windows["device_id"] == "hub-01"].iloc[-1]
    print(h[["window_end", "label"]].to_string())
    print(h[["temp_mean", "vib_mean", "vib_slope", "power_mean"]]
          .astype(float).round(2).to_string())

    print("\nFailing device, window ending 6h before failure:")
    f = windows[windows["device_id"] == "hub-02"].iloc[-2]
    print(f[["window_end", "label"]].to_string())
    print(f[["temp_mean", "vib_mean", "vib_slope", "power_mean"]]
          .astype(float).round(2).to_string())

    print("\nNote the drift signatures the model sees: vibration mean and "
          "trend slope climb on hub-02 while hub-01 stays flat.")


if __name__ == "__main__":
    main()
