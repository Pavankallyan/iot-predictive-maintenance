"""Generate synthetic smart-home hub sensor data with injected fault patterns.

Simulates N_DEVICES smart hubs reporting hourly: temperature (C), vibration
(mm/s) and power draw (W) for N_DAYS. A subset of devices develop faults:

- bearing_wear : vibration drifts up exponentially over ~5 days pre-failure,
                 temperature creeps up with it
- overheating  : temperature spikes intermittently, then stays high
- power_fault  : power draw turns erratic with dropouts

Outputs:
    data/sensor_readings.csv - device_id, timestamp, temperature_c,
                               vibration_mm_s, power_w
    data/faults.csv          - device_id, fault_type, failure_time
Prefailure windows (72h before each failure) are the positive class.
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

N_DEVICES = 25
N_DAYS = 60
N_FAULTY = 8

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

FAULT_TYPES = ["bearing_wear", "overheating", "power_fault"]


def _daily_cycle(t, amplitude, phase=0.0):
    return amplitude * np.sin(2 * np.pi * (t + phase) / 24.0)


def simulate_device(device_id, fault=None, failure_hour=None):
    n = N_DAYS * 24
    t = np.arange(n, dtype=float)
    temp = 45.0 + _daily_cycle(t, 5.0) + RNG.normal(0, 0.5, n)
    vib = 2.0 + RNG.normal(0, 0.2, n)
    power = 60.0 + _daily_cycle(t, 10.0, phase=6.0) + RNG.normal(0, 1.0, n)

    if fault == "bearing_wear":
        ramp = np.clip((t - (failure_hour - 120)) / 120.0, 0, 1) ** 2
        vib = vib + ramp * 10.0
        temp = temp + ramp * 8.0
    elif fault == "overheating":
        spikes = (RNG.random(n) < 0.05) & (t > failure_hour - 120)
        temp = temp + spikes * RNG.uniform(8, 15, n)
        temp = np.where(t > failure_hour - 48, temp + 12.0, temp)
    elif fault == "power_fault":
        bad = t > failure_hour - 96
        power = np.where(bad, power * RNG.uniform(0.3, 1.1, n), power)

    base = pd.Timestamp("2026-01-01")
    return pd.DataFrame({
        "device_id": device_id,
        "timestamp": base + pd.to_timedelta(t, unit="h"),
        "temperature_c": np.round(temp, 2),
        "vibration_mm_s": np.round(vib, 3),
        "power_w": np.round(power, 2),
    })


def generate():
    faulty_ids = RNG.choice(N_DEVICES, size=N_FAULTY, replace=False)
    frames, faults = [], []
    for i in range(N_DEVICES):
        device_id = f"hub_{i:02d}"
        fault, failure_hour = None, None
        if i in faulty_ids:
            fault = FAULT_TYPES[i % len(FAULT_TYPES)]
            # failures land in the last 30 days so prefailure windows exist
            failure_hour = int(RNG.integers(30 * 24, N_DAYS * 24))
            faults.append({"device_id": device_id, "fault_type": fault,
                           "failure_time": pd.Timestamp("2026-01-01")
                           + pd.to_timedelta(failure_hour, unit="h")})
        frames.append(simulate_device(device_id, fault, failure_hour))

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "sensor_readings.csv"), index=False)
    pd.DataFrame(faults).to_csv(os.path.join(DATA_DIR, "faults.csv"), index=False)
    print(f"{N_DEVICES} devices x {N_DAYS * 24} hourly readings, "
          f"{N_FAULTY} faults -> {DATA_DIR}")


if __name__ == "__main__":
    generate()
