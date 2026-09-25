"""Score new sensor readings for prefailure risk.

Usage:
    python src/infer.py [readings_csv]

Reads raw sensor readings (same columns as data/sensor_readings.csv),
builds 24h windows, loads models/anomaly_model.pkl, and prints every window
flagged as anomalous with its anomaly score, plus a per-device alert summary.
"""
import os
import pickle
import sys

import pandas as pd

from preprocess import FEATURE_COLS, build_windows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_READINGS = os.path.join(ROOT, "data", "sensor_readings.csv")
MODEL_PATH = os.path.join(ROOT, "models", "anomaly_model.pkl")


def main():
    readings_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_READINGS
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("train the model first: python src/train.py")
    readings = pd.read_csv(readings_csv)
    windows = build_windows(readings, pd.DataFrame(
        columns=["device_id", "failure_time"]))  # no labels at inference

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    Xs = bundle["scaler"].transform(windows[FEATURE_COLS].to_numpy())
    # lower decision_function => more anomalous; flip sign for readability
    windows["anomaly_score"] = -bundle["model"].decision_function(Xs)
    windows["alert"] = bundle["model"].predict(Xs) == -1

    alerts = windows[windows["alert"]].sort_values("anomaly_score",
                                                   ascending=False)
    print(f"scored {len(windows)} windows from {readings_csv}")
    print(f"{len(alerts)} anomalous windows flagged\n")
    if len(alerts):
        print(alerts[["device_id", "window_end", "anomaly_score"]]
              .head(15).to_string(index=False))
        print("\nper-device alert counts:")
        print(alerts["device_id"].value_counts().to_string())
    else:
        print("no anomalies detected - all devices healthy")


if __name__ == "__main__":
    main()
