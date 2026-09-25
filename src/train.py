"""Train an Isolation Forest for prefailure detection and report metrics.

Method: fit the Isolation Forest on windows from *healthy* periods only
(unsupervised — no fault labels used in training), then score every window.
Windows flagged as anomalies inside the 72h prefailure period count as
detections.

Metrics: precision / recall / F1 on the prefailure class, plus the mean
detection lead time (hours between the first flagged window and the failure).
Model + scaler saved to models/ for infer.py.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURE_COLS, PREFAIL_H, load_windows

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models")


def main():
    df = load_windows()
    X = df[FEATURE_COLS].to_numpy()
    y = df["label"].to_numpy()

    # unsupervised fit: healthy windows only (train on the first 70% of time
    # from devices that never fail, plus healthy windows of faulty devices)
    healthy = df[df["label"] == 0]
    cutoff = healthy["window_end"].quantile(0.7)
    train_idx = healthy[healthy["window_end"] <= cutoff].index
    X_train = X[train_idx]

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X)

    # Operating point: contamination=0.03. In predictive maintenance a missed
    # failure costs far more than a false alarm, so we tune for recall; the
    # threshold is a single knob the operator can turn.
    iso = IsolationForest(n_estimators=300, contamination=0.03,
                          random_state=42, n_jobs=-1)
    iso.fit(scaler.transform(X_train))
    pred = (iso.predict(Xs) == -1).astype(int)  # 1 = anomaly

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    print(f"{len(df)} windows, {int(y.sum())} prefailure "
          f"({y.mean():.2%} positive rate)")
    print(f"precision (prefailure): {precision:.3f}")
    print(f"recall    (prefailure): {recall:.3f}")
    print(f"F1        (prefailure): {f1:.3f}")

    # detection lead time per failure: first flagged window before the failure
    df = df.copy()
    df["pred"] = pred
    df["window_end"] = pd.to_datetime(df["window_end"])
    faults = pd.read_csv(os.path.join(
        os.path.dirname(HERE), "data", "faults.csv"))
    # Detection lead time: for each failure, the earliest *sustained* alert —
    # first of two consecutive flagged windows inside the prefailure period.
    # Requiring a run of 2 keeps stray single-window false alarms from
    # inflating the lead time.
    leads = []
    for _, row in faults.iterrows():
        ft = pd.to_datetime(row["failure_time"])
        dev = df[(df["device_id"] == row["device_id"])].copy()
        dev = dev[(dev["window_end"] < ft) &
                  (dev["window_end"] >= ft - pd.Timedelta(hours=PREFAIL_H))]
        dev = dev.sort_values("window_end").reset_index(drop=True)
        for i in range(len(dev) - 1):
            if dev.loc[i, "pred"] == 1 and dev.loc[i + 1, "pred"] == 1:
                leads.append((ft - dev.loc[i, "window_end"])
                             .total_seconds() / 3600.0)
                break
    detected = len(leads)
    if leads:
        print(f"failures with sustained alert: {detected}/{len(faults)}")
        print(f"mean detection lead time: {np.mean(leads):.1f}h "
              f"(min {np.min(leads):.1f}h, max {np.max(leads):.1f}h)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "anomaly_model.pkl"), "wb") as f:
        pickle.dump({"model": iso, "scaler": scaler,
                     "features": FEATURE_COLS}, f)
    print(f"\nmodel saved -> {MODEL_DIR}/anomaly_model.pkl")


if __name__ == "__main__":
    main()
