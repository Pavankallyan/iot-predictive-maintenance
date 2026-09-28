"""Unit tests for src/preprocess.py: windowing + feature engineering.

Uses small synthetic sensor frames so the tests are fast and deterministic
(the repo's full data/sensor_readings.csv is not required).
"""
import numpy as np
import pandas as pd
import pytest

from preprocess import (
    FEATURE_COLS,
    PREFAIL_H,
    SENSORS,
    STEP_H,
    WINDOW_H,
    _slope,
    build_windows,
)


def make_readings(device_id, n_hours, start="2026-01-01",
                  temp=45.0, vib=2.0, power=60.0, vib_ramp=0.0):
    ts = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "device_id": device_id,
        "timestamp": ts,
        "temperature_c": np.full(n_hours, temp),
        "vibration_mm_s": vib + vib_ramp * np.arange(n_hours, dtype=float),
        "power_w": np.full(n_hours, power),
    })


def make_faults(device_id, failure_time):
    return pd.DataFrame({"device_id": [device_id],
                         "failure_time": [failure_time]})


def test_feature_cols_count_and_names():
    # 3 sensors x 5 stats = 15, no duplicates
    assert len(FEATURE_COLS) == 15
    assert len(set(FEATURE_COLS)) == 15
    for prefix in ["temp", "vib", "power"]:
        assert sum(1 for c in FEATURE_COLS if c.startswith(prefix)) == 5


def test_window_count():
    readings = make_readings("hub_01", 30)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    expected = len(range(0, 30 - WINDOW_H + 1, STEP_H))
    assert len(out) == expected
    assert expected == 2


def test_window_end_and_step():
    readings = make_readings("hub_01", 48)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    ends = pd.to_datetime(out["window_end"])
    assert (ends.diff().dropna() == pd.Timedelta(hours=STEP_H)).all()
    assert ends.iloc[0] == pd.Timestamp("2026-01-01 23:00:00")


def test_feature_values_on_flat_signal():
    readings = make_readings("hub_01", 24, temp=50.0, vib=3.0, power=70.0)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    assert len(out) == 1
    row = out.iloc[0]
    for prefix, val in [("temp", 50.0), ("vib", 3.0), ("power", 70.0)]:
        assert row[f"{prefix}_mean"] == pytest.approx(val)
        assert row[f"{prefix}_min"] == pytest.approx(val)
        assert row[f"{prefix}_max"] == pytest.approx(val)
        assert row[f"{prefix}_std"] == pytest.approx(0.0)
        assert row[f"{prefix}_slope"] == pytest.approx(0.0, abs=1e-9)


def test_slope_sign():
    readings = make_readings("hub_01", 24, vib_ramp=0.5)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    assert out.iloc[0]["vib_slope"] == pytest.approx(0.5, rel=1e-6)
    assert out.iloc[0]["temp_slope"] == pytest.approx(0.0, abs=1e-9)


def test_slope_helper():
    assert _slope(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0, abs=1e-9)
    assert _slope(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0, rel=1e-6)
    assert _slope(np.array([4.0, 3.0, 2.0, 1.0])) < 0


def test_prefailure_label():
    # 30 hourly readings: windows end at hour 23 and hour 29.
    # Failure at hour 24 -> first window is 1h before failure (label 1),
    # second window ends 5h after failure (label 0).
    readings = make_readings("hub_01", 30)
    faults = make_faults("hub_01", "2026-01-02 00:00:00")
    out = build_windows(readings, faults)
    assert out["label"].tolist() == [1, 0]


def test_prefailure_boundary_72h():
    readings = make_readings("hub_01", 24)
    window_end = pd.Timestamp("2026-01-01 23:00:00")
    faults = make_faults("hub_01", window_end + pd.Timedelta(hours=PREFAIL_H))
    out = build_windows(readings, faults)
    assert out["label"].iloc[0] == 1
    faults2 = make_faults("hub_01",
                          window_end + pd.Timedelta(hours=PREFAIL_H + 1))
    out2 = build_windows(readings, faults2)
    assert out2["label"].iloc[0] == 0


def test_no_fault_means_label_zero():
    readings = make_readings("hub_01", 48)
    faults = make_faults("hub_02", "2026-01-02 12:00:00")  # other device
    out = build_windows(readings, faults)
    assert (out["label"] == 0).all()


def test_multiple_devices():
    r1 = make_readings("hub_01", 30)
    r2 = make_readings("hub_02", 30)
    faults = make_faults("hub_01", "2026-01-02 00:00:00")
    out = build_windows(pd.concat([r1, r2], ignore_index=True), faults)
    assert len(out) == 4
    assert set(out["device_id"]) == {"hub_01", "hub_02"}
    hub1 = out[out["device_id"] == "hub_01"]
    hub2 = out[out["device_id"] == "hub_02"]
    assert hub1["label"].tolist() == [1, 0]
    assert (hub2["label"] == 0).all()


def test_unsorted_input_still_chronological():
    readings = make_readings("hub_01", 48).sample(frac=1.0, random_state=7)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    ends = pd.to_datetime(out["window_end"])
    assert ends.is_monotonic_increasing
    assert len(out) == len(range(0, 48 - WINDOW_H + 1, STEP_H))


def test_metadata_columns_present():
    readings = make_readings("hub_01", 24)
    faults = pd.DataFrame(columns=["device_id", "failure_time"])
    out = build_windows(readings, faults)
    for col in ["device_id", "window_end", "label"]:
        assert col in out.columns
    assert out["device_id"].iloc[0] == "hub_01"
