"""Unit tests for src/train.py: model fitting + sustained-alert lead times.

Uses small synthetic window frames and feature matrices so the tests are
fast and deterministic (the repo's full data/sensor_readings.csv is not
required). One integration test runs the real build_windows -> fit_model
path on a small synthetic fleet.
"""
import numpy as np
import pandas as pd

from preprocess import FEATURE_COLS, build_windows
from train import fit_model, sustained_lead_times

FT = pd.Timestamp("2026-01-10 00:00")


def make_windows(device_id, failure_time, flags, start_offset_h=-72):
    """Windows ending every 6h from failure_time+start_offset_h up to -6h.

    `flags` is a 0/1 list, one entry per window, in chronological order.
    """
    ends = pd.date_range(failure_time + pd.Timedelta(hours=start_offset_h),
                         failure_time - pd.Timedelta(hours=6), freq="6h")
    assert len(ends) == len(flags), "one flag per window"
    return pd.DataFrame({
        "device_id": device_id,
        "window_end": ends,
        "pred": flags,
    })


def make_faults(device_id, failure_time):
    return pd.DataFrame({"device_id": [device_id],
                         "failure_time": [failure_time]})


def make_healthy_X(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n, len(FEATURE_COLS)))


# --- sustained_lead_times -------------------------------------------------

def test_lead_time_first_consecutive_pair():
    windows = make_windows("hub_01", FT, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    leads = sustained_lead_times(windows, make_faults("hub_01", FT))
    # first flagged window ends 48h before the failure
    assert leads == [48.0]


def test_lead_time_earliest_pair_wins():
    windows = make_windows("hub_01", FT, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    leads = sustained_lead_times(windows, make_faults("hub_01", FT))
    assert leads == [60.0]


def test_lead_time_single_blip_ignored():
    windows = make_windows("hub_01", FT, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert sustained_lead_times(windows, make_faults("hub_01", FT)) == []


def test_lead_time_no_flags():
    windows = make_windows("hub_01", FT, [0] * 12)
    assert sustained_lead_times(windows, make_faults("hub_01", FT)) == []


def test_lead_time_flags_before_prefailure_window_ignored():
    # flags at -120h/-114h only: outside the 72h prefailure period
    windows = make_windows("hub_01", FT, [1, 1] + [0] * 18,
                           start_offset_h=-120)
    assert sustained_lead_times(windows, make_faults("hub_01", FT)) == []


def test_lead_time_window_ending_at_failure_excluded():
    ends = pd.date_range(FT - pd.Timedelta(hours=12), FT, freq="6h")
    windows = pd.DataFrame({"device_id": "hub_01", "window_end": ends,
                            "pred": [0, 1, 1]})
    # the window ending exactly at failure time is out of range, so the
    # remaining single flag is not a sustained alert
    assert sustained_lead_times(windows, make_faults("hub_01", FT)) == []


def test_lead_time_multiple_failures():
    w1 = make_windows("hub_01", FT, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    ft2 = FT + pd.Timedelta(days=5)
    w2 = make_windows("hub_02", ft2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
    faults = pd.DataFrame({"device_id": ["hub_01", "hub_02"],
                           "failure_time": [FT, ft2]})
    leads = sustained_lead_times(pd.concat([w1, w2], ignore_index=True),
                                 faults)
    assert leads == [48.0, 18.0]


def test_lead_time_unsorted_windows():
    windows = make_windows("hub_01", FT, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    shuffled = windows.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert sustained_lead_times(shuffled, make_faults("hub_01", FT)) == [48.0]


def test_lead_time_failure_with_no_windows():
    windows = make_windows("hub_01", FT, [1, 1] + [0] * 10)
    faults = pd.DataFrame({"device_id": ["hub_99"],
                           "failure_time": [FT]})
    assert sustained_lead_times(windows, faults) == []


# --- fit_model ------------------------------------------------------------

def test_fit_model_returns_fitted_model_and_scaler():
    model, scaler = fit_model(make_healthy_X())
    assert model.n_estimators == 300
    assert model.contamination == 0.03
    assert scaler.mean_.shape == (len(FEATURE_COLS),)
    assert scaler.scale_.shape == (len(FEATURE_COLS),)


def test_fit_model_predicts_only_plus_or_minus_one():
    X = make_healthy_X()
    model, scaler = fit_model(X)
    pred = model.predict(scaler.transform(X))
    assert set(np.unique(pred)) <= {-1, 1}
    # contamination=0.03 flags ~3% of the training windows
    frac_flagged = (pred == -1).mean()
    assert 0.005 <= frac_flagged <= 0.10


def test_fit_model_deterministic():
    X = make_healthy_X()
    m1, s1 = fit_model(X)
    m2, s2 = fit_model(X)
    assert np.array_equal(m1.predict(s1.transform(X)),
                          m2.predict(s2.transform(X)))


def test_fit_model_outlier_scores_more_anomalous():
    X = make_healthy_X()
    model, scaler = fit_model(X)
    typical = np.zeros((1, len(FEATURE_COLS)))
    outlier = np.full((1, len(FEATURE_COLS)), 10.0)
    scores = model.decision_function(scaler.transform(
        np.vstack([typical, outlier])))
    assert np.isfinite(scores).all()
    assert scores[1] < scores[0]  # lower decision_function = more anomalous


# --- integration: build_windows -> fit_model -------------------------------

def test_integration_windows_fit_predict():
    ts = pd.date_range("2026-01-01", periods=96, freq="h")
    healthy = pd.DataFrame({
        "device_id": "hub_01", "timestamp": ts,
        "temperature_c": 45.0, "vibration_mm_s": 2.0, "power_w": 60.0})
    faulty = pd.DataFrame({
        "device_id": "hub_02", "timestamp": ts,
        "temperature_c": 45.0,
        "vibration_mm_s": 2.0 + np.linspace(0, 12, 96),
        "power_w": 60.0})
    readings = pd.concat([healthy, faulty], ignore_index=True)
    faults = pd.DataFrame({"device_id": ["hub_02"],
                           "failure_time": [ts[-1]]})
    windows = build_windows(readings, faults)
    assert len(windows) > 0
    X = windows[FEATURE_COLS].to_numpy()
    X_train = X[windows["label"].to_numpy() == 0]
    model, scaler = fit_model(X_train)
    pred = (model.predict(scaler.transform(X)) == -1).astype(int)
    assert len(pred) == len(windows)
    assert set(np.unique(pred)) <= {0, 1}
    windows = windows.copy()
    windows["pred"] = pred
    leads = sustained_lead_times(windows, faults)
    assert isinstance(leads, list)
    assert all(isinstance(lead, float) and 0 < lead <= 72 for lead in leads)
