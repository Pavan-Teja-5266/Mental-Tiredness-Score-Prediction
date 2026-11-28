import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
EVAL_DETAILS_PATH = BASE_DIR / "reports" / "evaluation_details.json"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "mental_tiredness_score_prediction_dataset.csv"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

FEATURE_COLUMNS = [
    "number_of_decisions_made",
    "context_switch_count",
    "notifications_received",
    "screen_time_min",
    "deep_work_min",
    "task_complexity_avg",
    "caffeine_mg",
    "break_frequency",
    "sleep_hours",
    "deep_sleep_pct",
    "hydration_l",
    "mood",
    "work_type",
    "work_environment",
    "noise_level_db",
    "temperature_c",
    "workload_score",
]

TARGET_COLUMN = "mental_tiredness_score"


def test_monitoring_artifacts_and_metrics():
    assert MODEL_PATH.exists(), f"Missing model artifact at {MODEL_PATH}"
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        assert isinstance(metrics, dict) and len(metrics) > 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"Non-finite metric value for '{key}': {value}"

    if EVAL_DETAILS_PATH.exists():
        details = json.loads(EVAL_DETAILS_PATH.read_text(encoding="utf-8"))
        assert isinstance(details, dict) and len(details) > 0


def test_prediction_distribution_and_drift_check():
    model = joblib.load(MODEL_PATH)

    if X_TEST_PATH.exists() and Y_TEST_PATH.exists():
        X_ref = pd.read_csv(X_TEST_PATH)[FEATURE_COLUMNS]
        y_ref = pd.read_csv(Y_TEST_PATH).squeeze("columns")
    elif RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        sample_df = df.sample(n=min(200, len(df)), random_state=42)
        X_ref = sample_df[FEATURE_COLUMNS]
        y_ref = sample_df[TARGET_COLUMN]
    else:
        return

    preds = np.asarray(model.predict(X_ref), dtype=float)
    assert len(preds) == len(X_ref)
    assert np.all(np.isfinite(preds)), "Model produced NaN or Inf predictions"

    ref_mean = float(np.mean(y_ref))
    pred_mean = float(np.mean(preds))
    ref_std = max(float(np.std(y_ref)), 1e-6)

    # Flag severe mean shift (> 3 standard deviations away from reference target mean)
    z_shift = abs(pred_mean - ref_mean) / ref_std
    assert z_shift < 3.0, (
        f"Potential prediction drift detected: pred_mean={pred_mean:.2f} vs "
        f"ref_mean={ref_mean:.2f} (z_shift={z_shift:.2f})"
    )


def test_missing_or_extreme_input_resilience():
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame(
        [
            {
                "number_of_decisions_made": 0,
                "context_switch_count": 0,
                "notifications_received": 0,
                "screen_time_min": 0.0,
                "deep_work_min": 0.0,
                "task_complexity_avg": 1.0,
                "caffeine_mg": 0.0,
                "break_frequency": 0,
                "sleep_hours": 0.0,
                "deep_sleep_pct": 0.0,
                "hydration_l": 0.0,
                "mood": "Neutral",
                "work_type": "Mixed",
                "work_environment": "Quiet Home / Office",
                "noise_level_db": 0.0,
                "temperature_c": 0.0,
                "workload_score": 0.0,
            },
            {
                "number_of_decisions_made": 100,
                "context_switch_count": 80,
                "notifications_received": 300,
                "screen_time_min": 900.0,
                "deep_work_min": 480.0,
                "task_complexity_avg": 10.0,
                "caffeine_mg": 600.0,
                "break_frequency": 10,
                "sleep_hours": 12.0,
                "deep_sleep_pct": 50.0,
                "hydration_l": 5.0,
                "mood": "Energized",
                "work_type": "Focused / Deep Work",
                "work_environment": "Hybrid / Travel",
                "noise_level_db": 95.0,
                "temperature_c": 40.0,
                "workload_score": 100.0,
            },
        ]
    )[FEATURE_COLUMNS]

    preds = model.predict(df)
    assert len(preds) == 2
    assert np.all(np.isfinite(preds))