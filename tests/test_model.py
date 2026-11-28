import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

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


def _make_sample(**overrides) -> pd.DataFrame:
    row = {
        "number_of_decisions_made": 12,
        "context_switch_count": 6,
        "notifications_received": 25,
        "screen_time_min": 300.0,
        "deep_work_min": 150.0,
        "task_complexity_avg": 4.5,
        "caffeine_mg": 100.0,
        "break_frequency": 3,
        "sleep_hours": 7.5,
        "deep_sleep_pct": 22.0,
        "hydration_l": 2.5,
        "mood": "Calm",
        "work_type": "Focused / Deep Work",
        "work_environment": "Quiet Home / Office",
        "noise_level_db": 40.0,
        "temperature_c": 24.0,
        "workload_score": 45.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def test_model_artifact_loads():
    assert MODEL_PATH.exists(), f"Trained model not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model object does not implement .predict()"


def test_single_and_batch_predictions():
    model = joblib.load(MODEL_PATH)

    calm_df = _make_sample()
    calm_pred = model.predict(calm_df)
    assert len(calm_pred) == 1
    assert np.isfinite(calm_pred[0])

    batch_df = pd.concat(
        [
            _make_sample(),
            _make_sample(
                mood="Stressed",
                work_type="Meetings / Collaboration",
                work_environment="Noisy / Open Office",
                number_of_decisions_made=45,
                context_switch_count=35,
                notifications_received=140,
                screen_time_min=600.0,
                deep_work_min=30.0,
                task_complexity_avg=8.5,
                caffeine_mg=300.0,
                break_frequency=1,
                sleep_hours=4.5,
                deep_sleep_pct=10.0,
                hydration_l=1.0,
                noise_level_db=78.0,
                temperature_c=31.0,
                workload_score=90.0,
            ),
        ],
        ignore_index=True,
    )[FEATURE_COLUMNS]

    batch_preds = model.predict(batch_df)
    assert len(batch_preds) == 2
    assert np.all(np.isfinite(batch_preds))


def test_higher_workload_increases_predicted_tiredness():
    model = joblib.load(MODEL_PATH)

    low_strain = _make_sample()
    high_strain = _make_sample(
        mood="Tired",
        work_type="Mixed",
        work_environment="Noisy / Open Office",
        number_of_decisions_made=40,
        context_switch_count=30,
        notifications_received=120,
        screen_time_min=570.0,
        deep_work_min=30.0,
        task_complexity_avg=8.5,
        caffeine_mg=250.0,
        break_frequency=1,
        sleep_hours=5.0,
        deep_sleep_pct=12.0,
        hydration_l=1.2,
        noise_level_db=75.0,
        temperature_c=30.0,
        workload_score=88.0,
    )

    low_score = float(model.predict(low_strain)[0])
    high_score = float(model.predict(high_strain)[0])
    assert high_score > low_score, (
        f"Expected higher tiredness score for high-strain day ({high_score:.2f}) "
        f"than low-strain day ({low_score:.2f})"
    )


def test_metrics_report_if_present():
    if not METRICS_PATH.exists():
        return
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and len(data) > 0