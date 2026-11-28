from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "mental_tiredness_score_prediction_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

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


def test_raw_dataset_columns_and_not_empty():
    assert RAW_DATA_PATH.exists(), f"Missing dataset at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)

    assert not df.empty, "Dataset is empty"
    expected_cols = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing expected columns: {sorted(missing)}"


def test_model_prediction_output():
    assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
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
        ]
    )[FEATURE_COLUMNS]

    preds = model.predict(sample)
    assert len(preds) == 1
    assert pd.notna(preds[0])