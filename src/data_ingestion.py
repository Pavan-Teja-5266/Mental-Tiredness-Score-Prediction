from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/mental_tiredness_score_prediction_dataset.csv")
OUTPUT_PATH = Path("data/ingested/mental_tiredness_score.csv")

REQUIRED_COLUMNS = [
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
    "mental_tiredness_score",
]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}. Put the mental tiredness CSV at this location."
        )

    data = pd.read_csv(RAW_PATH)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only required features and target in expected order, then drop exact duplicates
    data = data[REQUIRED_COLUMNS].drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError("The input dataset is empty after cleaning.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()