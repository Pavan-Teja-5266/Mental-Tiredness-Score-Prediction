from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/mental_tiredness_score.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
TARGET = "mental_tiredness_score"


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ingested dataset not found: {INPUT_PATH}. Run the data ingestion stage first."
        )

    data = pd.read_csv(INPUT_PATH)

    # Remove exact duplicate records before splitting.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate rows.")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    # Drop rows with missing target values if any exist before splitting
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    test_size = float(params.get("test_size", params.get("data", {}).get("test_size", 0.3)))
    random_state = int(params.get("random_state", params.get("project", {}).get("random_state", 42)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Training data: {train.shape}")
    print(f"Testing data: {test.shape}")
    print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()