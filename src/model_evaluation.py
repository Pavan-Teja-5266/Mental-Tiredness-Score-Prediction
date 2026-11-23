"""Final evaluation of the automatically selected champion on the untouched
test set.

The champion is chosen in model_building.py using ONLY training data
(cross-validation + size/time/gap criteria). This script never changes that
choice; test metrics for the other models are reported for comparison only.
"""
from pathlib import Path
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
SELECTION_PATH = Path("reports/model_selection.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
METRICS_PATH = Path("reports/metrics.json")
DETAILS_PATH = Path("reports/evaluation_details.json")
TARGET = "mental_tiredness_score"

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "SVR",
    "Ridge",
    "RandomForest",
    "GradientBoosting",
]


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, predictions))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def residual_summary(model, X_test, y_test):
    predictions = model.predict(X_test)
    residuals = y_test.to_numpy(dtype=float) - np.asarray(predictions, dtype=float)
    abs_errors = np.abs(residuals)
    return {
        "test_samples": int(len(y_test)),
        "y_test_mean": float(np.mean(y_test)),
        "y_test_std": float(np.std(y_test)),
        "y_pred_mean": float(np.mean(predictions)),
        "y_pred_std": float(np.std(predictions)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "abs_error_median": float(np.median(abs_errors)),
        "abs_error_p90": float(np.percentile(abs_errors, 90)),
    }


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET].astype(float)

    with open(SELECTION_PATH, "r", encoding="utf-8") as file:
        selection = json.load(file)

    champion_info = selection["champion"]
    champion_name = champion_info["model_name"]
    champion_path = Path(selection["champion_model_path"])
    champion = joblib.load(champion_path)
    criteria = champion_info["criteria"]
    policy = selection["selection_policy"]

    # Test metrics for ALL candidates are for comparison/reporting only.
    # They are never used to pick or change the champion.
    all_results = {}
    for model_name in MODEL_NAMES:
        model_file = Path("models") / f"{model_name}.pkl"
        if model_file.exists():
            model = joblib.load(model_file)
            all_results[model_name] = evaluate_model(model, X_test, y_test)

    champion_metrics = evaluate_model(champion, X_test, y_test)

    # Flat, numeric-only file so `dvc metrics show` stays readable.
    metrics = {
        "rmse": champion_metrics["rmse"],
        "mae": champion_metrics["mae"],
        "r2": champion_metrics["r2"],
        "champion_cv_val_rmse": float(champion_info["best_cv_val_rmse"]),
        "champion_training_rmse": float(criteria["training_rmse"]),
        "champion_validation_rmse": float(criteria["validation_rmse"]),
        "champion_rmse_gap": float(criteria["rmse_gap"]),
        "champion_model_size_mb": float(criteria["model_size_mb"]),
        "champion_training_time_seconds": float(criteria["training_time_seconds"]),
        "champion_prediction_time_ms": float(criteria["prediction_time_ms"]),
        "test_vs_validation_rmse_diff": float(
            champion_metrics["rmse"] - criteria["validation_rmse"]
        ),
    }

    details = {
        "target": TARGET,
        "champion_model": champion_name,
        "champion_display_name": champion_info.get("display_name", champion_name),
        "selection_basis": (
            "automated multi-criteria selection: RMSE gap strictly below "
            f"{policy.get('max_rmse_gap (strictly below)', policy.get('max_rmse_gap'))}, "
            "then lowest validation RMSE (within tolerance), then smaller model size, "
            "lower training time and lower prediction time"
        ),
        "selection_explanation": champion_info["explanation"],
        "selection_policy": policy,
        "selection_comparison": {
            name: {
                k: selection[name][k]
                for k in (
                    "model_size_mb",
                    "training_time_seconds",
                    "prediction_time_ms",
                    "training_rmse",
                    "validation_rmse",
                    "rmse_gap",
                    "eligible",
                    "status",
                )
            }
            for name in MODEL_NAMES
            if name in selection
        },
        "all_models_test_metrics": all_results,
        "champion_test_metrics": champion_metrics,
        "champion_residual_summary": residual_summary(champion, X_test, y_test),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    with open(DETAILS_PATH, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=4)

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.log_param("champion_model", champion_name)
        mlflow.log_param("selection_method", "multi_criteria_gap_rmse_size_time")
        mlflow.log_metrics({
            "test_rmse": champion_metrics["rmse"],
            "test_mae": champion_metrics["mae"],
            "test_r2": champion_metrics["r2"],
            "champion_training_rmse": float(criteria["training_rmse"]),
            "champion_validation_rmse": float(criteria["validation_rmse"]),
            "champion_rmse_gap": float(criteria["rmse_gap"]),
            "champion_model_size_mb": float(criteria["model_size_mb"]),
            "champion_training_time_seconds": float(criteria["training_time_seconds"]),
            "champion_prediction_time_ms": float(criteria["prediction_time_ms"]),
        })
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(DETAILS_PATH))
        mlflow.log_artifact(str(SELECTION_PATH))
        if COMPARISON_CSV_PATH.exists():
            mlflow.log_artifact(str(COMPARISON_CSV_PATH))

    print(f"Champion model: {champion_name}")
    print(f"Why: {champion_info['explanation']}")
    print(
        f"Validation RMSE: {criteria['validation_rmse']:.4f}  "
        f"Gap: {criteria['rmse_gap']:+.4f}"
    )
    print(f"Test RMSE: {champion_metrics['rmse']:.4f}")
    print(f"Test MAE:  {champion_metrics['mae']:.4f}")
    print(f"Test R2:   {champion_metrics['r2']:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()