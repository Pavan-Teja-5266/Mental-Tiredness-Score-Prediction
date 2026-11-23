"""Model building with automated multi-criteria champion selection (regression).

Step 1  Train candidate regression models (Optuna search + K-fold CV per model).
Step 2  Measure key criteria per model:
        model size (MB), training time (s), prediction time (ms),
        training RMSE, validation RMSE, and RMSE gap (val RMSE - train RMSE).
Step 3  Apply selection rules:
- RMSE gap must be strictly below the configured threshold
- lower validation RMSE preferred (finalists within rmse_tolerance of best)
- smaller size / lower training time / lower prediction time preferred
Step 4  Select the champion automatically and write reports/model_selection.json
        and models/model.pkl.

The test set is NEVER read here. Final evaluation happens in model_evaluation.py.
"""
from pathlib import Path
import json
import os
import shutil
import time
import warnings

# Set before importing scikit-learn/joblib components.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SELECTION_PATH = REPORTS_DIR / "model_selection.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD_PATH = REPORTS_DIR / "model_comparison.md"
TARGET = "mental_tiredness_score"

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "SVR",
    "Ridge",
    "RandomForest",
    "GradientBoosting",
]
DISPLAY_NAMES = {
    "KNN": "KNN Regressor",
    "DecisionTree": "Decision Tree Regressor",
    "SVR": "SVR",
    "Ridge": "Ridge Regression",
    "RandomForest": "Random Forest Regressor",
    "GradientBoosting": "Gradient Boosting Regressor",
}


def build_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler_type: {scaler_type}")


def build_preprocessor(X, scaler_type):
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", build_scaler(scaler_type)),
    ]
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def suggest_params(model_name, trial, random_state):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax", "robust"])

    if model_name == "KNN":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 3),
        }
    elif model_name == "DecisionTree":
        params = {
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    elif model_name == "SVR":
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
        }
        if kernel in ["rbf", "poly"]:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
    elif model_name == "Ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "lsqr", "cholesky"]),
        }
    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return scaler_type, params


def build_pipeline(model_name, params, scaler_type, X, random_state):
    preprocessor = build_preprocessor(X, scaler_type)

    if model_name == "KNN":
        estimator = KNeighborsRegressor(**params)
    elif model_name == "DecisionTree":
        estimator = DecisionTreeRegressor(**params, random_state=random_state)
    elif model_name == "SVR":
        estimator = SVR(**params)
    elif model_name == "Ridge":
        estimator = Ridge(**params, random_state=random_state)
    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    elif model_name == "GradientBoosting":
        estimator = GradientBoostingRegressor(**params, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("Preprocessor", preprocessor), ("Model", estimator)])


# --------------------------------------------------------------------------
# Step 2: criteria measurement helpers
# --------------------------------------------------------------------------
def fit_with_timing(model_name, params, scaler_type, X, y, repeats, random_state):
    """Fit the final pipeline `repeats` times; return last fitted model and
    median wall-clock fit time in seconds."""
    times = []
    pipeline = None
    for _ in range(max(1, repeats)):
        pipeline = build_pipeline(model_name, params, scaler_type, X, random_state)
        start = time.perf_counter()
        pipeline.fit(X, y)
        times.append(time.perf_counter() - start)
    return pipeline, float(np.median(times))


def measure_prediction_ms(model, X, n_samples, repeats, seed):
    """Median latency (ms) of ONE single-row prediction using training data only."""
    sample = X.sample(n=min(n_samples, len(X)), random_state=seed)
    rows = [sample.iloc[[i]] for i in range(len(sample))]

    for row in rows[:3]:  # warm-up
        model.predict(row)

    per_repeat_ms = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        for row in rows:
            model.predict(row)
        per_repeat_ms.append((time.perf_counter() - start) / len(rows) * 1000.0)
    return float(np.median(per_repeat_ms))


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# --------------------------------------------------------------------------
# Step 3 + 4: selection logic (lower RMSE is better)
# --------------------------------------------------------------------------
def _normalise(values):
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def select_champion(results, cfg):
    """Apply regression selection rules. Mutates `results` and returns
    (champion_name, explanation, fallback_used, ranked)."""
    gap_limit = float(cfg.get("max_rmse_gap", 1.0))
    tolerance = float(cfg.get("rmse_tolerance", 0.25))
    weights = cfg["weights"]

    # Rule 1: RMSE gap (val_rmse - train_rmse) must be STRICTLY below gap_limit.
    for r in results.values():
        r["eligible"] = bool(r["rmse_gap"] < gap_limit)

    eligible = {n: r for n, r in results.items() if r["eligible"]}
    fallback_used = False

    if not eligible:
        mode = cfg.get("fallback_if_none_eligible", "smallest_gap")
        if mode == "error":
            raise RuntimeError(
                f"No model has an RMSE gap strictly below {gap_limit}."
            )
        warnings.warn("No eligible model; falling back to smallest RMSE gap.")
        fallback_used = True
        eligible = {n: r for n, r in results.items()}

    # Rule 2: lowest validation RMSE first. Models within `tolerance` of best
    # validation RMSE are finalists.
    best_val_rmse = min(r["validation_rmse"] for r in eligible.values())
    if fallback_used:
        smallest_gap = min(r["rmse_gap"] for r in eligible.values())
        finalists = {
            n: r for n, r in eligible.items()
            if abs(r["rmse_gap"] - smallest_gap) < 1e-9
        }
    else:
        finalists = {
            n: r for n, r in eligible.items()
            if (r["validation_rmse"] - best_val_rmse) <= tolerance + 1e-9
        }

    # Rules 3-5: smaller size, lower training time, lower prediction time.
    norm_size = _normalise({n: r["model_size_mb"] for n, r in finalists.items()})
    norm_train = _normalise({n: r["training_time_seconds"] for n, r in finalists.items()})
    norm_pred = _normalise({n: r["prediction_time_ms"] for n, r in finalists.items()})

    for n, r in finalists.items():
        r["selection_cost"] = float(
            weights["model_size_mb"] * norm_size[n]
            + weights["training_time_seconds"] * norm_train[n]
            + weights["prediction_time_ms"] * norm_pred[n]
        )

    ranked = sorted(
        finalists,
        key=lambda n: (
            round(finalists[n]["selection_cost"], 12),
            finalists[n]["validation_rmse"],
            n,
        ),
    )
    champion = ranked[0]

    for n, r in results.items():
        if n == champion:
            r["status"] = "champion"
        elif n in finalists:
            r["status"] = "finalist_runner_up"
        elif r["eligible"]:
            r["status"] = "eligible_but_above_rmse_tolerance"
        else:
            r["status"] = f"rejected_rmse_gap>={gap_limit}"

    for rank, n in enumerate(ranked, start=1):
        results[n]["finalist_rank"] = rank

    explanation = (
        f"{DISPLAY_NAMES[champion]} selected: "
        f"{'no model met the RMSE-gap rule, so smallest gap was used' if fallback_used else f'passed RMSE-gap rule (< {gap_limit})'}; "
        f"{len(finalists)} finalist(s) within {tolerance} of best validation "
        f"RMSE ({best_val_rmse:.4f}); lowest weighted size/train-time/predict-time cost."
    )
    return champion, explanation, fallback_used, ranked


def write_comparison_reports(results, champion):
    rows = []
    for name in MODEL_NAMES:
        r = results[name]
        rows.append({
            "Model": DISPLAY_NAMES[name],
            "Size MB": round(r["model_size_mb"], 4),
            "Train time s": round(r["training_time_seconds"], 4),
            "Predict ms": round(r["prediction_time_ms"], 4),
            "Train RMSE": round(r["training_rmse"], 4),
            "Val RMSE": round(r["validation_rmse"], 4),
            "RMSE gap": round(r["rmse_gap"], 4),
            "Eligible": "Yes" if r["eligible"] else "No",
            "Status": r["status"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV_PATH, index=False)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    md = "\n".join(
        ["# Model Comparison", "", header, sep, *body, "",
         f"**Champion:** {DISPLAY_NAMES[champion]}", ""]
    )
    COMPARISON_MD_PATH.write_text(md, encoding="utf-8")
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    sel_cfg = config["selection"]
    meas_cfg = config["measurement"]
    gap_limit = float(sel_cfg.get("max_rmse_gap", 1.0))
    use_penalty = bool(sel_cfg.get("gap_penalty_in_search", True))
    penalty_weight = float(sel_cfg.get("gap_penalty_weight", 1.0))

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET].astype(float)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    random_state = int(config["random_state"])
    cv_folds = int(config["cv_folds"])
    n_trials = int(config["n_trials"])

    mlflow.set_experiment(config["experiment_name"])

    results = {}

    # ---------------- Step 1 + 2: train all models, measure criteria ------
    for model_name in MODEL_NAMES:
        print(f"\n--- Optimizing {model_name} ---")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("max_rmse_gap", gap_limit)

            def objective(trial):
                scaler_type, model_params = suggest_params(model_name, trial, random_state)
                candidate = build_pipeline(
                    model_name, model_params, scaler_type, X_train, random_state
                )
                cv = KFold(
                    n_splits=cv_folds, shuffle=True, random_state=random_state
                )
                cv_result = cross_validate(
                    candidate,
                    X_train,
                    y_train,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    n_jobs=1,
                    return_train_score=True,
                )
                val_rmse = float(-cv_result["test_score"].mean())
                train_rmse = float(-cv_result["train_score"].mean())
                rmse_gap = float(val_rmse - train_rmse)

                # Optuna minimizes objective_value; penalize high RMSE gap.
                objective_value = val_rmse
                if use_penalty and rmse_gap >= gap_limit:
                    objective_value += penalty_weight * (rmse_gap - gap_limit + 1.0)

                trial.set_user_attr("cv_validation_rmse", val_rmse)
                trial.set_user_attr("cv_training_rmse", train_rmse)
                trial.set_user_attr("cv_rmse_gap", rmse_gap)

                with mlflow.start_run(
                    run_name=f"{model_name}_trial_{trial.number}", nested=True
                ):
                    mlflow.log_param("scaler_type", scaler_type)
                    mlflow.log_params(model_params)
                    mlflow.log_metric("cv_val_rmse", val_rmse)
                    mlflow.log_metric("cv_train_rmse", train_rmse)
                    mlflow.log_metric("cv_rmse_gap", rmse_gap)
                    mlflow.log_metric("objective_value", float(objective_value))

                return float(objective_value)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            search_start = time.time()
            study.optimize(objective, n_trials=n_trials)
            search_time = time.time() - search_start

            best_scaler = study.best_params["scaler_type"]
            best_model_params = {
                k: v for k, v in study.best_params.items() if k != "scaler_type"
            }
            best_cv_val_rmse = float(study.best_trial.user_attrs["cv_validation_rmse"])

            final_pipeline, train_time = fit_with_timing(
                model_name,
                best_model_params,
                best_scaler,
                X_train,
                y_train,
                int(meas_cfg["training_time_repeats"]),
                random_state,
            )

            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(final_pipeline, model_path)
            size_mb = file_size_mb(model_path)

            train_rmse = float(
                root_mean_squared_error(y_train, final_pipeline.predict(X_train))
            )
            val_rmse = best_cv_val_rmse
            rmse_gap = float(val_rmse - train_rmse)

            pred_ms = measure_prediction_ms(
                final_pipeline,
                X_train,
                int(meas_cfg["prediction_n_samples"]),
                int(meas_cfg["prediction_repeats"]),
                random_state,
            )

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metrics({
                "best_cv_val_rmse": best_cv_val_rmse,
                "training_rmse": train_rmse,
                "validation_rmse": val_rmse,
                "rmse_gap": rmse_gap,
                "model_size_mb": size_mb,
                "training_time_seconds": train_time,
                "prediction_time_ms": pred_ms,
                "search_time_seconds": float(search_time),
            })
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.sklearn.log_model(
                final_pipeline,
                name=f"{model_name}_mental_tiredness_model",
                serialization_format="cloudpickle",
            )

            results[model_name] = {
                "best_params": study.best_params,
                "model_size_mb": float(size_mb),
                "training_time_seconds": float(train_time),
                "prediction_time_ms": float(pred_ms),
                "training_rmse": float(train_rmse),
                "validation_rmse": float(val_rmse),
                "rmse_gap": float(rmse_gap),
                "search_time_seconds": float(search_time),
                "model_path": str(model_path).replace("\\", "/"),
            }

            print(
                f"{model_name}: val_rmse={val_rmse:.4f} train_rmse={train_rmse:.4f} "
                f"gap={rmse_gap:+.4f} size={size_mb:.3f}MB "
                f"fit={train_time:.3f}s predict={pred_ms:.3f}ms"
            )

    # ---------------- Step 3 + 4: rules and automatic champion ------------
    champion, explanation, fallback_used, ranked = select_champion(results, sel_cfg)

    champion_path = MODELS_DIR / "model.pkl"
    shutil.copy2(MODELS_DIR / f"{champion}.pkl", champion_path)

    champ = results[champion]
    selection = dict(results)
    selection["champion"] = {
        "model_name": champion,
        "best_cv_val_rmse": champ["validation_rmse"],
        "display_name": DISPLAY_NAMES[champion],
        "criteria": {
            "model_size_mb": champ["model_size_mb"],
            "training_time_seconds": champ["training_time_seconds"],
            "prediction_time_ms": champ["prediction_time_ms"],
            "training_rmse": champ["training_rmse"],
            "validation_rmse": champ["validation_rmse"],
            "rmse_gap": champ["rmse_gap"],
        },
        "explanation": explanation,
    }
    selection["champion_model_path"] = str(champion_path).replace("\\", "/")
    selection["selection_policy"] = {
        "max_rmse_gap (strictly below)": gap_limit,
        "rmse_tolerance": float(sel_cfg.get("rmse_tolerance", 0.25)),
        "weights": sel_cfg["weights"],
        "fallback_used": fallback_used,
        "finalist_ranking": ranked,
        "selection_data": "training data only (CV validation); test set untouched",
    }

    with open(SELECTION_PATH, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=4)

    comparison = write_comparison_reports(results, champion)

    with mlflow.start_run(run_name="champion_selection"):
        mlflow.log_param("champion_model", champion)
        mlflow.log_param("fallback_used", fallback_used)
        mlflow.log_metrics({
            "champion_validation_rmse": champ["validation_rmse"],
            "champion_rmse_gap": champ["rmse_gap"],
            "champion_model_size_mb": champ["model_size_mb"],
            "champion_training_time_seconds": champ["training_time_seconds"],
            "champion_prediction_time_ms": champ["prediction_time_ms"],
        })
        mlflow.log_artifact(str(SELECTION_PATH))
        mlflow.log_artifact(str(COMPARISON_CSV_PATH))
        mlflow.log_artifact(str(COMPARISON_MD_PATH))

    print("\n=== Model comparison ===")
    print(comparison.drop(columns=["Status"]).to_string(index=False))
    print(f"\nChampion: {DISPLAY_NAMES[champion]}")
    print(explanation)
    print(f"Champion model saved to: {champion_path}")
    print(f"Selection details: {SELECTION_PATH}")
    print(f"Comparison report: {COMPARISON_CSV_PATH} / {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()