import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REF_DATA = BASE_DIR / "data" / "processed" / "X_test.csv"
DEFAULT_RAW_DATA = BASE_DIR / "data" / "raw" / "mental_tiredness_score_prediction_dataset.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "drift_report.json"
DEFAULT_SAMPLE_DIR = BASE_DIR / "data" / "monitoring"

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

CATEGORICAL_COLUMNS = ["mood", "work_type", "work_environment"]
NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def _classify_psi(psi_value: float) -> str:
    if psi_value >= 0.25:
        return "MAJOR_DRIFT"
    if psi_value >= 0.10:
        return "MINOR_DRIFT"
    return "OK"


def _compute_numeric_psi(ref_series: pd.Series, cur_series: pd.Series, bins: int = 10) -> float:
    ref_vals = pd.to_numeric(ref_series, errors="coerce").dropna().to_numpy(dtype=float)
    cur_vals = pd.to_numeric(cur_series, errors="coerce").dropna().to_numpy(dtype=float)

    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    edges = np.unique(np.percentile(ref_vals, np.linspace(0.0, 100.0, bins + 1)))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_vals, bins=edges)
    cur_counts, _ = np.histogram(cur_vals, bins=edges)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(len(ref_vals), 1), eps, 1.0)
    cur_pct = np.clip(cur_counts / max(len(cur_vals), 1), eps, 1.0)

    return max(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 0.0)


def _compute_categorical_drift(ref_series: pd.Series, cur_series: pd.Series) -> dict[str, Any]:
    ref_dist = ref_series.fillna("Missing").astype(str).value_counts(normalize=True)
    cur_dist = cur_series.fillna("Missing").astype(str).value_counts(normalize=True)

    all_cats = sorted(set(ref_dist.index) | set(cur_dist.index))
    eps = 1e-6
    ref_probs = np.array([max(float(ref_dist.get(c, 0.0)), eps) for c in all_cats], dtype=float)
    cur_probs = np.array([max(float(cur_dist.get(c, 0.0)), eps) for c in all_cats], dtype=float)

    psi = max(float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs))), 0.0)
    tv_distance = float(0.5 * np.sum(np.abs(cur_probs - ref_probs)))

    shifts = [
        {
            "category": cat,
            "ref_share": round(float(ref_dist.get(cat, 0.0)), 4),
            "cur_share": round(float(cur_dist.get(cat, 0.0)), 4),
            "delta": round(float(cur_dist.get(cat, 0.0) - ref_dist.get(cat, 0.0)), 4),
        }
        for cat in all_cats
    ]
    shifts.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "psi": round(psi, 4),
        "tv_distance": round(tv_distance, 4),
        "status": _classify_psi(psi),
        "top_category_shifts": shifts[:5],
    }


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in '{path}': {missing}")
    aligned = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        aligned[col] = aligned[col].fillna("Unknown").astype(str)
    return aligned


def generate_sample_datasets(source_path: Path, output_dir: Path = DEFAULT_SAMPLE_DIR) -> tuple[Path, Path]:
    base_df = _load_dataset(source_path)
    sample_n = min(200, len(base_df))

    normal_df = base_df.sample(n=sample_n, random_state=42).copy()

    drifted_df = base_df.sample(n=sample_n, random_state=99).copy()
    drifted_df["context_switch_count"] = drifted_df["context_switch_count"] * 2.2 + 12
    drifted_df["notifications_received"] = drifted_df["notifications_received"] * 2.0 + 40
    drifted_df["screen_time_min"] = drifted_df["screen_time_min"] * 1.4 + 90
    drifted_df["workload_score"] = np.clip(drifted_df["workload_score"] * 1.35 + 20, 0, 100)
    drifted_df["sleep_hours"] = np.clip(drifted_df["sleep_hours"] - 2.0, 1.0, 12.0)
    drifted_df["mood"] = "Stressed"
    drifted_df["work_environment"] = "Noisy / Open Office"

    output_dir.mkdir(parents=True, exist_ok=True)
    normal_path = output_dir / "normal_traffic.csv"
    drift_path = output_dir / "simulated_drift.csv"
    normal_df.to_csv(normal_path, index=False)
    drifted_df.to_csv(drift_path, index=False)
    return normal_path, drift_path


def run_drift_monitoring(
    reference_path: Path,
    current_path: Path,
    model_path: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    ref_df = _load_dataset(reference_path)
    cur_df = _load_dataset(current_path)

    numeric_report: dict[str, Any] = {}
    drifted_features: list[str] = []

    for col in NUMERIC_COLUMNS:
        ref_s = ref_df[col].dropna()
        cur_s = cur_df[col].dropna()
        psi_val = _compute_numeric_psi(ref_s, cur_s)
        ks_stat, ks_pvalue = (
            stats.ks_2samp(ref_s.to_numpy(dtype=float), cur_s.to_numpy(dtype=float))
            if len(ref_s) > 0 and len(cur_s) > 0
            else (0.0, 1.0)
        )
        status = _classify_psi(psi_val)
        if status != "OK":
            drifted_features.append(col)

        numeric_report[col] = {
            "psi": round(psi_val, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "ref_mean": round(float(ref_s.mean()), 4) if len(ref_s) else None,
            "cur_mean": round(float(cur_s.mean()), 4) if len(cur_s) else None,
            "mean_shift": round(float(cur_s.mean() - ref_s.mean()), 4) if len(ref_s) and len(cur_s) else None,
            "status": status,
        }

    categorical_report: dict[str, Any] = {}
    for col in CATEGORICAL_COLUMNS:
        cat_res = _compute_categorical_drift(ref_df[col], cur_df[col])
        if cat_res["status"] != "OK":
            drifted_features.append(col)
        categorical_report[col] = cat_res

    prediction_drift: dict[str, Any] | None = None
    if model_path and model_path.exists():
        model = joblib.load(model_path)
        ref_preds = np.asarray(model.predict(ref_df[FEATURE_COLUMNS]), dtype=float)
        cur_preds = np.asarray(model.predict(cur_df[FEATURE_COLUMNS]), dtype=float)
        pred_psi = _compute_numeric_psi(pd.Series(ref_preds), pd.Series(cur_preds))
        pred_ks_stat, pred_ks_pval = stats.ks_2samp(ref_preds, cur_preds)
        pred_status = _classify_psi(pred_psi)

        prediction_drift = {
            "psi": round(pred_psi, 4),
            "ks_stat": round(float(pred_ks_stat), 4),
            "ks_pvalue": round(float(pred_ks_pval), 6),
            "ref_pred_mean": round(float(np.mean(ref_preds)), 4),
            "cur_pred_mean": round(float(np.mean(cur_preds)), 4),
            "pred_mean_shift": round(float(np.mean(cur_preds) - np.mean(ref_preds)), 4),
            "status": pred_status,
        }

    statuses = [v["status"] for v in numeric_report.values()] + [
        v["status"] for v in categorical_report.values()
    ]
    if prediction_drift:
        statuses.append(prediction_drift["status"])

    if "MAJOR_DRIFT" in statuses:
        overall_status = "MAJOR_DRIFT"
    elif "MINOR_DRIFT" in statuses:
        overall_status = "MINOR_DRIFT"
    else:
        overall_status = "OK"

    report = {
        "overall_status": overall_status,
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_df)),
        "drifted_features": drifted_features,
        "numeric_features": numeric_report,
        "categorical_features": categorical_report,
        "prediction_drift": prediction_drift,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    default_ref = DEFAULT_REF_DATA if DEFAULT_REF_DATA.exists() else DEFAULT_RAW_DATA
    parser = argparse.ArgumentParser(description="Monitor data and prediction drift for mental tiredness model.")
    parser.add_argument("--reference-data", type=Path, default=default_ref, help="Path to reference CSV")
    parser.add_argument("--current-data", type=Path, default=None, help="Path to new/current CSV to evaluate")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained model.pkl")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Output JSON report path")
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate normal_traffic.csv and simulated_drift.csv and run normal traffic check",
    )
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Generate simulated_drift.csv and run drift check against it",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if overall_status is MAJOR_DRIFT",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_for_samples = args.reference_data if args.reference_data.exists() else DEFAULT_RAW_DATA
    normal_path, drift_path = (
        generate_sample_datasets(source_for_samples)
        if (args.generate_sample_data or args.simulate_drift)
        else (DEFAULT_SAMPLE_DIR / "normal_traffic.csv", DEFAULT_SAMPLE_DIR / "simulated_drift.csv")
    )

    if args.current_data is not None:
        current_path = args.current_data
    elif args.simulate_drift:
        current_path = drift_path
    else:
        current_path = normal_path

    report = run_drift_monitoring(
        reference_path=args.reference_data,
        current_path=current_path,
        model_path=args.model_path,
        report_path=args.report_path,
    )

    print(f"Overall Drift Status: {report['overall_status']}")
    print(f"Reference rows: {report['reference_rows']} | Current rows: {report['current_rows']}")
    print(f"Drifted features: {report['drifted_features'] or 'None'}")
    if report.get("prediction_drift"):
        pd_info = report["prediction_drift"]
        print(
            f"Prediction Drift -> PSI: {pd_info['psi']} | "
            f"Ref Mean: {pd_info['ref_pred_mean']} | Cur Mean: {pd_info['cur_pred_mean']} | "
            f"Status: {pd_info['status']}"
        )
    print(f"Saved drift report to: {args.report_path}")

    if args.strict and report["overall_status"] == "MAJOR_DRIFT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())