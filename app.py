from pathlib import Path
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
for path_str in (str(BASE_DIR), str(SRC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

FEATURE_COLUMNS_FALLBACK = [
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


def _load_drift_module():
    drift_file = SRC_DIR / "monitor_drift.py"
    if drift_file.exists():
        spec = importlib.util.spec_from_file_location("monitor_drift", str(drift_file))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


_drift_mod = _load_drift_module()

FEATURE_COLUMNS = getattr(_drift_mod, "FEATURE_COLUMNS", FEATURE_COLUMNS_FALLBACK)
DEFAULT_RAW_DATA = getattr(
    _drift_mod,
    "DEFAULT_RAW_DATA",
    BASE_DIR / "data" / "raw" / "mental_tiredness_score_prediction_dataset.csv",
)
DEFAULT_SAMPLE_DIR = getattr(_drift_mod, "DEFAULT_SAMPLE_DIR", BASE_DIR / "data" / "monitoring")
run_drift_monitoring = getattr(_drift_mod, "run_drift_monitoring", None)
generate_sample_datasets = getattr(_drift_mod, "generate_sample_datasets", None)

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
SELECTION_PATH = BASE_DIR / "reports" / "model_selection.json"
COMPARISON_CSV_PATH = BASE_DIR / "reports" / "model_comparison.csv"
REFERENCE_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "train.csv"
    if (BASE_DIR / "data" / "processed" / "train.csv").exists()
    else (
        BASE_DIR / "data" / "processed" / "X_test.csv"
        if (BASE_DIR / "data" / "processed" / "X_test.csv").exists()
        else BASE_DIR / "data" / "raw" / "mental_tiredness_score_prediction_dataset.csv"
    )
)
INFERENCE_LOG_PATH = BASE_DIR / "data" / "monitoring" / "inference_logs.csv"
DRIFT_JSON_PATH = BASE_DIR / "reports" / "drift_report.json"

st.set_page_config(
    page_title="Mental Tiredness Predictor & MLOps Monitor",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_champion_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_tiredness(score: float) -> tuple[str, str]:
    if score < 40.0:
        return "Low Tiredness", "Good recovery and low cognitive strain."
    if score < 80.0:
        return "Moderate Tiredness", "Noticeable daily workload; take regular breaks."
    return "High / Severe Tiredness", "Elevated cognitive load or recovery deficit; prioritize rest."


def log_inference(input_df: pd.DataFrame, prediction: float, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = input_df[FEATURE_COLUMNS].copy()
    record["predicted_mental_tiredness_score"] = round(float(prediction), 4)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, record], ignore_index=True)
    else:
        combined = record
    combined.to_csv(log_path, index=False)


def main():
    st.title("🧠 Mental Tiredness Score Prediction & MLOps Dashboard")
    st.caption(
        "Predict mental tiredness score using the champion pipeline, inspect model "
        "evaluation artifacts, and monitor feature and prediction drift."
    )

    model = load_champion_model(MODEL_PATH)
    if model is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. Commit `models/model.pkl` "
            "or run your training pipeline first."
        )
        st.stop()

    tab_predict, tab_model, tab_drift = st.tabs(
        ["🔮 Tiredness Prediction", "📊 Model & Selection", "📉 Data Drift"]
    )

    with tab_predict:
        st.subheader("Daily Routine & Environment Inputs")
        col1, col2, col3 = st.columns(3)

        with col1:
            mood = st.selectbox(
                "Mood",
                ["Calm", "Neutral", "Stressed", "Anxious", "Tired", "Energized"],
            )
            work_type = st.selectbox(
                "Work Type",
                ["Focused / Deep Work", "Meetings / Collaboration", "Routine / Admin", "Mixed"],
            )
            work_environment = st.selectbox(
                "Work Environment",
                ["Quiet Home / Office", "Noisy / Open Office", "Hybrid / Travel"],
            )
            number_of_decisions_made = st.number_input("Decisions Made", min_value=0, value=15, step=1)
            context_switch_count = st.number_input("Context Switch Count", min_value=0, value=10, step=1)
            notifications_received = st.number_input("Notifications Received", min_value=0, value=45, step=5)

        with col2:
            screen_time_min = st.number_input("Screen Time (min)", min_value=0.0, value=360.0, step=15.0)
            deep_work_min = st.number_input("Deep Work (min)", min_value=0.0, value=120.0, step=15.0)
            task_complexity_avg = st.slider("Avg Task Complexity (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
            caffeine_mg = st.number_input("Caffeine (mg)", min_value=0.0, value=100.0, step=25.0)
            break_frequency = st.number_input("Break Frequency", min_value=0, value=3, step=1)
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

        with col3:
            deep_sleep_pct = st.slider("Deep Sleep (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            hydration_l = st.number_input("Hydration (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.25)
            noise_level_db = st.number_input("Noise Level (dB)", min_value=0.0, max_value=130.0, value=45.0, step=5.0)
            temperature_c = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=24.0, step=0.5)
            workload_score = st.slider("Workload Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

        log_to_csv = st.checkbox("Log prediction to inference monitoring CSV", value=True)

        if st.button("Predict Mental Tiredness Score", type="primary"):
            input_df = pd.DataFrame(
                [
                    {
                        "number_of_decisions_made": number_of_decisions_made,
                        "context_switch_count": context_switch_count,
                        "notifications_received": notifications_received,
                        "screen_time_min": screen_time_min,
                        "deep_work_min": deep_work_min,
                        "task_complexity_avg": task_complexity_avg,
                        "caffeine_mg": caffeine_mg,
                        "break_frequency": break_frequency,
                        "sleep_hours": sleep_hours,
                        "deep_sleep_pct": deep_sleep_pct,
                        "hydration_l": hydration_l,
                        "mood": mood,
                        "work_type": work_type,
                        "work_environment": work_environment,
                        "noise_level_db": noise_level_db,
                        "temperature_c": temperature_c,
                        "workload_score": workload_score,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            pred_score = float(model.predict(input_df)[0])
            band, guidance = classify_tiredness(pred_score)

            if pred_score >= 80.0:
                st.warning(f"Predicted Mental Tiredness Score: **{pred_score:.2f}** ({band})")
            else:
                st.success(f"Predicted Mental Tiredness Score: **{pred_score:.2f}** ({band})")
            st.caption(guidance)

            if log_to_csv:
                log_inference(input_df, pred_score, log_path=INFERENCE_LOG_PATH)
                st.caption(f"Logged inference row to `{INFERENCE_LOG_PATH}`.")

    with tab_model:
        metrics = load_json_if_exists(METRICS_PATH)
        selection = load_json_if_exists(SELECTION_PATH)

        if selection and "champion" in selection:
            champ = selection["champion"]
            st.subheader(f"Champion Model: {champ.get('display_name', champ.get('model_name', 'Ridge'))}")
            if champ.get("explanation"):
                st.write(champ["explanation"])
        else:
            st.subheader("Champion Model: Ridge Regression")

        if metrics:
            st.markdown("### Evaluation Metrics")
            cols = st.columns(min(4, max(1, len(metrics))))
            for idx, (k, v) in enumerate(list(metrics.items())[:4]):
                if isinstance(v, (int, float)):
                    cols[idx].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        else:
            st.info("`reports/metrics.json` not found.")

        if COMPARISON_CSV_PATH.exists():
            st.markdown("### Candidate Model Comparison")
            st.dataframe(pd.read_csv(COMPARISON_CSV_PATH), use_container_width=True)

    with tab_drift:
        st.subheader("Current vs. Reference Drift Check")
        if run_drift_monitoring is None or generate_sample_datasets is None:
            st.warning(
                "`src/monitor_drift.py` was not found in the deployment environment. "
                "Push `src/monitor_drift.py` to your repository and reboot the app."
            )
        else:
            action_col1, action_col2, action_col3 = st.columns(3)

            normal_sample_path = DEFAULT_SAMPLE_DIR / "normal_traffic.csv"
            drift_sample_path = DEFAULT_SAMPLE_DIR / "simulated_drift.csv"

            options = [
                path
                for path in [
                    str(normal_sample_path) if normal_sample_path.exists() else None,
                    str(drift_sample_path) if drift_sample_path.exists() else None,
                    str(INFERENCE_LOG_PATH) if INFERENCE_LOG_PATH.exists() else None,
                ]
                if path is not None
            ] or [str(normal_sample_path)]

            if "drift_batch_select" not in st.session_state or st.session_state["drift_batch_select"] not in options:
                st.session_state["drift_batch_select"] = options[0]

            with action_col1:
                if st.button("Generate Healthy Sample Logs"):
                    source_csv = REFERENCE_DATA_PATH if REFERENCE_DATA_PATH.exists() else DEFAULT_RAW_DATA
                    normal_p, drift_p = generate_sample_datasets(source_csv, output_dir=DEFAULT_SAMPLE_DIR)
                    for p in (str(normal_p), str(drift_p)):
                        if p not in options:
                            options.insert(0, p)
                    st.session_state["drift_batch_select"] = str(normal_p)
                    run_drift_monitoring(
                        reference_path=REFERENCE_DATA_PATH,
                        current_path=normal_p,
                        model_path=MODEL_PATH,
                        report_path=DRIFT_JSON_PATH,
                    )

            with action_col2:
                if st.button("Simulate Drifted Traffic"):
                    source_csv = REFERENCE_DATA_PATH if REFERENCE_DATA_PATH.exists() else DEFAULT_RAW_DATA
                    normal_p, drift_p = generate_sample_datasets(source_csv, output_dir=DEFAULT_SAMPLE_DIR)
                    for p in (str(normal_p), str(drift_p)):
                        if p not in options:
                            options.insert(0, p)
                    st.session_state["drift_batch_select"] = str(drift_p)
                    run_drift_monitoring(
                        reference_path=REFERENCE_DATA_PATH,
                        current_path=drift_p,
                        model_path=MODEL_PATH,
                        report_path=DRIFT_JSON_PATH,
                    )

            with action_col3:
                run_check = st.button("Run Drift Analysis Now", type="primary")

            batch_choice = st.selectbox(
                "Batch Dataset to Evaluate",
                options=options,
                key="drift_batch_select",
            )

            if run_check:
                if not REFERENCE_DATA_PATH.exists():
                    st.error(f"Reference dataset missing at `{REFERENCE_DATA_PATH}`.")
                else:
                    target_batch = Path(batch_choice)
                    if not target_batch.exists():
                        source_csv = REFERENCE_DATA_PATH
                        generate_sample_datasets(source_csv, output_dir=DEFAULT_SAMPLE_DIR)
                        target_batch = normal_sample_path

                    run_drift_monitoring(
                        reference_path=REFERENCE_DATA_PATH,
                        current_path=target_batch,
                        model_path=MODEL_PATH,
                        report_path=DRIFT_JSON_PATH,
                    )

            drift_report = load_json_if_exists(DRIFT_JSON_PATH)
            if drift_report:
                overall = drift_report.get("overall_status", "OK")
                if overall == "MAJOR_DRIFT":
                    st.error(f"**Status:** 🚨 {overall}")
                elif overall == "MINOR_DRIFT":
                    st.warning(f"**Status:** ⚠️ {overall}")
                else:
                    st.success(f"**Status:** ✅ {overall}")

                st.caption(
                    f"Reference rows: **{drift_report.get('reference_rows', 0)}** | "
                    f"Current rows: **{drift_report.get('current_rows', 0)}**"
                )

                drifted_list = drift_report.get("drifted_features", [])
                st.write(
                    f"Drifted features: **{len(drifted_list)}** "
                    f"({', '.join(drifted_list) if drifted_list else 'None'})"
                )

                if drift_report.get("prediction_drift"):
                    pd_info = drift_report["prediction_drift"]
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Prediction PSI", f"{pd_info.get('psi', 0.0):.4f}")
                    p2.metric("KS p-value", f"{pd_info.get('ks_pvalue', 0.0):.4f}")
                    p3.metric("Ref Mean Score", f"{pd_info.get('ref_pred_mean', 0.0):.2f}")
                    p4.metric("Cur Mean Score", f"{pd_info.get('cur_pred_mean', 0.0):.2f}")

                feat_rows = []
                for feat_name, info in drift_report.get("numeric_features", {}).items():
                    feat_rows.append(
                        {
                            "Feature": feat_name,
                            "Type": "numeric",
                            "PSI": info.get("psi", 0.0),
                            "KS p-value": info.get("ks_pvalue", 0.0),
                            "Ref Mean": info.get("ref_mean"),
                            "Cur Mean": info.get("cur_mean"),
                            "Status": info.get("status", "OK"),
                        }
                    )
                for feat_name, info in drift_report.get("categorical_features", {}).items():
                    feat_rows.append(
                        {
                            "Feature": feat_name,
                            "Type": "categorical",
                            "PSI": info.get("psi", 0.0),
                            "KS p-value": None,
                            "Ref Mean": None,
                            "Cur Mean": None,
                            "Status": info.get("status", "OK"),
                        }
                    )
                if feat_rows:
                    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
            else:
                st.info("No drift report generated yet. Click **Run Drift Analysis Now** above.")


if __name__ == "__main__":
    main()