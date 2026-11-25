import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
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

st.set_page_config(
    page_title="Mental Tiredness Score Prediction",
    page_icon="🧠",
    layout="centered",
)


def _load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def run_dvc_pipeline(force: bool = False) -> None:
    python_exe = sys.executable
    venv_bin = str(Path(python_exe).parent)

    dvc_tmp = BASE_DIR / ".dvc" / "tmp"
    if dvc_tmp.exists():
        shutil.rmtree(dvc_tmp, ignore_errors=True)

    for folder in (BASE_DIR / "data" / "ingested", BASE_DIR / "data" / "processed"):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        shim_path = Path(tmp_dir) / "python"
        shim_path.write_text(f'#!/usr/bin/env sh\nexec "{python_exe}" "$@"\n', encoding="utf-8")
        shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR)

        mlruns_dir = Path(tmp_dir) / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PATH"] = f"{tmp_dir}{os.pathsep}{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(Path(python_exe).parent.parent)
        env["DVC_NO_ANALYTICS"] = "1"
        env["MLFLOW_TRACKING_URI"] = mlruns_dir.as_uri()

        dvc_bin = Path(python_exe).parent / "dvc"
        base_cmd = (
            [str(dvc_bin), "repro"]
            if dvc_bin.exists()
            else ([shutil.which("dvc"), "repro"] if shutil.which("dvc") else [python_exe, "-m", "dvc", "repro"])
        )
        cmd = [*base_cmd, "--force"] if force else base_cmd

        try:
            subprocess.run(
                cmd,
                cwd=BASE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            st.error(f"DVC pipeline failed (exit code {exc.returncode}).")
            if exc.stdout:
                st.code(exc.stdout, language="text")
            raise


if not MODEL_PATH.exists():
    with st.spinner("Running DVC pipeline (dvc repro)..."):
        run_dvc_pipeline()

try:
    model = _load_model(MODEL_PATH)
except Exception:
    if MODEL_PATH.exists():
        MODEL_PATH.unlink(missing_ok=True)
    with st.spinner("Regenerating model via DVC pipeline..."):
        run_dvc_pipeline(force=True)
    model = _load_model(MODEL_PATH)

st.title("🧠 Mental Tiredness Score Prediction App")
st.write("Enter your daily routine, workload, and environment details below:")

mood = st.selectbox(
    "Mood",
    options=["Calm", "Neutral", "Stressed", "Anxious", "Tired", "Energized"],
)
work_type = st.selectbox(
    "Work Type",
    options=["Focused / Deep Work", "Meetings / Collaboration", "Routine / Admin", "Mixed"],
)
work_environment = st.selectbox(
    "Work Environment",
    options=["Quiet Home / Office", "Noisy / Open Office", "Hybrid / Travel"],
)

number_of_decisions_made = st.number_input("Number of Decisions Made", min_value=0, value=15, step=1)
context_switch_count = st.number_input("Context Switch Count", min_value=0, value=10, step=1)
notifications_received = st.number_input("Notifications Received", min_value=0, value=45, step=5)
screen_time_min = st.number_input("Screen Time (Minutes)", min_value=0.0, value=360.0, step=15.0)
deep_work_min = st.number_input("Deep Work (Minutes)", min_value=0.0, value=120.0, step=15.0)
task_complexity_avg = st.slider("Average Task Complexity (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
caffeine_mg = st.number_input("Caffeine Intake (mg)", min_value=0.0, value=100.0, step=25.0)
break_frequency = st.number_input("Break Frequency (Breaks / Day)", min_value=0, value=3, step=1)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
deep_sleep_pct = st.slider("Deep Sleep (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
hydration_l = st.number_input("Hydration (Liters)", min_value=0.0, max_value=10.0, value=2.5, step=0.25)
noise_level_db = st.number_input("Noise Level (dB)", min_value=0.0, max_value=130.0, value=45.0, step=5.0)
temperature_c = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=24.0, step=0.5)
workload_score = st.slider("Workload Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

if st.button("Predict Mental Tiredness Score"):
    input_data = pd.DataFrame(
        {
            "number_of_decisions_made": [number_of_decisions_made],
            "context_switch_count": [context_switch_count],
            "notifications_received": [notifications_received],
            "screen_time_min": [screen_time_min],
            "deep_work_min": [deep_work_min],
            "task_complexity_avg": [task_complexity_avg],
            "caffeine_mg": [caffeine_mg],
            "break_frequency": [break_frequency],
            "sleep_hours": [sleep_hours],
            "deep_sleep_pct": [deep_sleep_pct],
            "hydration_l": [hydration_l],
            "mood": [mood],
            "work_type": [work_type],
            "work_environment": [work_environment],
            "noise_level_db": [noise_level_db],
            "temperature_c": [temperature_c],
            "workload_score": [workload_score],
        }
    )[FEATURE_COLUMNS]

    prediction = float(model.predict(input_data)[0])
    st.success(f"Predicted Mental Tiredness Score: {prediction:.2f}")