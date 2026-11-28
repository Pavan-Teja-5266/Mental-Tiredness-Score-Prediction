from pathlib import Path

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


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Run your training pipeline locally and commit models/model.pkl."
        )
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error("Failed to load the prediction model.")
    st.exception(exc)
    st.stop()

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