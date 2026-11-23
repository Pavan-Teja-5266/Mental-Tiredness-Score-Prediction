from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"

# Must match the column names/order in mental_tiredness_score_prediction_dataset.csv
FEATURES = [
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
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


st.title("🧠 Mental Tiredness Score Prediction App")

model = load_model()
if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Run `dvc repro` first to generate `models/model.pkl`."
    )
    st.stop()

st.write("Enter daily cognitive, work, sleep, and environment details below:")

col1, col2 = st.columns(2)
with col1:
    number_of_decisions_made = st.number_input(
        "Number of Decisions Made", min_value=82, max_value=165, value=120, step=1
    )
    context_switch_count = st.number_input(
        "Context Switch Count", min_value=0, max_value=21, value=8, step=1
    )
    notifications_received = st.number_input(
        "Notifications Received", min_value=30, max_value=101, value=65, step=1
    )
    screen_time_min = st.number_input(
        "Screen Time (min)", min_value=20.0, max_value=852.2, value=302.0, step=1.0
    )
    deep_work_min = st.number_input(
        "Deep Work (min)", min_value=0.0, max_value=312.8, value=94.8, step=1.0
    )
    task_complexity_avg = st.number_input(
        "Task Complexity Avg", min_value=1.0, max_value=10.0, value=5.5, step=0.1
    )
    caffeine_mg = st.number_input(
        "Caffeine Intake (mg)", min_value=0.0, max_value=422.9, value=129.0, step=1.0
    )
    break_frequency = st.number_input(
        "Break Frequency", min_value=0, max_value=13, value=4, step=1
    )
    mood = st.selectbox(
        "Mood", options=["Happy", "Neutral", "Low"], index=1
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Hours", min_value=3.0, max_value=11.0, value=6.9, step=0.1
    )
    deep_sleep_pct = st.number_input(
        "Deep Sleep (%)", min_value=0.0, max_value=41.38, value=18.98, step=0.1
    )
    hydration_l = st.number_input(
        "Hydration (L)", min_value=0.3, max_value=4.57, value=1.89, step=0.05
    )
    work_type = st.selectbox(
        "Work Type", options=["Remote", "Office", "Manual", "Student"], index=0
    )
    work_environment = st.selectbox(
        "Work Environment", options=["Quiet", "Moderate Noise", "Noisy"], index=1
    )
    noise_level_db = st.number_input(
        "Noise Level (dB)", min_value=20.0, max_value=84.8, value=47.9, step=0.5
    )
    temperature_c = st.number_input(
        "Temperature (°C)", min_value=15.0, max_value=35.0, value=23.0, step=0.5
    )
    workload_score = st.number_input(
        "Workload Score", min_value=1.0, max_value=10.0, value=5.7, step=0.1
    )

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[
            number_of_decisions_made,
            context_switch_count,
            notifications_received,
            screen_time_min,
            deep_work_min,
            task_complexity_avg,
            caffeine_mg,
            break_frequency,
            sleep_hours,
            deep_sleep_pct,
            hydration_l,
            mood,
            work_type,
            work_environment,
            noise_level_db,
            temperature_c,
            workload_score,
        ]],
        columns=FEATURES,
    )

    prediction = float(model.predict(input_data)[0])
    st.success(f"Predicted Mental Tiredness Score: **{prediction:.2f}** (out of ~76.74)")