from pathlib import Path
import pickle
import subprocess

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

st.set_page_config(
    page_title="Mental Tiredness Score Prediction",
    page_icon="🧠",
    layout="centered",
)

if not MODEL_PATH.exists():
    with st.spinner("Running DVC pipeline (dvc repro)..."):
        subprocess.run(["dvc", "repro"], cwd=BASE_DIR, check=True)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🧠 Mental Tiredness Score Prediction App")
st.write("Enter your daily routine, workload, and context details below:")

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
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
sleep_quality = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)
screen_time_hours = st.number_input("Screen Time (Hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
work_hours = st.number_input("Work Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
breaks_taken = st.number_input("Breaks Taken", min_value=0, value=2, step=1)
physical_activity_mins = st.number_input("Physical Activity (Minutes)", min_value=0, value=30, step=5)
caffeine_intake = st.number_input("Caffeine Intake (Cups / Servings)", min_value=0, value=1, step=1)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=4)
focus_level = st.slider("Focus Level (1-10)", min_value=1, max_value=10, value=7)
meetings_count = st.number_input("Meetings Count", min_value=0, value=2, step=1)
interruptions_count = st.number_input("Interruptions Count", min_value=0, value=3, step=1)
hydration_glasses = st.number_input("Hydration (Glasses of Water)", min_value=0, value=8, step=1)
workload_score = st.slider("Workload Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Predict Mental Tiredness Score"):
    input_data = pd.DataFrame(
        {
            "number_of_decisions_made": [number_of_decisions_made],
            "sleep_hours": [sleep_hours],
            "sleep_quality": [sleep_quality],
            "screen_time_hours": [screen_time_hours],
            "work_hours": [work_hours],
            "breaks_taken": [breaks_taken],
            "physical_activity_mins": [physical_activity_mins],
            "caffeine_intake": [caffeine_intake],
            "stress_level": [stress_level],
            "focus_level": [focus_level],
            "meetings_count": [meetings_count],
            "interruptions_count": [interruptions_count],
            "hydration_glasses": [hydration_glasses],
            "mood": [mood],
            "work_type": [work_type],
            "work_environment": [work_environment],
            "workload_score": [workload_score],
        }
    )

    prediction = float(model.predict(input_data)[0])
    st.success(f"Predicted Mental Tiredness Score: {prediction:.2f}")