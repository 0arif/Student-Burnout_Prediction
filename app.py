import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# Load Model
# ==========================

model_data = joblib.load("models/burnout_model.pkl")

model = model_data["model"]

feature_names = model_data["feature_names"]


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Student Burnout Prediction",
    page_icon="🎓",
    layout="centered"
)

with st.sidebar:

    st.title("🎓 Student Burnout")

    st.markdown("---")

    st.write("### Model")

    st.info("Random Forest Classifier")

    st.write("### Features")

    st.write("""
- Study Hours
- Sleep Hours
- Attendance
- Assignment Load
- Social Media
- Exercise
- Stress Level
""")

    st.markdown("---")

    st.success("AI/ML University Project")

st.title("🎓 Student Burnout Prediction")

st.markdown("""
Predict the burnout risk of students using a **Machine Learning (Random Forest)** model.
Adjust the student information below and click the prediction button.
""")

st.info("This application is developed for educational purposes using Machine Learning.")

st.divider()

# ==========================
# User Inputs
# ==========================

study_hours = st.slider("Study Hours (per day)", 1, 12, 6)

sleep_hours = st.slider("Sleep Hours (per day)", 3, 10, 7)

attendance = st.slider("Attendance (%)", 50, 100, 85)

assignment_load = st.slider("Assignment Load", 0, 8, 3)

social_media_hours = st.slider("Social Media Hours", 0, 10, 3)

exercise_hours = st.slider("Exercise Hours", 0, 3, 1)

stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

# ==========================
# Prediction
# ==========================

if st.button("🔍 Predict Burnout Risk",
    use_container_width=True):

    input_data = pd.DataFrame([[
        study_hours,
        sleep_hours,
        attendance,
        assignment_load,
        social_media_hours,
        exercise_hours,
        stress_level
    ]], columns=[
        "Study_Hours",
        "Sleep_Hours",
        "Attendance",
        "Assignment_Load",
        "Social_Media_Hours",
        "Exercise_Hours",
        "Stress_Level"
    ])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    labels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    risk = labels[prediction]

    st.subheader("Prediction Result")

    if risk == "Low":
        st.success(f"✅ Burnout Risk: {risk}")

    elif risk == "Medium":
        st.warning(f"⚠️ Burnout Risk: {risk}")

    else:
        st.error(f"🚨 Burnout Risk: {risk}")

    st.divider()

    st.subheader("Prediction Confidence")

    st.progress(int(probabilities[0] * 100))
    st.write(f"Low : {probabilities[0] * 100:.2f}%")

    st.progress(int(probabilities[1] * 100))
    st.write(f"Medium : {probabilities[1] * 100:.2f}%")

    st.progress(int(probabilities[2] * 100))
    st.write(f"High : {probabilities[2] * 100:.2f}%")

    st.divider()
    st.subheader("Recommendations")

    if risk == "Low":
        st.write("✔ Keep maintaining a healthy study-life balance.")
        st.write("✔ Continue regular exercise.")
        st.write("✔ Maintain good sleeping habits.")

    elif risk == "Medium":
        st.write("✔ Take short breaks while studying.")
        st.write("✔ Reduce social media usage.")
        st.write("✔ Sleep at least 7 hours daily.")
        st.write("✔ Exercise regularly.")

    else:
        st.write("✔ Reduce study overload.")
        st.write("✔ Sleep at least 7–8 hours.")
        st.write("✔ Limit social media usage.")
        st.write("✔ Exercise every day.")
        st.write("✔ Talk to a mentor or counselor if stress continues.")

    st.divider()

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel("Importance")

    ax.set_ylabel("Feature")

    ax.set_title("Feature Importance")

    ax.invert_yaxis()

    plt.tight_layout()

    st.pyplot(fig)    

st.divider()

st.subheader("About This Project")

st.write("""
This project predicts student burnout risk using Machine Learning.

Algorithm:
- Random Forest Classifier

Target:
- Low
- Medium
- High

Total Features:
- 7 Student Lifestyle Features
""")

st.divider()

st.caption(
    "Developed by Arifur Rahman | Department of ICE | AI/ML Project"
)